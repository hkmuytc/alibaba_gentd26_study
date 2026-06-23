"""Core experiment helpers shared by results entry points."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.evaluation.trainer import evaluate_model, prepare_data, prepare_data_for_temporal_blocks, train_model
from src.models.inference import autoregressive_forecast

from .window_sweep_config import (
    DROPOUT,
    FIXED_WINDOW,
    HIDDEN_DIM,
    MAX_HORIZON,
    MODEL_NAMES,
    N_CUTOFFS,
    NUM_LAYERS,
    ROLLING_ORIGIN_TEST_STEPS,
    ROLLING_ORIGIN_TRAIN_RATIOS,
    ROLLING_ORIGIN_VAL_STEPS,
    STEP_MINUTES,
    TARGET_COL,
    TRAIN_RATIO,
    WINDOW_SIZES,
)
from .window_sweep_models import (
    build_window_model,
    load_saved_model_window,
    save_trained_model_window,
    trained_model_label,
)


def persistence_metrics_1step(df: pd.DataFrame, split_idx: int, window_size: int) -> dict:
    """Compute the naive one-step persistence baseline on the test set."""
    targets = df[TARGET_COL].values
    test_start = split_idx + window_size
    if test_start >= len(targets):
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}

    actuals = targets[test_start:]
    preds = targets[test_start - 1: test_start - 1 + len(actuals)]
    n = min(len(actuals), len(preds))
    actuals, preds = actuals[:n], preds[:n]

    mae = float(mean_absolute_error(actuals, preds))
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    mask = actuals != 0
    mape = (
        float(np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100)
        if mask.any() else float("nan")
    )
    ss_res = float(np.sum((actuals - preds) ** 2))
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def persistence_metrics_for_target_indices(df: pd.DataFrame, target_indices: np.ndarray) -> dict:
    """Compute one-step persistence metrics on an explicit set of target indices."""
    if len(target_indices) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}

    actuals = df[TARGET_COL].values[target_indices]
    preds = df[TARGET_COL].values[target_indices - 1]

    mae = float(mean_absolute_error(actuals, preds))
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    mask = actuals != 0
    mape = (
        float(np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100)
        if mask.any() else float("nan")
    )
    ss_res = float(np.sum((actuals - preds) ** 2))
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def persistence_horizon_errors(
    df: pd.DataFrame,
    split_idx: int,
    window_size: int,
    cutoff_indices: np.ndarray,
    max_horizon: int,
) -> np.ndarray:
    """Return per-step absolute errors for the persistence baseline."""
    targets = df[TARGET_COL].values
    errors = np.full((max_horizon, len(cutoff_indices)), np.nan)

    for col_idx, cutoff_idx in enumerate(cutoff_indices):
        last_known = targets[cutoff_idx - 1]
        for horizon_idx in range(max_horizon):
            future_idx = cutoff_idx + horizon_idx
            if future_idx < len(targets):
                errors[horizon_idx, col_idx] = abs(targets[future_idx] - last_known)

    return errors


def prepare_model_inputs(df: pd.DataFrame, window_size: int) -> dict:
    """Build scaled sliding-window model inputs for one lookback length."""
    return prepare_data(df, target_col=TARGET_COL, window_size=window_size, train_ratio=TRAIN_RATIO)


def prepare_model_inputs_for_origin(df, window_size: int, train_end: int, val_end: int, test_end: int) -> dict:
    """Build scaled train/validation/test loaders for one rolling forecast origin."""
    return prepare_data_for_temporal_blocks(
        df,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        target_col=TARGET_COL,
        window_size=window_size,
    )


def train_model_on_window_data(
    model,
    data: dict,
    device: str,
    epochs: int,
    patience: int,
) -> dict:
    """Train one already-constructed model on one prepared window dataset."""
    return train_model(
        model,
        data["train_loader"],
        val_loader=data.get("val_loader") or data["test_loader"],
        epochs=epochs,
        lr=1e-3,
        patience=patience,
        device=device,
    )


def evaluate_one_step_model(model, data: dict, device: str) -> dict:
    """Evaluate one trained model on the held-out one-step test split."""
    _, _, metrics = evaluate_model(model, data["test_loader"], data["target_scaler"], device=device)
    return metrics


def load_or_train_single_model_window(
    data: dict,
    model_name: str,
    window_size: int,
    device: str,
    epochs: int,
    patience: int,
) -> dict:
    """Return one window-specific trained model run, loading a saved artifact when possible."""
    saved_run = load_saved_model_window(model_name, data["input_dim"], data["feature_names"], window_size)
    if saved_run is not None:
        print(f"  Loaded saved model {trained_model_label(model_name, window_size)}")
        saved_run["data"] = data
        saved_run["source"] = "saved"
        return saved_run

    model = build_window_model(model_name, data["input_dim"])
    history = train_model_on_window_data(model, data, device, epochs, patience)
    metrics = evaluate_one_step_model(model, data, device)
    save_trained_model_window(
        model,
        model_name,
        metrics,
        history,
        data["input_dim"],
        data["feature_names"],
        window_size,
    )
    return {
        "model": model,
        "metrics": metrics,
        "history": history,
        "data": data,
        "source": "trained",
    }


def train_fresh_single_model_window(
    data: dict,
    model_name: str,
    window_size: int,
    device: str,
    epochs: int,
    patience: int,
) -> dict:
    """Train one window-specific model from scratch for a single rolling-origin fold."""
    model = build_window_model(model_name, data["input_dim"])
    history = train_model_on_window_data(model, data, device, epochs, patience)
    metrics = evaluate_one_step_model(model, data, device)
    save_trained_model_window(
        model,
        model_name,
        metrics,
        history,
        data["input_dim"],
        data["feature_names"],
        window_size,
    )
    return {
        "model": model,
        "metrics": metrics,
        "history": history,
        "data": data,
        "source": "trained",
    }


def _collect_autoregressive_error_by_horizon(
    model,
    df: pd.DataFrame,
    feature_names: list[str],
    scaler,
    target_scaler,
    window_size: int,
    max_horizon: int,
    cutoff_indices: np.ndarray,
    device: str,
) -> np.ndarray:
    """Run autoregressive forecasts and collect absolute error by forecast step."""
    targets = df[TARGET_COL].values
    errors = np.full((max_horizon, len(cutoff_indices)), np.nan)

    for col_idx, cutoff_idx in enumerate(cutoff_indices):
        avail = min(max_horizon, len(df) - cutoff_idx)
        if avail < 1:
            continue
        try:
            preds = autoregressive_forecast(
                model,
                df,
                feature_names,
                TARGET_COL,
                scaler,
                target_scaler,
                window_size,
                int(cutoff_idx),
                avail,
                device,
            )
            actuals = targets[cutoff_idx: cutoff_idx + avail]
            n = min(len(preds), len(actuals))
            errors[:n, col_idx] = np.abs(actuals[:n] - preds[:n])
        except Exception:
            pass

    return errors


def _load_study_results_cache(cache_path: Path, rebuild_cache: bool) -> dict:
    """Load one JSON cache or start from an empty results dictionary."""
    if rebuild_cache and cache_path.exists():
        cache_path.unlink()
        print(f"  Removed cached study results: {cache_path.name}")

    if not cache_path.exists():
        return {}

    with open(cache_path) as handle:
        return json.load(handle)


def _reset_legacy_one_step_cache_if_needed(results: dict, cache_path: Path) -> dict:
    """Discard the pre-rolling-origin one-step cache format when detected.

    The legacy format stored metrics directly under each window key, e.g.
    ``results['GRU']['24']['MAE']``. The rolling-origin format stores per-fold
    metrics under each window key, e.g. ``results['GRU']['24']['fold_1']``.
    """
    for model_name in MODEL_NAMES:
        model_results = results.get(model_name, {})
        if not isinstance(model_results, dict):
            continue
        for window_key, value in model_results.items():
            if not isinstance(value, dict):
                continue
            if any(metric_name in value for metric_name in ("MAE", "RMSE", "MAPE", "R2")):
                print(f"  Resetting legacy one-step cache: {cache_path.name}")
                cache_path.unlink(missing_ok=True)
                return {}
    return results


def _reset_legacy_horizon_cache_if_needed(results: dict, cache_path: Path) -> dict:
    """Discard the pre-rolling-origin horizon cache format when detected.

    The legacy format stored a horizon profile directly under each model key,
    whereas the rolling-origin format stores one horizon profile per fold.
    """
    for model_name, value in results.items():
        if not isinstance(value, dict):
            continue
        if "mean_mae_per_step" in value or "std_mae_per_step" in value:
            print(f"  Resetting legacy horizon cache: {cache_path.name}")
            cache_path.unlink(missing_ok=True)
            return {}
    return results


def _save_study_results_cache(results: dict, cache_path: Path) -> None:
    """Persist one study-results cache to disk."""
    with open(cache_path, "w") as handle:
        json.dump(results, handle, indent=2)


def _train_models_for_window_size(
    data: dict,
    model_names: list[str],
    window_size: int,
    device: str,
    epochs: int,
    patience: int,
) -> dict[str, dict]:
    """Train all models from scratch for one lookback window on one origin split."""
    model_runs: dict[str, dict] = {}
    for model_name in model_names:
        run = train_fresh_single_model_window(
            data, model_name, window_size, device, epochs, patience,
        )
        model_runs[model_name] = run
    return model_runs


def _store_one_step_window_metrics(results: dict, model_runs: dict[str, dict], window_size: int) -> None:
    """Write one-step metrics for one lookback window into the results dictionary."""
    window_key = str(window_size)
    for model_name, run in model_runs.items():
        results.setdefault(model_name, {})
        results[model_name][window_key] = run["metrics"]


def _record_one_step_fold_metrics(results: dict, fold_key: str, window_size: int, model_runs: dict[str, dict]) -> None:
    """Store one-step metrics for one rolling-origin fold and one window size."""
    window_key = str(window_size)
    for model_name, run in model_runs.items():
        results.setdefault(model_name, {}).setdefault(window_key, {})[fold_key] = run["metrics"]


def _lookback_window_is_cached(results: dict, model_names: list[str], window_size: int) -> bool:
    """Return whether all model metrics for one lookback window already exist."""
    window_key = str(window_size)
    return all(results.get(model_name, {}).get(window_key) is not None for model_name in model_names)


def _build_rolling_origins(df, val_steps: int = ROLLING_ORIGIN_VAL_STEPS, test_steps: int = ROLLING_ORIGIN_TEST_STEPS) -> list[dict]:
    """Return chronological train/validation/test boundaries for several forecast origins."""
    origins = []
    n_rows = len(df)
    for fold_idx, train_ratio in enumerate(ROLLING_ORIGIN_TRAIN_RATIOS, start=1):
        train_end = int(n_rows * train_ratio)
        val_end = train_end + val_steps
        test_end = val_end + test_steps
        if train_end <= 0 or test_end > n_rows:
            continue
        origins.append({
            "fold_key": f"fold_{fold_idx}",
            "train_ratio": train_ratio,
            "train_end": train_end,
            "val_end": val_end,
            "test_end": test_end,
        })
    return origins


def _print_origin_plan(origins: list[dict]) -> None:
    """Print the rolling-origin plan so the split structure is explicit."""
    print("  Rolling-origin evaluation plan:")
    for origin in origins:
        print(
            f"    {origin['fold_key']}: train_end={origin['train_end']}, "
            f"val_end={origin['val_end']}, test_end={origin['test_end']}"
        )


def _aggregate_one_step_fold_metrics(fold_results: dict, window_sizes: list[int]) -> dict:
    """Average one-step metrics across rolling-origin folds for each window size."""
    aggregated: dict = {"Persistence": {}}
    for model_name in MODEL_NAMES:
        aggregated[model_name] = {}
        for window_size in window_sizes:
            fold_metrics = list(fold_results.get(model_name, {}).get(str(window_size), {}).values())
            if not fold_metrics:
                aggregated[model_name][str(window_size)] = None
                continue
            aggregated[model_name][str(window_size)] = {
                metric_name: float(np.mean([fold_metric[metric_name] for fold_metric in fold_metrics]))
                for metric_name in fold_metrics[0]
            }
    return aggregated


def _record_persistence_one_step_fold(results: dict, fold_key: str, df, split_idx: int, window_size: int) -> None:
    """Store one-step persistence metrics for one fold/window pair."""
    window_key = str(window_size)
    metrics = persistence_metrics_1step(df, split_idx, window_size)
    results.setdefault("Persistence", {}).setdefault(window_key, {})[fold_key] = metrics


def _record_persistence_one_step_fold_for_targets(
    results: dict,
    fold_key: str,
    df: pd.DataFrame,
    window_size: int,
    target_indices: np.ndarray,
) -> None:
    """Store one-step persistence metrics for one fold/window pair on explicit test indices."""
    window_key = str(window_size)
    metrics = persistence_metrics_for_target_indices(df, target_indices)
    results.setdefault("Persistence", {}).setdefault(window_key, {})[fold_key] = metrics


def _aggregate_persistence_one_step_folds(aggregated_results: dict, fold_results: dict, window_sizes: list[int]) -> None:
    """Average one-step persistence metrics across folds for each window size."""
    for window_size in window_sizes:
        window_key = str(window_size)
        fold_metrics = list(fold_results.get("Persistence", {}).get(window_key, {}).values())
        if not fold_metrics:
            aggregated_results["Persistence"][window_key] = None
            continue
        aggregated_results["Persistence"][window_key] = {
            metric_name: float(np.mean([fold_metric[metric_name] for fold_metric in fold_metrics]))
            for metric_name in fold_metrics[0]
        }


def _record_horizon_fold_metrics(results: dict, fold_key: str, model_name: str, metrics: dict) -> None:
    """Store one horizon profile for one model and one fold."""
    results.setdefault(model_name, {})[fold_key] = metrics


def _aggregate_horizon_fold_metrics(fold_results: dict) -> dict:
    """Average horizon profiles across rolling-origin folds for each model."""
    aggregated: dict = {}
    for model_name, model_fold_results in fold_results.items():
        fold_values = list(model_fold_results.values())
        if not fold_values:
            continue
        mean_stack = np.asarray([row["mean_mae_per_step"] for row in fold_values], dtype=np.float64)
        std_stack = np.asarray([row["std_mae_per_step"] for row in fold_values], dtype=np.float64)
        aggregated[model_name] = {
            "mean_mae_per_step": np.nanmean(mean_stack, axis=0).tolist(),
            "std_mae_per_step": np.nanmean(std_stack, axis=0).tolist(),
        }
    return aggregated


def _measure_persistence_horizon_profile(df, split_idx: int, fixed_window: int, max_horizon: int, n_cutoffs: int) -> dict:
    """Measure the persistence baseline by forecast horizon for one fold."""
    test_start = split_idx + fixed_window
    test_end = len(df) - max_horizon
    cutoff_indices = np.linspace(test_start, test_end, n_cutoffs, dtype=int)
    p_errors = persistence_horizon_errors(df, split_idx, fixed_window, cutoff_indices, max_horizon)
    return {
        "mean_mae_per_step": np.nanmean(p_errors, axis=1).tolist(),
        "std_mae_per_step": np.nanstd(p_errors, axis=1).tolist(),
    }


def _origin_cutoff_indices(origin: dict, max_horizon: int, *, full_horizon_only: bool) -> np.ndarray:
    """Return forecast cutoffs drawn from one origin's unseen test block."""
    test_start = origin["val_end"]
    test_end = origin["test_end"]
    if full_horizon_only:
        last_start = test_end - max_horizon
        if last_start < test_start:
            return np.empty((0,), dtype=int)
        return np.arange(test_start, last_start + 1, dtype=int)
    return np.arange(test_start, test_end, dtype=int)


def _measure_persistence_horizon_profile_for_origin(df, cutoff_indices: np.ndarray, max_horizon: int) -> dict:
    """Measure the persistence baseline by horizon using origin-specific unseen cutoffs."""
    p_errors = persistence_horizon_errors(df, 0, 1, cutoff_indices, max_horizon)
    return {
        "mean_mae_per_step": np.nanmean(p_errors, axis=1).tolist(),
        "std_mae_per_step": np.nanstd(p_errors, axis=1).tolist(),
    }


def _train_fixed_window_models(
    df: pd.DataFrame,
    model_names: list[str],
    fixed_window: int,
    device: str,
    epochs: int,
    patience: int,
) -> dict[str, dict]:
    """Train all models from scratch at the fixed window used for horizon evaluation."""
    model_runs: dict[str, dict] = {}
    for model_name in model_names:
        print(f"  Training {model_name}  w={fixed_window} for horizon sweep ... ", end="", flush=True)
        started = time.time()
        try:
            run = train_fresh_single_model_window(
                df, model_name, fixed_window, device, epochs, patience,
            )
        except Exception as exc:
            print(f"FAILED to train - {exc}")
            continue
        print(f"trained ({time.time() - started:.0f}s)")
        model_runs[model_name] = run
    return model_runs


def _measure_horizon_profile(
    model,
    data: dict,
    df: pd.DataFrame,
    cutoff_indices: np.ndarray,
    max_horizon: int,
    device: str,
) -> dict:
    """Measure rollout error by horizon step for one trained model."""
    errors = _collect_autoregressive_error_by_horizon(
        model,
        df,
        data["feature_names"],
        data["scaler"],
        data["target_scaler"],
        data["window_size"],
        max_horizon,
        cutoff_indices,
        device,
    )
    return {
        "mean_mae_per_step": np.nanmean(errors, axis=1).tolist(),
        "std_mae_per_step": np.nanstd(errors, axis=1).tolist(),
    }


def _evaluate_replay_window_against_persistence(
    model,
    data: dict,
    df: pd.DataFrame,
    cutoff_idx: int,
    horizon: int,
    device: str,
) -> dict | None:
    """Evaluate one fixed-horizon autoregressive replay window against persistence."""
    try:
        preds = autoregressive_forecast(
            model,
            df,
            data["feature_names"],
            TARGET_COL,
            data["scaler"],
            data["target_scaler"],
            data["window_size"],
            int(cutoff_idx),
            horizon,
            device,
        )
    except Exception:
        return None

    actuals = df[TARGET_COL].values[cutoff_idx: cutoff_idx + horizon]
    if len(actuals) < horizon:
        return None

    baseline = np.full(horizon, df[TARGET_COL].values[cutoff_idx - 1])
    model_mae = float(mean_absolute_error(actuals, preds))
    baseline_mae = float(mean_absolute_error(actuals, baseline))
    return {
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "win": model_mae < baseline_mae,
    }


def _summarize_replay_windows(rows: list[dict]) -> dict:
    """Summarize fixed-horizon replay windows against persistence."""
    if not rows:
        return {
            "avg_mae": np.nan,
            "baseline_mae": np.nan,
            "improvement_pct": np.nan,
            "win_rate": np.nan,
            "n_windows": 0,
        }

    model_maes = np.asarray([row["model_mae"] for row in rows], dtype=np.float64)
    baseline_maes = np.asarray([row["baseline_mae"] for row in rows], dtype=np.float64)
    avg_model = float(np.mean(model_maes))
    avg_baseline = float(np.mean(baseline_maes))
    improvement_pct = float((avg_baseline - avg_model) / avg_baseline * 100) if avg_baseline != 0 else float("nan")
    win_rate = float(np.mean([1.0 if row["win"] else 0.0 for row in rows]) * 100)
    return {
        "avg_mae": avg_model,
        "baseline_mae": avg_baseline,
        "improvement_pct": improvement_pct,
        "win_rate": win_rate,
        "n_windows": len(rows),
    }


def run_fixed_horizon_replay_summary(
    df: pd.DataFrame,
    device: str,
    epochs: int,
    patience: int,
    horizon: int,
) -> dict:
    """Run a fixed-horizon replay comparison against persistence across rolling origins."""
    origins = _build_rolling_origins(df)
    summary: dict = {}
    for origin in origins:
        fold_key = origin["fold_key"]
        summary[fold_key] = {}
        data = prepare_model_inputs_for_origin(
            df,
            FIXED_WINDOW,
            train_end=origin["train_end"],
            val_end=origin["val_end"],
            test_end=origin["test_end"],
        )
        cutoff_indices = _origin_cutoff_indices(origin, horizon, full_horizon_only=True)
        model_runs = _train_fixed_window_models(data, MODEL_NAMES, FIXED_WINDOW, device, epochs, patience)
        for model_name in MODEL_NAMES:
            if model_name not in model_runs:
                continue
            rows = []
            run = model_runs[model_name]
            for cutoff_idx in cutoff_indices:
                row = _evaluate_replay_window_against_persistence(
                    run["model"],
                    run["data"],
                    df,
                    int(cutoff_idx),
                    horizon,
                    device,
                )
                if row is not None:
                    rows.append(row)
            summary[fold_key][model_name] = _summarize_replay_windows(rows)
    return summary


def run_one_step_lookback_study(
    df: pd.DataFrame,
    device: str,
    epochs: int,
    patience: int,
    cache_path: Path,
    rebuild_cache: bool = False,
) -> dict:
    """Run one-step lookback evaluation across several rolling forecast origins."""
    fold_results = _load_study_results_cache(cache_path, rebuild_cache)
    fold_results = _reset_legacy_one_step_cache_if_needed(fold_results, cache_path)
    origins = _build_rolling_origins(df)
    _print_origin_plan(origins)

    for origin in origins:
        fold_key = origin["fold_key"]
        print(f"\n  [{fold_key}] one-step lookback evaluation")
        for window_size in WINDOW_SIZES:
            cached_metrics = fold_results.get(MODEL_NAMES[0], {}).get(str(window_size), {}).get(fold_key)
            if cached_metrics is not None:
                print(f"    [skip] w={window_size:>2}  ({fold_key} cached)")
                continue

            data = prepare_model_inputs_for_origin(
                df,
                window_size,
                train_end=origin["train_end"],
                val_end=origin["val_end"],
                test_end=origin["test_end"],
            )
            print(f"    training all models at w={window_size:>2}")
            model_runs = _train_models_for_window_size(
                data,
                MODEL_NAMES,
                window_size,
                device,
                epochs,
                patience,
            )
            _record_one_step_fold_metrics(fold_results, fold_key, window_size, model_runs)
            _record_persistence_one_step_fold_for_targets(
                fold_results,
                fold_key,
                df,
                window_size,
                data["test_target_indices"],
            )
            _save_study_results_cache(fold_results, cache_path)

    aggregated_results = _aggregate_one_step_fold_metrics(fold_results, WINDOW_SIZES)
    _aggregate_persistence_one_step_folds(aggregated_results, fold_results, WINDOW_SIZES)
    return aggregated_results


def run_multistep_horizon_study(
    df: pd.DataFrame,
    device: str,
    epochs: int,
    patience: int,
    cache_path: Path,
    rebuild_cache: bool = False,
) -> dict:
    """Run multistep horizon evaluation across several rolling forecast origins."""
    fold_results = _load_study_results_cache(cache_path, rebuild_cache)
    fold_results = _reset_legacy_horizon_cache_if_needed(fold_results, cache_path)
    origins = _build_rolling_origins(df)
    _print_origin_plan(origins)

    for origin in origins:
        fold_key = origin["fold_key"]
        print(f"\n  [{fold_key}] multistep horizon evaluation")
        cutoff_indices = _origin_cutoff_indices(origin, MAX_HORIZON, full_horizon_only=False)

        if fold_key not in fold_results.get("Persistence", {}):
            fold_results.setdefault("Persistence", {})[fold_key] = _measure_persistence_horizon_profile_for_origin(
                df,
                cutoff_indices,
                MAX_HORIZON,
            )
            _save_study_results_cache(fold_results, cache_path)

        data = prepare_model_inputs_for_origin(
            df,
            FIXED_WINDOW,
            train_end=origin["train_end"],
            val_end=origin["val_end"],
            test_end=origin["test_end"],
        )
        model_runs = _train_fixed_window_models(
            data,
            MODEL_NAMES,
            FIXED_WINDOW,
            device,
            epochs,
            patience,
        )

        for model_name in MODEL_NAMES:
            if fold_key in fold_results.get(model_name, {}):
                print(f"    [skip] {model_name}  ({fold_key} cached)")
                continue
            if model_name not in model_runs:
                continue

            print(f"    evaluating horizon profile for {model_name} ... ", end="", flush=True)
            started = time.time()
            run = model_runs[model_name]
            metrics = _measure_horizon_profile(
                run["model"],
                run["data"],
                df,
                cutoff_indices,
                MAX_HORIZON,
                device,
            )
            _record_horizon_fold_metrics(fold_results, fold_key, model_name, metrics)
            print(f"done ({time.time() - started:.0f}s)")
            _save_study_results_cache(fold_results, cache_path)

    return _aggregate_horizon_fold_metrics(fold_results)