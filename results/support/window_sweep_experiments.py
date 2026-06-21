"""Core experiment helpers shared by results entry points."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.evaluation.trainer import evaluate_model, prepare_data, train_model
from src.models.architectures import build_model_kwargs, get_model
from src.models.inference import autoregressive_forecast

from .window_sweep_config import (
    DROPOUT,
    FIXED_WINDOW,
    HIDDEN_DIM,
    MAX_HORIZON,
    MODEL_NAMES,
    N_CUTOFFS,
    NUM_LAYERS,
    STEP_MINUTES,
    TARGET_COL,
    TRAIN_RATIO,
    WINDOW_SIZES,
)
from .window_sweep_models import (
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


def train_and_score_single_model_window(
    df: pd.DataFrame,
    model_name: str,
    window_size: int,
    device: str,
    epochs: int,
    patience: int,
) -> tuple[dict, object, dict]:
    """Train one model/window configuration and return metrics, model, and data."""
    data = prepare_model_inputs(df, window_size)

    saved_model, saved_metrics = load_saved_model_window(model_name, data["input_dim"], window_size)
    if saved_model is not None:
        print(f"  Loaded saved model {trained_model_label(model_name, window_size)}")
        return saved_metrics or {}, saved_model, data

    kwargs = build_model_kwargs(model_name, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    model = get_model(model_name, input_dim=data["input_dim"], **kwargs)

    history = train_model(
        model,
        data["train_loader"],
        val_loader=data["test_loader"],
        epochs=epochs,
        lr=1e-3,
        patience=patience,
        device=device,
    )

    _, _, metrics = evaluate_model(model, data["test_loader"], data["target_scaler"], device=device)
    save_trained_model_window(
        model,
        model_name,
        metrics,
        history,
        data["input_dim"],
        data["feature_names"],
        window_size,
    )
    return metrics, model, data


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


def _save_study_results_cache(results: dict, cache_path: Path) -> None:
    """Persist one study-results cache to disk."""
    with open(cache_path, "w") as handle:
        json.dump(results, handle, indent=2)


def _train_models_for_window_size(
    df: pd.DataFrame,
    model_names: list[str],
    window_size: int,
    device: str,
    epochs: int,
    patience: int,
) -> dict[str, dict]:
    """Train all models for one lookback window and return their runs."""
    model_runs: dict[str, dict] = {}
    for model_name in model_names:
        metrics, model, data = train_and_score_single_model_window(
            df, model_name, window_size, device, epochs, patience,
        )
        model_runs[model_name] = {
            "metrics": metrics,
            "model": model,
            "data": data,
        }
    return model_runs


def _store_one_step_window_metrics(results: dict, model_runs: dict[str, dict], window_size: int) -> None:
    """Write one-step metrics for one lookback window into the results dictionary."""
    window_key = str(window_size)
    for model_name, run in model_runs.items():
        results.setdefault(model_name, {})
        results[model_name][window_key] = run["metrics"]


def _lookback_window_is_cached(results: dict, model_names: list[str], window_size: int) -> bool:
    """Return whether all model metrics for one lookback window already exist."""
    window_key = str(window_size)
    return all(results.get(model_name, {}).get(window_key) is not None for model_name in model_names)


def _train_fixed_window_models(
    df: pd.DataFrame,
    model_names: list[str],
    fixed_window: int,
    device: str,
    epochs: int,
    patience: int,
) -> dict[str, dict]:
    """Train all models once at the fixed window used for horizon evaluation."""
    model_runs: dict[str, dict] = {}
    for model_name in model_names:
        print(f"  Training {model_name}  w={fixed_window} for horizon sweep ... ", end="", flush=True)
        started = time.time()
        try:
            _, model, data = train_and_score_single_model_window(
                df, model_name, fixed_window, device, epochs, patience,
            )
        except Exception as exc:
            print(f"FAILED to train - {exc}")
            continue
        print(f"trained ({time.time() - started:.0f}s)")
        model_runs[model_name] = {"model": model, "data": data}
    return model_runs


def _measure_horizon_profile(
    model,
    data: dict,
    df: pd.DataFrame,
    fixed_window: int,
    max_horizon: int,
    n_cutoffs: int,
    device: str,
) -> dict:
    """Measure rollout error by horizon step for one trained model."""
    split_idx = int(len(df) * TRAIN_RATIO)
    test_start = split_idx + fixed_window
    test_end = len(df) - max_horizon
    cutoff_indices = np.linspace(test_start, test_end, n_cutoffs, dtype=int)
    errors = _collect_autoregressive_error_by_horizon(
        model,
        df,
        data["feature_names"],
        data["scaler"],
        data["target_scaler"],
        fixed_window,
        max_horizon,
        cutoff_indices,
        device,
    )
    return {
        "mean_mae_per_step": np.nanmean(errors, axis=1).tolist(),
        "std_mae_per_step": np.nanstd(errors, axis=1).tolist(),
    }


def run_one_step_lookback_study(
    df: pd.DataFrame,
    device: str,
    epochs: int,
    patience: int,
    cache_path: Path,
    rebuild_cache: bool = False,
) -> dict:
    """Resume or run the one-step lookback study and update its JSON cache."""
    results = _load_study_results_cache(cache_path, rebuild_cache)
    done = sum(
        1
        for model_name in MODEL_NAMES
        for window_size in WINDOW_SIZES
        if results.get(model_name, {}).get(str(window_size)) is not None
    )
    if done > 0:
        print(f"  Resuming: {done}/{len(MODEL_NAMES) * len(WINDOW_SIZES)} configs already cached")

    for window_size in WINDOW_SIZES:
        if _lookback_window_is_cached(results, MODEL_NAMES, window_size):
            print(f"  [skip] all models at w={window_size:>2}  (cached)")
            continue

        label = f"{window_size} steps ({window_size * STEP_MINUTES} min)"
        print(f"  Training all models at w={window_size:>2}  ({label})")
        try:
            model_runs = _train_models_for_window_size(
                df, MODEL_NAMES, window_size, device, epochs, patience,
            )
            _store_one_step_window_metrics(results, model_runs, window_size)
        except Exception as exc:
            print(f"  FAILED at w={window_size:>2} - {exc}")
            for model_name in MODEL_NAMES:
                results.setdefault(model_name, {})[str(window_size)] = None

        _save_study_results_cache(results, cache_path)

    results.setdefault("Persistence", {})
    split_idx = int(len(df) * TRAIN_RATIO)
    for window_size in WINDOW_SIZES:
        window_key = str(window_size)
        if window_key not in results["Persistence"]:
            results["Persistence"][window_key] = persistence_metrics_1step(df, split_idx, window_size)

    _save_study_results_cache(results, cache_path)
    return results


def run_multistep_horizon_study(
    df: pd.DataFrame,
    device: str,
    epochs: int,
    patience: int,
    cache_path: Path,
    rebuild_cache: bool = False,
) -> dict:
    """Resume or run the multistep horizon study and update its JSON cache."""
    results = _load_study_results_cache(cache_path, rebuild_cache)
    if results:
        print(f"  Resuming from cached horizon results ({cache_path.name})")

    split_idx = int(len(df) * TRAIN_RATIO)
    test_start = split_idx + FIXED_WINDOW
    test_end = len(df) - MAX_HORIZON

    if test_end <= test_start:
        raise ValueError(
            f"Dataset too small for a {MAX_HORIZON}-step horizon sweep "
            f"with window {FIXED_WINDOW}. Reduce MAX_HORIZON or FIXED_WINDOW."
        )

    cutoff_indices = np.linspace(test_start, test_end, N_CUTOFFS, dtype=int)

    if "Persistence" not in results:
        print("  Computing persistence baseline ... ", end="", flush=True)
        p_errors = persistence_horizon_errors(df, split_idx, FIXED_WINDOW, cutoff_indices, MAX_HORIZON)
        results["Persistence"] = {
            "mean_mae_per_step": np.nanmean(p_errors, axis=1).tolist(),
            "std_mae_per_step": np.nanstd(p_errors, axis=1).tolist(),
        }
        print("done")
        _save_study_results_cache(results, cache_path)

    model_runs = _train_fixed_window_models(df, MODEL_NAMES, FIXED_WINDOW, device, epochs, patience)

    for model_name in MODEL_NAMES:
        if model_name in results:
            print(f"  [skip] {model_name} horizon sweep (cached)")
            continue

        if model_name not in model_runs:
            continue

        print(f"  Evaluating horizon profile for {model_name} ... ", end="", flush=True)
        started = time.time()
        run = model_runs[model_name]
        results[model_name] = _measure_horizon_profile(
            run["model"],
            run["data"],
            df,
            FIXED_WINDOW,
            MAX_HORIZON,
            N_CUTOFFS,
            device,
        )
        print(f"done ({time.time() - started:.0f}s)")
        _save_study_results_cache(results, cache_path)

    return results