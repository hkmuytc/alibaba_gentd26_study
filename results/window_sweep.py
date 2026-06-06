#!/usr/bin/env python3
"""Window Sweep Study
====================
Systematically measures how two key sliding-window hyperparameters affect
forecasting accuracy across all three model architectures (LSTM, GRU,
Transformer).

Experiment 1 — Lookback sweep
    Fix the forecast target to 1 step ahead.  Vary the lookback window size
    (number of history steps the model sees) across WINDOW_SIZES.  Each
    (model, window_size) pair is trained from scratch and evaluated on the
    held-out test set.  Reveals whether more history helps and whether there
    is a sweet spot beyond which extra context starts to hurt.

Experiment 2 — Horizon sweep
    Fix the lookback window at FIXED_WINDOW (default 24 steps = 2 hours).
    Roll each model autoregressively forward for 1 to MAX_HORIZON steps,
    starting from N_CUTOFFS evenly-spaced points in the test region.
    Average the per-step absolute error across all cutoffs.  Reveals how
    quickly error compounds as we predict further into the future, and which
    architecture degrades most gracefully.

Outputs (written to results/window_sweep/):
    lookback_results.json   per-model metrics at each window size
    horizon_results.json    per-model mean + std MAE at each horizon step
    fig_lookback.png        2×2 grid — MAE / RMSE / MAPE / R² vs window size
    fig_horizon.png         MAE vs forecast horizon with ±1 std error band
    fig_heatmap.png         MAE heatmap — models (rows) × window sizes (cols)

Usage:
    # Run both experiments then plot (recommended first run)
    python results/window_sweep.py

    # Only one experiment
    python results/window_sweep.py --lookback-only
    python results/window_sweep.py --horizon-only

    # Regenerate plots without re-running (requires saved JSON)
    python results/window_sweep.py --plot-only

    # Smoke-test with fewer epochs
    python results/window_sweep.py --quick

    # Different dataset (must have a processed CSV in data/processed/)
    python results/window_sweep.py --dataset gpu_v2020

Results are cached to JSON after every training run.  Re-running the script
will skip any (model, window_size) pair that already has a result, so it is
safe to interrupt and resume.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

# ── Path setup ────────────────────────────────────────────────────────────────
# This file lives at  <project_root>/results/window_sweep.py
# Project root is therefore one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.loading import load_processed_datasets       # noqa: E402
from src.evaluation.trainer import prepare_data, train_model, evaluate_model  # noqa: E402
from src.models.architectures import get_model, build_model_kwargs, MODEL_REGISTRY  # noqa: E402
from src.models.inference import autoregressive_forecast               # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration — edit these to change the sweep range
# ─────────────────────────────────────────────────────────────────────────────

# Experiment 1: lookback window sizes to evaluate (in time steps)
WINDOW_SIZES: list[int] = [4, 8, 12, 16, 24, 36, 48, 72]

# Experiment 2: fixed lookback for the horizon sweep
FIXED_WINDOW: int = 24

# Experiment 2: maximum forecast horizon (evaluates steps 1 … MAX_HORIZON)
MAX_HORIZON: int = 24

# Number of test-region cutoff points for the horizon sweep
N_CUTOFFS: int = 50

# Architecture shared defaults (same for all window sizes; matches the
# dashboard defaults so results are comparable)
HIDDEN_DIM: int  = 64
NUM_LAYERS: int  = 2
DROPOUT: float   = 0.2

# Data
TARGET_COL:   str   = "power_total_kw"
TRAIN_RATIO:  float = 0.8
STEP_MINUTES: int   = 5          # one time step = 5 minutes

# Training budget (full run)
DEFAULT_EPOCHS:   int = 100
DEFAULT_PATIENCE: int = 10

# Training budget (--quick smoke-test)
QUICK_EPOCHS:   int = 30
QUICK_PATIENCE: int = 5

# Output directory (always relative to this script)
OUTPUT_DIR: Path = Path(__file__).resolve().parent / "window_sweep"

# Colour palette — consistent across all figures
MODEL_COLORS: dict[str, str] = {
    "LSTM":        "#1f77b4",
    "GRU":         "#ff7f0e",
    "Transformer": "#2ca02c",
    "Persistence": "#333333",
}
MODEL_NAMES: list[str] = list(MODEL_REGISTRY.keys())   # ["LSTM", "GRU", "Transformer"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_name: str = "genai") -> pd.DataFrame:
    """Return the processed power-series DataFrame for *dataset_name*.

    Looks for ``data/processed/<dataset_name>_300s.csv`` under PROJECT_ROOT.
    Raises ``FileNotFoundError`` with a helpful message if the file is absent.
    """
    csv_path = PROJECT_ROOT / "data" / "processed" / f"{dataset_name}_300s.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {csv_path}\n"
            "Run the Processing Pipeline in the dashboard first, or:\n"
            "  python -c \"from src.data_processing.pipeline import process_and_save;"
            " process_and_save('genai')\""
        )
    df = pd.read_csv(csv_path)
    print(f"  Loaded '{dataset_name}': {len(df):,} rows × {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Persistence (naive) baselines
# ─────────────────────────────────────────────────────────────────────────────

def persistence_metrics_1step(
    df: pd.DataFrame, split_idx: int, window_size: int
) -> dict:
    """Compute naive 1-step-ahead baseline metrics on the test set.

    Prediction: y_hat[t+1] = y[t]  (repeat last observed value).
    The test set starts at split_idx + window_size (matching TimeSeriesDataset).
    """
    targets = df[TARGET_COL].values
    test_start = split_idx + window_size
    if test_start >= len(targets):
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}

    actuals = targets[test_start:]
    preds   = targets[test_start - 1 : test_start - 1 + len(actuals)]
    n = min(len(actuals), len(preds))
    actuals, preds = actuals[:n], preds[:n]

    mae  = float(mean_absolute_error(actuals, preds))
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
    """Per-step absolute errors for the persistence baseline over the horizon sweep.

    At each cutoff position c, the baseline predicts:
        y_hat[c + h] = y[c - 1]   for all h = 1 … max_horizon

    Returns an array of shape (max_horizon, n_cutoffs).
    """
    targets  = df[TARGET_COL].values
    errors   = np.full((max_horizon, len(cutoff_indices)), np.nan)

    for j, idx in enumerate(cutoff_indices):
        last_known = targets[idx - 1]
        for h in range(max_horizon):
            future_idx = idx + h
            if future_idx < len(targets):
                errors[h, j] = abs(targets[future_idx] - last_known)

    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — Lookback sweep
# ─────────────────────────────────────────────────────────────────────────────

def train_one_config(
    df: pd.DataFrame,
    model_name: str,
    window_size: int,
    device: str,
    epochs: int,
    patience: int,
) -> tuple[dict, object, dict]:
    """Train *model_name* with *window_size* and return (metrics, model, data).

    *data* is the dict returned by ``prepare_data`` and includes the fitted
    scalers and feature names required for the horizon sweep.
    """
    data   = prepare_data(df, target_col=TARGET_COL, window_size=window_size,
                          train_ratio=TRAIN_RATIO)
    kwargs = build_model_kwargs(model_name, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    model  = get_model(model_name, input_dim=data["input_dim"], **kwargs)

    train_model(
        model,
        data["train_loader"],
        val_loader=data["test_loader"],
        epochs=epochs,
        lr=1e-3,
        patience=patience,
        device=device,
    )

    _, _, metrics = evaluate_model(
        model, data["test_loader"], data["target_scaler"], device=device
    )
    return metrics, model, data


def run_lookback_sweep(
    df: pd.DataFrame,
    window_sizes: list[int],
    model_names: list[str],
    device: str,
    epochs: int,
    patience: int,
    cache_path: Path,
) -> dict:
    """Train every (model, window_size) combination and cache results.

    Results structure::

        {
          "LSTM":        {"4": {"MAE": …, "RMSE": …, "MAPE": …, "R2": …}, …},
          "GRU":         {…},
          "Transformer": {…},
          "Persistence": {…},        ← naive 1-step baseline
        }

    Already-computed entries are skipped on re-runs, so the sweep can be
    safely interrupted and resumed.
    """
    # Load cached results so re-runs skip completed configs
    results: dict = {}
    if cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)
        done = sum(
            1 for mn in model_names
            for ws in window_sizes
            if results.get(mn, {}).get(str(ws)) is not None
        )
        print(f"  Resuming: {done}/{len(model_names) * len(window_sizes)} configs already cached")

    for model_name in model_names:
        results.setdefault(model_name, {})
        for ws in window_sizes:
            ws_key = str(ws)
            if ws_key in results[model_name]:
                print(f"  [skip] {model_name}  w={ws:>2}  (cached)")
                continue

            label = f"{ws} steps ({ws * STEP_MINUTES} min)"
            print(f"  Training {model_name}  w={ws:>2}  ({label}) … ", end="", flush=True)
            t0 = time.time()
            try:
                metrics, _, _ = train_one_config(df, model_name, ws, device, epochs, patience)
                results[model_name][ws_key] = metrics
                elapsed = time.time() - t0
                print(
                    f"MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}"
                    f"  R²={metrics['R2']:.3f}  ({elapsed:.0f}s)"
                )
            except Exception as exc:
                print(f"FAILED — {exc}")
                results[model_name][ws_key] = None

            # Persist after every run so a crash doesn't lose earlier work
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)

    # Persistence baseline (fast, no training)
    results.setdefault("Persistence", {})
    split_idx = int(len(df) * TRAIN_RATIO)
    for ws in window_sizes:
        ws_key = str(ws)
        if ws_key not in results["Persistence"]:
            results["Persistence"][ws_key] = persistence_metrics_1step(df, split_idx, ws)

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Horizon sweep
# ─────────────────────────────────────────────────────────────────────────────

def _collect_horizon_errors(
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
    """Run autoregressive forecasts and collect per-step absolute errors.

    Returns an array of shape (max_horizon, n_cutoffs) where entry [h, j]
    is the absolute error at forecast step h+1 for cutoff j.
    """
    targets = df[TARGET_COL].values
    errors  = np.full((max_horizon, len(cutoff_indices)), np.nan)

    for j, idx in enumerate(cutoff_indices):
        avail = min(max_horizon, len(df) - idx)
        if avail < 1:
            continue
        try:
            preds   = autoregressive_forecast(
                model, df, feature_names, TARGET_COL,
                scaler, target_scaler, window_size,
                int(idx), avail, device,
            )
            actuals = targets[idx : idx + avail]
            n = min(len(preds), len(actuals))
            errors[:n, j] = np.abs(actuals[:n] - preds[:n])
        except Exception:
            pass    # leave as NaN; one bad cutoff doesn't invalidate the sweep

    return errors


def run_horizon_sweep(
    df: pd.DataFrame,
    model_names: list[str],
    fixed_window: int,
    max_horizon: int,
    n_cutoffs: int,
    device: str,
    epochs: int,
    patience: int,
    cache_path: Path,
) -> dict:
    """Train each model at *fixed_window* and evaluate autoregressive MAE by step.

    Results structure::

        {
          "LSTM": {
              "mean_mae_per_step": [mae_h1, mae_h2, …, mae_hN],
              "std_mae_per_step":  [std_h1, std_h2, …, std_hN],
          },
          …
          "Persistence": {…},
        }
    """
    results: dict = {}
    if cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)
        print(f"  Resuming from cached horizon results ({cache_path.name})")

    split_idx  = int(len(df) * TRAIN_RATIO)
    test_start = split_idx + fixed_window
    test_end   = len(df) - max_horizon

    if test_end <= test_start:
        raise ValueError(
            f"Dataset too small for a {max_horizon}-step horizon sweep "
            f"with window {fixed_window}.  Reduce MAX_HORIZON or FIXED_WINDOW."
        )

    # Evenly-spaced cutoff positions inside the test region
    cutoff_indices = np.linspace(test_start, test_end, n_cutoffs, dtype=int)

    # Persistence baseline
    if "Persistence" not in results:
        print("  Computing persistence baseline … ", end="", flush=True)
        p_errors = persistence_horizon_errors(
            df, split_idx, fixed_window, cutoff_indices, max_horizon
        )
        results["Persistence"] = {
            "mean_mae_per_step": np.nanmean(p_errors, axis=1).tolist(),
            "std_mae_per_step":  np.nanstd(p_errors, axis=1).tolist(),
        }
        print("done")
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)

    for model_name in model_names:
        if model_name in results:
            print(f"  [skip] {model_name} horizon sweep (cached)")
            continue

        print(f"  Training {model_name}  w={fixed_window} for horizon sweep … ", end="", flush=True)
        t0 = time.time()
        try:
            _, model, data = train_one_config(
                df, model_name, fixed_window, device, epochs, patience
            )
        except Exception as exc:
            print(f"FAILED to train — {exc}")
            continue
        print(f"trained ({time.time() - t0:.0f}s)  →  evaluating {n_cutoffs} cutoffs … ",
              end="", flush=True)

        t0 = time.time()
        errors = _collect_horizon_errors(
            model, df, data["feature_names"],
            data["scaler"], data["target_scaler"],
            fixed_window, max_horizon, cutoff_indices, device,
        )
        print(f"done ({time.time() - t0:.0f}s)")

        results[model_name] = {
            "mean_mae_per_step": np.nanmean(errors, axis=1).tolist(),
            "std_mae_per_step":  np.nanstd(errors, axis=1).tolist(),
        }
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _window_tick_labels(window_sizes: list[int]) -> list[str]:
    """'24\n(2h)' style tick labels — shows steps and wall-clock time."""
    labels = []
    for ws in window_sizes:
        mins = ws * STEP_MINUTES
        if mins >= 60 and mins % 60 == 0:
            labels.append(f"{ws}\n({mins // 60}h)")
        else:
            labels.append(f"{ws}\n({mins}m)")
    return labels


def _horizon_tick_labels(steps: np.ndarray) -> list[str]:
    """Tick labels for the forecast-horizon axis."""
    labels = []
    for s in steps:
        mins = int(s) * STEP_MINUTES
        if mins >= 60 and mins % 60 == 0:
            labels.append(f"{s}\n({mins // 60}h)")
        else:
            labels.append(f"{s}\n({mins}m)")
    return labels


def plot_lookback_results(
    results: dict, window_sizes: list[int], output_dir: Path
) -> None:
    """2×2 grid: MAE / RMSE / MAPE / R² vs lookback window size.

    One line per model plus the persistence baseline.  A filled circle marks
    the best value for each model.
    """
    metric_specs = [
        ("MAE",  "MAE (kW)",    False),   # (key, y-label, higher_is_better)
        ("RMSE", "RMSE (kW)",   False),
        ("MAPE", "MAPE (%)",    False),
        ("R2",   "R²",          True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Model Performance vs Lookback Window Size\n"
        f"(1-step-ahead forecast · 80/20 split · {STEP_MINUTES}-min time bins)",
        fontsize=13, fontweight="bold",
    )
    axes_flat = axes.flatten()
    ws_arr    = np.array(window_sizes, dtype=float)

    for ax_idx, (key, ylabel, higher_better) in enumerate(metric_specs):
        ax = axes_flat[ax_idx]

        for model_name in MODEL_NAMES + ["Persistence"]:
            if model_name not in results:
                continue

            values = np.array([
                float(results[model_name].get(str(ws), {}).get(key, np.nan)
                      if results[model_name].get(str(ws)) else np.nan)
                for ws in window_sizes
            ])

            is_baseline = model_name == "Persistence"
            color   = MODEL_COLORS.get(model_name, "grey")
            lw      = 1.5 if is_baseline else 2.0
            ls      = "--" if is_baseline else "-"
            label   = "Persistence baseline" if is_baseline else model_name
            zorder  = 2 if is_baseline else 3

            ax.plot(ws_arr, values, color=color, linewidth=lw,
                    linestyle=ls, marker="o" if not is_baseline else None,
                    markersize=5, label=label, zorder=zorder, alpha=0.85)

            # Mark the best point per model (not for baseline)
            if not is_baseline:
                valid = ~np.isnan(values)
                if valid.any():
                    best_idx = (np.nanargmax(values) if higher_better
                                else np.nanargmin(values))
                    ax.scatter(
                        ws_arr[best_idx], values[best_idx],
                        color=color, s=100, zorder=5,
                        edgecolors="black", linewidths=1.2,
                    )

        ax.set_title(key, fontsize=11, fontweight="bold")
        ax.set_xlabel("Lookback window size")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ws_arr)
        ax.set_xticklabels(_window_tick_labels(window_sizes), fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(fontsize=8)

        if key == "R2":
            ax.set_ylabel("R²  (higher = better)")

    plt.tight_layout()
    out_path = output_dir / "fig_lookback.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_horizon_results(
    results: dict, max_horizon: int, output_dir: Path
) -> None:
    """MAE vs forecast horizon for each model with ±1 std shaded band.

    The shaded band shows how error varies across different starting points
    in the test set (not just uncertainty in the mean estimate).
    """
    steps   = np.arange(1, max_horizon + 1)
    minutes = steps * STEP_MINUTES

    # Only label a subset of ticks to avoid crowding
    tick_mask = np.array([s in {1, 2, 3, 4, 6, 8, 12, 18, 24} for s in steps])
    tick_positions = steps[tick_mask]

    fig, ax = plt.subplots(figsize=(11, 6))

    for model_name in MODEL_NAMES + ["Persistence"]:
        if model_name not in results:
            continue

        entry = results[model_name]
        means = np.array(entry["mean_mae_per_step"], dtype=float)
        stds  = np.array(entry["std_mae_per_step"],  dtype=float)

        is_baseline = model_name == "Persistence"
        color  = MODEL_COLORS.get(model_name, "grey")
        lw     = 1.5 if is_baseline else 2.0
        ls     = "--" if is_baseline else "-"
        label  = "Persistence baseline" if is_baseline else model_name
        zorder = 2 if is_baseline else 3

        ax.plot(steps, means, color=color, linewidth=lw, linestyle=ls,
                marker="o" if not is_baseline else None, markersize=4,
                label=label, zorder=zorder)
        ax.fill_between(steps, means - stds, means + stds,
                        color=color, alpha=0.12 if is_baseline else 0.18)

    ax.set_xlabel("Forecast horizon (step ahead)", fontsize=11)
    ax.set_ylabel("MAE (kW)", fontsize=11)
    ax.set_title(
        f"Forecast Error vs Horizon\n"
        f"Lookback = {FIXED_WINDOW} steps ({FIXED_WINDOW * STEP_MINUTES} min)  ·  "
        f"Averaged over {N_CUTOFFS} test cutoffs  ·  Shaded band = ±1 std",
        fontsize=11,
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(_horizon_tick_labels(tick_positions), fontsize=8)
    ax.set_xlim(0.5, max_horizon + 0.5)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=9)

    # Secondary x-axis showing absolute hours
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    hour_steps = [s for s in steps if (s * STEP_MINUTES) % 60 == 0]
    if hour_steps:
        ax2.set_xticks(hour_steps)
        ax2.set_xticklabels(
            [f"{s * STEP_MINUTES // 60}h" for s in hour_steps], fontsize=8
        )
        ax2.set_xlabel("Forecast horizon (hours ahead)", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "fig_horizon.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_heatmap(
    lookback_results: dict, window_sizes: list[int], output_dir: Path
) -> None:
    """MAE heatmap: rows = model architectures, columns = window sizes.

    Green = lower error (better), Red = higher error (worse).
    Cell values are annotated directly in the heatmap.
    """
    n_rows = len(MODEL_NAMES)
    n_cols = len(window_sizes)
    mat    = np.full((n_rows, n_cols), np.nan)

    for i, model_name in enumerate(MODEL_NAMES):
        for j, ws in enumerate(window_sizes):
            entry = lookback_results.get(model_name, {}).get(str(ws))
            if entry and entry.get("MAE") is not None:
                mat[i, j] = float(entry["MAE"])

    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)
    mid  = (vmin + vmax) / 2.0

    fig, ax = plt.subplots(figsize=(max(9, n_cols * 1.3), 3.2))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(_window_tick_labels(window_sizes), fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(MODEL_NAMES, fontsize=10)
    ax.set_xlabel("Lookback window size")
    ax.set_title(
        "MAE (kW) — Model × Lookback Window  "
        "(green = lower error = better)",
        fontsize=11,
    )

    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            if not np.isnan(val):
                text_color = "white" if val > mid else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, label="MAE (kW)", shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = output_dir / "fig_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary tables
# ─────────────────────────────────────────────────────────────────────────────

def print_lookback_summary(results: dict, window_sizes: list[int]) -> None:
    """Print MAE at every window size in a compact table."""
    print("\n" + "═" * 72)
    print("  LOOKBACK SWEEP — test-set MAE (kW)")
    print("  Architecture and window size (steps) — 80/20 train/test, 1 step ahead")
    print("═" * 72)

    header = f"  {'Model':<14}" + "".join(f"  {ws:>5}" for ws in window_sizes)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for model_name in MODEL_NAMES + ["Persistence"]:
        if model_name not in results:
            continue
        row = f"  {model_name:<14}"
        for ws in window_sizes:
            entry = results[model_name].get(str(ws))
            if entry and entry.get("MAE") is not None:
                row += f"  {entry['MAE']:>5.3f}"
            else:
                row += f"  {'—':>5}"
        print(row)

    # Find best window per model
    print()
    for model_name in MODEL_NAMES:
        if model_name not in results:
            continue
        pairs = [
            (ws, results[model_name].get(str(ws), {}) or {})
            for ws in window_sizes
        ]
        valid = [(ws, e["MAE"]) for ws, e in pairs if e.get("MAE") is not None]
        if valid:
            best_ws, best_mae = min(valid, key=lambda x: x[1])
            print(f"  {model_name}: best window = {best_ws} steps "
                  f"({best_ws * STEP_MINUTES} min), MAE = {best_mae:.3f} kW")


def print_horizon_summary(results: dict, max_horizon: int) -> None:
    """Print mean MAE at key forecast horizons."""
    key_steps = [1, 2, 4, 6, 8, 12, 18, 24]
    key_steps = [s for s in key_steps if s <= max_horizon]

    print("\n" + "═" * 72)
    print(f"  HORIZON SWEEP — mean MAE (kW) at key forecast steps")
    print(f"  Fixed lookback = {FIXED_WINDOW} steps ({FIXED_WINDOW * STEP_MINUTES} min)"
          f"  ·  averaged over {N_CUTOFFS} test cutoffs")
    print("═" * 72)

    header = f"  {'Model':<14}" + "".join(
        f"  t+{s:>2}({s * STEP_MINUTES}m)" for s in key_steps
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for model_name in MODEL_NAMES + ["Persistence"]:
        if model_name not in results:
            continue
        means = results[model_name]["mean_mae_per_step"]
        row = f"  {model_name:<14}"
        for s in key_steps:
            idx = s - 1   # 0-indexed
            if idx < len(means):
                row += f"  {means[idx]:>10.3f}"
            else:
                row += f"  {'—':>10}"
        print(row)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Window sweep study — vary lookback size and forecast horizon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--lookback-only", action="store_true",
        help="Run only Experiment 1 (lookback sweep)",
    )
    p.add_argument(
        "--horizon-only", action="store_true",
        help="Run only Experiment 2 (horizon sweep)",
    )
    p.add_argument(
        "--plot-only", action="store_true",
        help="Skip training; regenerate figures from saved JSON",
    )
    p.add_argument(
        "--quick", action="store_true",
        help=f"Smoke-test mode: {QUICK_EPOCHS} epochs, patience {QUICK_PATIENCE}",
    )
    p.add_argument(
        "--dataset", default="genai", metavar="NAME",
        help="Dataset name matching data/processed/<NAME>_300s.csv (default: genai)",
    )
    p.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="PyTorch device (default: auto-detect mps > cuda > cpu)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Device selection
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    epochs   = QUICK_EPOCHS   if args.quick else DEFAULT_EPOCHS
    patience = QUICK_PATIENCE if args.quick else DEFAULT_PATIENCE

    run_lookback = not args.horizon_only
    run_horizon  = not args.lookback_only

    print("═" * 60)
    print("  Window Sweep Study")
    print("═" * 60)
    print(f"  Dataset      : {args.dataset}")
    print(f"  Device       : {device}")
    print(f"  Epochs/run   : {epochs}  (early-stop patience={patience})")
    print(f"  Window sizes : {WINDOW_SIZES}")
    print(f"  Max horizon  : {MAX_HORIZON} steps ({MAX_HORIZON * STEP_MINUTES} min)")
    print(f"  Fixed window : {FIXED_WINDOW} steps (used for horizon sweep)")
    print(f"  Output dir   : {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    if args.quick:
        print("  [quick mode — fewer epochs, for smoke-testing only]")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ─────────────────────────────────────────────────────────
    if not args.plot_only:
        print("[Data]")
        df = load_dataset(args.dataset)

    lookback_cache = OUTPUT_DIR / "lookback_results.json"
    horizon_cache  = OUTPUT_DIR / "horizon_results.json"

    # ── Experiment 1: Lookback sweep ─────────────────────────────────────────
    lookback_results: Optional[dict] = None

    if run_lookback and not args.plot_only:
        n_runs = len(MODEL_NAMES) * len(WINDOW_SIZES)
        print(f"\n[Experiment 1] Lookback sweep  ({n_runs} training runs)")
        lookback_results = run_lookback_sweep(
            df, WINDOW_SIZES, MODEL_NAMES, device, epochs, patience, lookback_cache,
        )
        print_lookback_summary(lookback_results, WINDOW_SIZES)
    elif lookback_cache.exists():
        with open(lookback_cache) as f:
            lookback_results = json.load(f)

    # ── Experiment 2: Horizon sweep ──────────────────────────────────────────
    horizon_results: Optional[dict] = None

    if run_horizon and not args.plot_only:
        print(f"\n[Experiment 2] Horizon sweep  "
              f"(window={FIXED_WINDOW}, horizons 1–{MAX_HORIZON}, "
              f"{N_CUTOFFS} cutoffs each)")
        horizon_results = run_horizon_sweep(
            df, MODEL_NAMES, FIXED_WINDOW, MAX_HORIZON, N_CUTOFFS,
            device, epochs, patience, horizon_cache,
        )
        print_horizon_summary(horizon_results, MAX_HORIZON)
    elif horizon_cache.exists():
        with open(horizon_cache) as f:
            horizon_results = json.load(f)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[Plots]")
    if lookback_results:
        plot_lookback_results(lookback_results, WINDOW_SIZES, OUTPUT_DIR)
        plot_heatmap(lookback_results, WINDOW_SIZES, OUTPUT_DIR)
    else:
        print("  (skipping lookback plots — no data)")

    if horizon_results:
        plot_horizon_results(horizon_results, MAX_HORIZON, OUTPUT_DIR)
    else:
        print("  (skipping horizon plot — no data)")

    print(f"\nDone.  All outputs in: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
