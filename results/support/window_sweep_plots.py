"""Plotting and console summary helpers for the window sweep study."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .window_sweep_config import (
    FIXED_WINDOW,
    MAX_HORIZON,
    MODEL_COLORS,
    MODEL_NAMES,
    N_CUTOFFS,
    PROJECT_ROOT,
    STEP_MINUTES,
)


def _window_tick_labels(window_sizes: list[int]) -> list[str]:
    labels = []
    for window_size in window_sizes:
        minutes = window_size * STEP_MINUTES
        if minutes >= 60 and minutes % 60 == 0:
            labels.append(f"{window_size}\n({minutes // 60}h)")
        else:
            labels.append(f"{window_size}\n({minutes}m)")
    return labels


def _horizon_tick_labels(steps: np.ndarray) -> list[str]:
    labels = []
    for step in steps:
        minutes = int(step) * STEP_MINUTES
        if minutes >= 60 and minutes % 60 == 0:
            labels.append(f"{step}\n({minutes // 60}h)")
        else:
            labels.append(f"{step}\n({minutes}m)")
    return labels


def _save_numeric_table(df: pd.DataFrame, output_path: Path) -> None:
    """Persist one numeric companion table for a plotted figure."""
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")


def plot_lookback_results(results: dict, window_sizes: list[int], output_dir: Path) -> None:
    """Plot one-step metrics against lookback window size."""
    lookback_rows = []
    for model_name in MODEL_NAMES + ["Persistence"]:
        if model_name not in results:
            continue
        for window_size in window_sizes:
            entry = results[model_name].get(str(window_size))
            if not entry:
                continue
            lookback_rows.append({
                "model": model_name,
                "window_size_steps": window_size,
                "window_size_minutes": window_size * STEP_MINUTES,
                "MAE": entry.get("MAE"),
                "RMSE": entry.get("RMSE"),
                "MAPE": entry.get("MAPE"),
                "R2": entry.get("R2"),
            })

    metric_specs = [
        ("MAE", "MAE (kW)", False),
        ("RMSE", "RMSE (kW)", False),
        ("MAPE", "MAPE (%)", False),
        ("R2", "R2", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Model Performance vs Lookback Window Size\n"
        f"(1-step-ahead forecast, 80/20 split, {STEP_MINUTES}-min time bins)",
        fontsize=13,
        fontweight="bold",
    )
    axes_flat = axes.flatten()
    ws_arr = np.array(window_sizes, dtype=float)

    for ax_idx, (metric_key, ylabel, higher_better) in enumerate(metric_specs):
        ax = axes_flat[ax_idx]

        for model_name in MODEL_NAMES + ["Persistence"]:
            if model_name not in results:
                continue

            values = np.array([
                float(results[model_name].get(str(window_size), {}).get(metric_key, np.nan)
                      if results[model_name].get(str(window_size)) else np.nan)
                for window_size in window_sizes
            ])

            is_baseline = model_name == "Persistence"
            color = MODEL_COLORS.get(model_name, "grey")
            linewidth = 1.5 if is_baseline else 2.0
            linestyle = "--" if is_baseline else "-"
            label = "Persistence baseline" if is_baseline else model_name

            ax.plot(
                ws_arr,
                values,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                marker="o" if not is_baseline else None,
                markersize=5,
                label=label,
                alpha=0.85,
            )

            if not is_baseline:
                valid = ~np.isnan(values)
                if valid.any():
                    best_idx = np.nanargmax(values) if higher_better else np.nanargmin(values)
                    ax.scatter(
                        ws_arr[best_idx],
                        values[best_idx],
                        color=color,
                        s=100,
                        edgecolors="black",
                        linewidths=1.2,
                    )

        ax.set_title(metric_key, fontsize=11, fontweight="bold")
        ax.set_xlabel("Lookback window size")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ws_arr)
        ax.set_xticklabels(_window_tick_labels(window_sizes), fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "fig_lookback.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")
    _save_numeric_table(pd.DataFrame(lookback_rows), output_dir / "fig_lookback_data.csv")


def plot_horizon_results(results: dict, max_horizon: int, output_dir: Path) -> None:
    """Plot MAE against forecast horizon with a shaded standard deviation band."""
    steps = np.arange(1, max_horizon + 1)
    tick_mask = np.array([step in {1, 2, 3, 4, 6, 8, 12, 18, 24} for step in steps])
    tick_positions = steps[tick_mask]
    horizon_rows = []

    for model_name in MODEL_NAMES + ["Persistence"]:
        if model_name not in results:
            continue
        entry = results[model_name]
        means = np.array(entry["mean_mae_per_step"], dtype=float)
        stds = np.array(entry["std_mae_per_step"], dtype=float)
        for step_idx, (mean_value, std_value) in enumerate(zip(means, stds), start=1):
            horizon_rows.append({
                "model": model_name,
                "horizon_step": step_idx,
                "horizon_minutes": step_idx * STEP_MINUTES,
                "mean_mae": float(mean_value),
                "std_mae": float(std_value),
            })

    fig, ax = plt.subplots(figsize=(11, 6))

    for model_name in MODEL_NAMES + ["Persistence"]:
        if model_name not in results:
            continue

        entry = results[model_name]
        means = np.array(entry["mean_mae_per_step"], dtype=float)
        stds = np.array(entry["std_mae_per_step"], dtype=float)

        is_baseline = model_name == "Persistence"
        color = MODEL_COLORS.get(model_name, "grey")
        linewidth = 1.5 if is_baseline else 2.0
        linestyle = "--" if is_baseline else "-"
        label = "Persistence baseline" if is_baseline else model_name

        ax.plot(
            steps,
            means,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            marker="o" if not is_baseline else None,
            markersize=4,
            label=label,
        )
        ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.12 if is_baseline else 0.18)

    ax.set_xlabel("Forecast horizon (step ahead)", fontsize=11)
    ax.set_ylabel("MAE (kW)", fontsize=11)
    ax.set_title(
        f"Forecast Error vs Horizon\n"
        f"Lookback = {FIXED_WINDOW} steps ({FIXED_WINDOW * STEP_MINUTES} min)  |  "
        f"Averaged over {N_CUTOFFS} test cutoffs  |  Shaded band = +/-1 std",
        fontsize=11,
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(_horizon_tick_labels(tick_positions), fontsize=8)
    ax.set_xlim(0.5, max_horizon + 0.5)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=9)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    hour_steps = [step for step in steps if (step * STEP_MINUTES) % 60 == 0]
    if hour_steps:
        ax2.set_xticks(hour_steps)
        ax2.set_xticklabels([f"{step * STEP_MINUTES // 60}h" for step in hour_steps], fontsize=8)
        ax2.set_xlabel("Forecast horizon (hours ahead)", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "fig_horizon.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")
    _save_numeric_table(pd.DataFrame(horizon_rows), output_dir / "fig_horizon_data.csv")


def plot_heatmap(lookback_results: dict, window_sizes: list[int], output_dir: Path) -> None:
    """Plot a model-by-window MAE heatmap."""
    n_rows = len(MODEL_NAMES)
    n_cols = len(window_sizes)
    mat = np.full((n_rows, n_cols), np.nan)
    heatmap_rows = []

    for row_idx, model_name in enumerate(MODEL_NAMES):
        for col_idx, window_size in enumerate(window_sizes):
            entry = lookback_results.get(model_name, {}).get(str(window_size))
            if entry and entry.get("MAE") is not None:
                mat[row_idx, col_idx] = float(entry["MAE"])
                heatmap_rows.append({
                    "model": model_name,
                    "window_size_steps": window_size,
                    "window_size_minutes": window_size * STEP_MINUTES,
                    "MAE": float(entry["MAE"]),
                })

    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)
    midpoint = (vmin + vmax) / 2.0

    fig, ax = plt.subplots(figsize=(max(9, n_cols * 1.3), 3.2))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(_window_tick_labels(window_sizes), fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(MODEL_NAMES, fontsize=10)
    ax.set_xlabel("Lookback window size")
    ax.set_title("MAE (kW) - Model x Lookback Window  (green = lower error = better)", fontsize=11)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            value = mat[row_idx, col_idx]
            if not np.isnan(value):
                text_color = "white" if value > midpoint else "black"
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=9,
                        color=text_color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, label="MAE (kW)", shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = output_dir / "fig_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")
    _save_numeric_table(pd.DataFrame(heatmap_rows), output_dir / "fig_heatmap_data.csv")


def print_lookback_summary(results: dict, window_sizes: list[int]) -> None:
    """Print a compact table of one-step MAE by lookback size."""
    print("\n" + "=" * 72)
    print("  LOOKBACK SWEEP - test-set MAE (kW)")
    print("  Architecture and window size (steps) - 80/20 train/test, 1 step ahead")
    print("=" * 72)

    header = f"  {'Model':<14}" + "".join(f"  {window_size:>5}" for window_size in window_sizes)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model_name in MODEL_NAMES + ["Persistence"]:
        if model_name not in results:
            continue
        row = f"  {model_name:<14}"
        for window_size in window_sizes:
            entry = results[model_name].get(str(window_size))
            if entry and entry.get("MAE") is not None:
                row += f"  {entry['MAE']:>5.3f}"
            else:
                row += f"  {'-':>5}"
        print(row)

    print()
    for model_name in MODEL_NAMES:
        if model_name not in results:
            continue
        valid = [
            (window_size, entry["MAE"])
            for window_size in window_sizes
            for entry in [results[model_name].get(str(window_size), {}) or {}]
            if entry.get("MAE") is not None
        ]
        if valid:
            best_window, best_mae = min(valid, key=lambda item: item[1])
            print(f"  {model_name}: best window = {best_window} steps ({best_window * STEP_MINUTES} min), "
                  f"MAE = {best_mae:.3f} kW")


def print_horizon_summary(results: dict, max_horizon: int) -> None:
    """Print a compact table of mean MAE at selected forecast horizons."""
    key_steps = [step for step in [1, 2, 4, 6, 8, 12, 18, 24] if step <= max_horizon]

    print("\n" + "=" * 72)
    print("  HORIZON SWEEP - mean MAE (kW) at key forecast steps")
    print(f"  Fixed lookback = {FIXED_WINDOW} steps ({FIXED_WINDOW * STEP_MINUTES} min)"
          f"  |  averaged over {N_CUTOFFS} test cutoffs")
    print("=" * 72)

    header = f"  {'Model':<14}" + "".join(f"  t+{step:>2}({step * STEP_MINUTES}m)" for step in key_steps)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model_name in MODEL_NAMES + ["Persistence"]:
        if model_name not in results:
            continue
        means = results[model_name]["mean_mae_per_step"]
        row = f"  {model_name:<14}"
        for step in key_steps:
            idx = step - 1
            if idx < len(means):
                row += f"  {means[idx]:>10.3f}"
            else:
                row += f"  {'-':>10}"
        print(row)
    print()