"""Analysis tables and figures for the window sweep workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

from src.evaluation.trainer import evaluate_model, train_model
from src.models.architectures import build_model_kwargs, get_model

from .window_sweep_config import (
    DROPOUT,
    FIXED_WINDOW,
    HIDDEN_DIM,
    MODEL_COLORS,
    MODEL_NAMES,
    NUM_LAYERS,
    PROJECT_ROOT,
    TARGET_COL,
)
from .window_sweep_experiments import prepare_model_inputs


EXOGENOUS_OBSERVED = {"gpu_util_frac", "gpu_mem_util", "mem_util_frac", "qps"}
CALENDAR_FEATURES = {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "hour"}


def build_input_variable_table(feature_names: list[str]) -> pd.DataFrame:
    """Describe each model input for methodology traceability."""
    rows = []
    for feature_name in feature_names:
        if feature_name in EXOGENOUS_OBSERVED:
            rows.append({
                "feature": feature_name,
                "type": "Observed exogenous",
                "available_at_inference": "History only",
                "updated_during_rollout": "No",
                "rollout_rule": "Hold last observed value fixed",
            })
        elif feature_name in CALENDAR_FEATURES:
            rows.append({
                "feature": feature_name,
                "type": "Engineered calendar",
                "available_at_inference": "Deterministic future",
                "updated_during_rollout": "Yes",
                "rollout_rule": "Advance from forecast timestamp",
            })
        elif feature_name == TARGET_COL:
            rows.append({
                "feature": feature_name,
                "type": "Observed target history",
                "available_at_inference": "Observed history",
                "updated_during_rollout": "Yes",
                "rollout_rule": "Replace with predicted power",
            })
        elif feature_name.startswith(f"{TARGET_COL}_"):
            rows.append({
                "feature": feature_name,
                "type": "Engineered from target history",
                "available_at_inference": "Derived from history/predictions",
                "updated_during_rollout": "Yes",
                "rollout_rule": "Recompute from predicted power history",
            })
        else:
            rows.append({
                "feature": feature_name,
                "type": "Other engineered",
                "available_at_inference": "Depends on feature source",
                "updated_during_rollout": "Custom",
                "rollout_rule": "Review feature-specific logic",
            })

    return pd.DataFrame(rows)


def save_input_variable_table(feature_names: list[str], output_dir: Path) -> Path:
    """Persist the input-variable documentation table as CSV and Markdown."""
    table = build_input_variable_table(feature_names)
    csv_path = output_dir / "table_input_variables.csv"
    md_path = output_dir / "table_input_variables.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(_dataframe_to_markdown(table), encoding="utf-8")
    print(f"  Saved: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"  Saved: {md_path.relative_to(PROJECT_ROOT)}")
    return csv_path


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as dependency-free Markdown."""
    headers = list(df.columns)
    rows = [headers, ["---"] * len(headers)]
    for _, row in df.iterrows():
        rows.append([str(row[col]) for col in headers])
    return "\n".join("| " + " | ".join(values) + " |" for values in rows) + "\n"


def train_fresh_fixed_window_models_for_analysis(
    df: pd.DataFrame,
    device: str,
    epochs: int,
    patience: int,
) -> dict[str, dict]:
    """Train fresh fixed-window models for analysis-only artifacts."""
    data = prepare_model_inputs(df, FIXED_WINDOW)
    runs: dict[str, dict] = {}

    for model_name in MODEL_NAMES:
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
        runs[model_name] = {
            "model": model,
            "history": history,
            "metrics": metrics,
            "data": data,
        }
    return runs


def save_training_loss_curves(model_runs: dict[str, dict], output_dir: Path) -> Path:
    """Save one training-loss figure with a panel for each model."""
    fig, axes = plt.subplots(1, len(model_runs), figsize=(5 * len(model_runs), 4), squeeze=False)
    axes_flat = axes.flatten()

    for ax, model_name in zip(axes_flat, MODEL_NAMES):
        history = model_runs[model_name]["history"]
        ax.plot(history["train_loss"], label="Train", color="#1f77b4", linewidth=2)
        ax.plot(history["val_loss"], label="Validation", color="#ff7f0e", linewidth=2, linestyle="--")
        ax.set_title(model_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(fontsize=8)

    fig.suptitle("Training Loss Curves at Fixed Lookback Window", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = output_dir / "fig_training_loss_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


def save_feature_importance_plot(model_runs: dict[str, dict], output_dir: Path, repeats: int = 5) -> Path:
    """Compute permutation importance and save one figure across all models."""
    importance_df = compute_permutation_importance(model_runs, device=_infer_device(model_runs), repeats=repeats)
    csv_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path.relative_to(PROJECT_ROOT)}")

    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(6 * len(MODEL_NAMES), 6), squeeze=False)
    axes_flat = axes.flatten()
    max_importance = max(0.0, float(importance_df["importance_mae_increase"].max()))

    for ax, model_name in zip(axes_flat, MODEL_NAMES):
        model_df = importance_df[importance_df["model"] == model_name].sort_values("importance_mae_increase")
        ax.barh(
            model_df["feature"],
            model_df["importance_mae_increase"],
            color=MODEL_COLORS.get(model_name, "#4c4c4c"),
            alpha=0.85,
        )
        ax.set_title(model_name)
        ax.set_xlabel("MAE increase after permutation")
        ax.grid(True, axis="x", alpha=0.3, linestyle=":")
        ax.set_xlim(0, max_importance * 1.1 if max_importance > 0 else 1.0)

    fig.suptitle("Permutation Feature Importance at Fixed Lookback Window", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = output_dir / "fig_feature_importance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


def compute_permutation_importance(model_runs: dict[str, dict], device: str, repeats: int = 5) -> pd.DataFrame:
    """Estimate one-step permutation importance on the fixed-window test set."""
    rows = []
    for model_name, run in model_runs.items():
        data = run["data"]
        x_test, y_test_scaled = _collect_test_sequences(data)
        if len(x_test) == 0:
            continue
        baseline_mae = _evaluate_sequence_mae(
            run["model"], x_test, y_test_scaled, data["target_scaler"], device,
        )

        rng = np.random.default_rng(42)
        for feature_idx, feature_name in enumerate(data["feature_names"]):
            deltas = []
            for _ in range(repeats):
                perm = rng.permutation(len(x_test))
                x_perm = x_test.copy()
                x_perm[:, :, feature_idx] = x_test[perm, :, feature_idx]
                permuted_mae = _evaluate_sequence_mae(
                    run["model"], x_perm, y_test_scaled, data["target_scaler"], device,
                )
                deltas.append(permuted_mae - baseline_mae)

            rows.append({
                "model": model_name,
                "feature": feature_name,
                "baseline_mae": baseline_mae,
                "importance_mae_increase": float(np.mean(deltas)),
            })

    return pd.DataFrame(rows)


def _collect_test_sequences(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract all test windows and targets from the prepared dataset."""
    dataset = data["test_loader"].dataset
    x_rows = []
    y_rows = []
    for idx in range(len(dataset)):
        x_item, y_item = dataset[idx]
        x_rows.append(x_item.numpy())
        y_rows.append(float(y_item.item()))
    if not x_rows:
        return np.empty((0, 0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(x_rows).astype(np.float32), np.asarray(y_rows, dtype=np.float32)


def _evaluate_sequence_mae(model, x_sequences: np.ndarray, y_scaled: np.ndarray, target_scaler, device: str) -> float:
    """Run one-step inference on prebuilt sequences and return MAE on the original scale."""
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_sequences, dtype=torch.float32, device=device)
        preds_scaled = model(x_tensor).detach().cpu().numpy().reshape(-1, 1)
    preds = target_scaler.inverse_transform(preds_scaled).flatten()
    actuals = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    return float(mean_absolute_error(actuals, preds))


def _infer_device(model_runs: dict[str, dict]) -> str:
    """Recover the device string from a trained model bundle."""
    first_model = next(iter(model_runs.values()))["model"]
    return str(next(first_model.parameters()).device)


def generate_analysis_outputs(df: pd.DataFrame, device: str, epochs: int, patience: int, output_dir: Path) -> None:
    """Generate the extra table and figures requested for model analysis."""
    print("\n[Additional Analysis]")
    fixed_window_data = prepare_model_inputs(df, FIXED_WINDOW)
    save_input_variable_table(fixed_window_data["feature_names"], output_dir)
    print("  Training fresh fixed-window models for loss curves and feature importance")
    model_runs = train_fresh_fixed_window_models_for_analysis(df, device, epochs, patience)
    save_training_loss_curves(model_runs, output_dir)
    save_feature_importance_plot(model_runs, output_dir)