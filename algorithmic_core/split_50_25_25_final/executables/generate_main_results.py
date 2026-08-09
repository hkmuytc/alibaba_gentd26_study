#!/usr/bin/env python3
"""
Generate results for the 50/25/25, w=60, Transformer configuration:
  1. Power overlay figure (honest — no future information)
  2. Power domain metrics table (Markdown)
"""

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.pipeline import (load_raw_signals, aggregate_to_cluster, engineer_features,
                      estimate_power_kw, prepare_data)
from core.models import build_model
from core.train import (train_model, evaluate_model, get_device, compute_metrics,
                   persistence_baseline)

R = Path(__file__).resolve().parent.parent / "results"
TRAIN_R, VAL_R, WINDOW = 0.50, 0.25, 60
SEED = 0

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 14, "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize": 16, "axes.labelsize": 15, "legend.fontsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.facecolor": "white", "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc", "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    print(f"=== Generating main results (50/25/25, w={WINDOW}, Transformer, seed={SEED}) ===\n")

    data, feat, agg, n_gpus = load_and_prepare()
    print(f"  Data: {len(feat)} timesteps, {data['n_train']} train, "
          f"{data['n_val']} val, {data['n_test']} test")
    print(f"  Features: {len(data['feature_cols'])}")
    print(f"  n_gpus: {n_gpus}\n")

    model = get_transformer(data)

    # GPU util metrics
    m_gpu = evaluate_model(model, data["test_loader"], data["tgt_scaler"],
                            residual_mode=True,
                            yp_test_orig=data["yp_test_orig"],
                            y_test_orig=data["y_test_orig"])
    p_gpu = persistence_baseline(data["yp_test_orig"], data["y_test_orig"])
    print(f"\n  GPU Util — Transformer: MAE={m_gpu['MAE']:.4f}, R²={m_gpu['R2']:.4f}")
    print(f"  GPU Util — Persistence: MAE={p_gpu['MAE']:.4f}, R²={p_gpu['R2']:.4f}")
    print(f"  Δ MAE: {(p_gpu['MAE']-m_gpu['MAE'])/p_gpu['MAE']*100:+.1f}%, "
          f"Δ R²: {m_gpu['R2']-p_gpu['R2']:+.4f}\n")

    m_power, p_power = fig_power_overlay(data, feat, n_gpus, model)

    # Ablation: full transformer vs history-only vs persistence
    history_power = run_ablation(data, feat, n_gpus, m_power, p_power)
    write_power_table(m_power, p_power, history_power)

    print("\nAll results generated.")


def _eval_ablation_model(model_name, data, feat, n_gpus):
    """Load an ablation model from cache, evaluate, return power-domain predictions and metrics."""
    from core.model_cache import ModelCache
    cache = ModelCache(Path(__file__).resolve().parent.parent / "models" / "onestep")
    cache_name = f"{model_name}_w{WINDOW}_{int(TRAIN_R*100)}-{int(VAL_R*100)}-{int((1-TRAIN_R-VAL_R)*100)}"

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = build_model(model_name, len(data["feature_cols"]), hidden_dim=32)
    cached = cache.load(cache_name, model)

    if cached is not None:
        model, _, _ = cached
    else:
        model, _ = train_model(model, data["train_loader"], data["val_loader"],
                                epochs=200, patience=20, lr=1e-3, weight_decay=1e-4)

    device = get_device()
    model = model.to(device).eval()
    preds_list = []
    with torch.no_grad():
        for Xb, _ in data["test_loader"]:
            preds_list.append(model(Xb.to(device)).cpu().numpy())
    preds_s = np.concatenate(preds_list)

    pred_d = data["tgt_scaler"].inverse_transform(preds_s.reshape(-1, 1)).flatten()
    yp = data["yp_test_orig"]
    pred_gpu = yp + pred_d

    n = len(feat)
    val_end = int(n * (TRAIN_R + VAL_R))
    test_start = max(val_end, WINDOW)
    n_test = len(yp)
    target_indices = list(range(test_start, test_start + n_test))
    prev_idx = [i - 1 for i in target_indices]
    apr_pred = feat["active_pod_ratio"].values[prev_idx]     # last known for prediction
    apr_true = feat["active_pod_ratio"].values[target_indices]  # actual for ground truth

    pred_kw = estimate_power_kw(pred_gpu, apr_pred, n_gpus)

    tgts_s = np.concatenate([yb.detach().numpy() for _, yb in data["test_loader"]])
    tgt_d = data["tgt_scaler"].inverse_transform(tgts_s.reshape(-1, 1)).flatten()
    true_kw = estimate_power_kw(yp + tgt_d, apr_true, n_gpus)

    metrics = compute_metrics(true_kw, pred_kw)
    return pred_kw, true_kw, metrics


def run_ablation(data, feat, n_gpus, m_power, p_power):
    """Ablation: full Transformer vs GPU-only Transformer vs persistence."""
    print(f"\n  Running ablation study...")

    # Full model predictions
    device = get_device()
    full_model = get_transformer(data).to(device).eval()
    full_preds_list = []
    with torch.no_grad():
        for Xb, _ in data["test_loader"]:
            full_preds_list.append(full_model(Xb.to(device)).cpu().numpy())
    full_preds_s = np.concatenate(full_preds_list)
    full_pred_d = data["tgt_scaler"].inverse_transform(full_preds_s.reshape(-1, 1)).flatten()
    yp = data["yp_test_orig"]
    n = len(feat)
    val_end = int(n * (TRAIN_R + VAL_R))
    test_start = max(val_end, WINDOW)
    n_test = len(yp)
    target_indices = list(range(test_start, test_start + n_test))
    prev_idx = [i - 1 for i in target_indices]
    apr_pred = feat["active_pod_ratio"].values[prev_idx]
    pred_kw_full = estimate_power_kw(yp + full_pred_d, apr_pred, n_gpus)
    persist_kw = estimate_power_kw(yp, apr_pred, n_gpus)

    # History-only model (11 features: gpu_util + derived statistics, no exogenous telemetry)
    pred_kw_gpu, true_kw, gpu_power = _eval_ablation_model("transformer_hist", data, feat, n_gpus)
    print(f"  History-only — Power MAE={gpu_power['MAE']:.4f} kW, R²={gpu_power['R2']:.4f}")

    # --- Ablation figure: time-series overlay ---
    import pandas as pd
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    t = np.arange(n_test)

    ax1.plot(t, true_kw, "#333", lw=1.8, label="Actual power", zorder=4)
    ax1.plot(t, pred_kw_full, "#E65100", lw=1.3,
             label=f"Full (27 features, MAE={m_power['MAE']:.4f})", alpha=0.85, zorder=3)
    ax1.plot(t, pred_kw_gpu, "#1565C0", lw=1.3,
             label=f"History only (10 features, MAE={gpu_power['MAE']:.4f})", alpha=0.85, zorder=2)
    ax1.plot(t, persist_kw, "#888", lw=1.0, ls="--",
             label=f"Persistence (MAE={p_power['MAE']:.4f})", alpha=0.5, zorder=1)
    ax1.set_ylabel("Estimated Power (kW)")
    ax1.set_title(
        f"Ablation: Full vs History-Only Transformer (50/25/25, w={WINDOW})")
    ax1.legend(loc="upper right", framealpha=0.9)

    rw = 10
    err_full = pd.Series(np.abs(pred_kw_full - true_kw)).rolling(rw, min_periods=1).mean()
    err_gpu = pd.Series(np.abs(pred_kw_gpu - true_kw)).rolling(rw, min_periods=1).mean()
    err_persist = pd.Series(np.abs(persist_kw - true_kw)).rolling(rw, min_periods=1).mean()

    ax2.plot(t, err_full, "#E65100", lw=1.5, label=f"Full ({rw}-min MAE)")
    ax2.plot(t, err_gpu, "#1565C0", lw=1.3, label=f"History only ({rw}-min MAE)")
    ax2.plot(t, err_persist, "#888", lw=1.2, ls="--", label=f"Persistence ({rw}-min MAE)")
    ax2.set_ylabel("Rolling MAE (kW)")
    ax2.set_xlabel("Time (minutes into test set)")
    ax2.set_title(f"Rolling Error ({rw}-min Window)")
    ax2.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(R / "fig_onestep_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_onestep_ablation.png")
    return gpu_power


def load_and_prepare():
    gpu, gmem, qps = load_raw_signals()
    agg = aggregate_to_cluster(gpu, gmem, qps)
    feat = engineer_features(agg)
    train_idx = feat.index[:int(len(feat) * TRAIN_R)]
    n_gpus = int(agg.loc[train_idx, "gpu_n_pods"].median())
    feat["power_kw"] = estimate_power_kw(
        feat["gpu_util"].values, feat["active_pod_ratio"].values, n_gpus)
    model_df = feat.drop(columns=["power_kw"])
    data = prepare_data(model_df, target_col="gpu_util", window_size=WINDOW,
                        train_ratio=TRAIN_R, val_ratio=VAL_R,
                        batch_size=32, residual_mode=True)
    return data, feat, agg, n_gpus


def get_transformer(data):
    """Load Transformer from cache (trained by split_study.py), or train if not cached."""
    from core.model_cache import ModelCache
    cache = ModelCache(Path(__file__).resolve().parent.parent / "models" / "onestep")
    cache_name = f"transformer_w{WINDOW}_{int(TRAIN_R*100)}-{int(VAL_R*100)}-{int((1-TRAIN_R-VAL_R)*100)}"

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_model("transformer", len(data["feature_cols"]), hidden_dim=32)
    cached = cache.load(cache_name, model)

    if cached is not None:
        model, _, metrics = cached
        cached_mae = metrics.get("MAE")
        if cached_mae is None:
            print("  Transformer loaded from cache")
        else:
            print(f"  Transformer loaded from cache (MAE={cached_mae:.4f} kW)")
    else:
        model, hist = train_model(model, data["train_loader"], data["val_loader"],
                                   epochs=200, patience=20, lr=1e-3, weight_decay=1e-4)
        print(f"  Transformer trained: {hist['epochs_trained']} epochs (seed={SEED})")
        print(f"  (Run split_study.py first to populate the model cache)")
    return model


# =====================================================================
# 1. POWER OVERLAY FIGURE
# =====================================================================

def fig_power_overlay(data, feat, n_gpus, model):
    print("[1/2] Power overlay figure...")
    device = get_device()
    model = model.to(device).eval()

    preds_s, tgts_s = [], []
    with torch.no_grad():
        for X_b, y_b in data["test_loader"]:
            preds_s.append(model(X_b.to(device)).cpu().numpy())
            tgts_s.append(y_b.numpy())
    preds_s = np.concatenate(preds_s)
    tgts_s = np.concatenate(tgts_s)

    pred_d = data["tgt_scaler"].inverse_transform(preds_s.reshape(-1, 1)).flatten()
    tgt_d = data["tgt_scaler"].inverse_transform(tgts_s.reshape(-1, 1)).flatten()
    yp = data["yp_test_orig"]

    pred_gpu = yp + pred_d
    true_gpu = yp + tgt_d
    persist_gpu = yp

    n = len(feat)
    val_end = int(n * (TRAIN_R + VAL_R))
    test_start = max(val_end, WINDOW)
    n_test = len(pred_gpu)
    target_indices = list(range(test_start, test_start + n_test))
    prev_idx = [i - 1 for i in target_indices]

    # Active pod ratio: last known for prediction, actual for ground truth
    apr = feat["active_pod_ratio"].values
    apr_pred = apr[prev_idx]
    apr_true = apr[target_indices]

    pred_kw = estimate_power_kw(pred_gpu, apr_pred, n_gpus)
    true_kw = estimate_power_kw(true_gpu, apr_true, n_gpus)
    persist_kw = estimate_power_kw(persist_gpu, apr_pred, n_gpus)

    # Metrics
    m_model = compute_metrics(true_kw, pred_kw)
    m_persist = compute_metrics(true_kw, persist_kw)

    print(f"  Power — Transformer: MAE={m_model['MAE']:.4f} kW, RMSE={m_model['RMSE']:.4f} kW, "
          f"R²={m_model['R2']:.4f}, MAPE={m_model['MAPE']:.2f}%")
    print(f"  Power — Persistence: MAE={m_persist['MAE']:.4f} kW, RMSE={m_persist['RMSE']:.4f} kW, "
          f"R²={m_persist['R2']:.4f}, MAPE={m_persist['MAPE']:.2f}%")
    dm = (m_persist["MAE"] - m_model["MAE"]) / m_persist["MAE"] * 100
    dr = m_model["R2"] - m_persist["R2"]
    print(f"  Δ: MAE {dm:+.1f}%, R² {dr:+.4f}")

    # Plot
    t = np.arange(n_test)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    ax1.plot(t, true_kw, "#333", lw=1.8, label="Actual", zorder=3)
    ax1.plot(t, pred_kw, "#E65100", lw=1.3, label="Transformer", alpha=0.85, zorder=2)
    ax1.plot(t, persist_kw, "#888", lw=1.1, label="Persistence", alpha=0.6, ls="--", zorder=1)
    ax1.fill_between(t, true_kw, pred_kw, alpha=0.08, color="#E65100")
    ax1.set_ylabel("Estimated Power (kW)")
    ax1.set_title(
        f"One-Step Power Forecast (50/25/25, w={WINDOW})"
    )
    ax1.legend(loc="upper right", framealpha=0.9)

    rw = 10
    em = pd.Series(np.abs(pred_kw - true_kw)).rolling(rw, min_periods=1).mean()
    ep = pd.Series(np.abs(persist_kw - true_kw)).rolling(rw, min_periods=1).mean()
    ax2.fill_between(t, em, alpha=0.25, color="#E65100")
    ax2.plot(t, em, "#E65100", lw=1.2, label=f"Transformer ({rw}-min MAE)")
    ax2.fill_between(t, ep, alpha=0.15, color="#888")
    ax2.plot(t, ep, "#888", lw=1.2, label=f"Persistence ({rw}-min MAE)")
    ax2.set_ylabel("Rolling MAE (kW)")
    ax2.set_xlabel("Time (minutes into test set)")
    ax2.set_title(f"Rolling Error ({rw}-min Window)")
    ax2.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(R / "fig_onestep_power_overlay.png", bbox_inches="tight")
    plt.close(fig)
    print("  → fig_onestep_power_overlay.png")

    return m_model, m_persist


# =====================================================================
# 2. POWER DOMAIN TABLE
# =====================================================================

def write_power_table(m_model, m_persist, history_power):
    print("[2/2] Power domain and ablation metrics table...")

    def mae_improvement(metrics):
        return (m_persist["MAE"] - metrics["MAE"]) / m_persist["MAE"] * 100

    def rmse_improvement(metrics):
        return (m_persist["RMSE"] - metrics["RMSE"]) / m_persist["RMSE"] * 100

    def mape_improvement(metrics):
        return (m_persist["MAPE"] - metrics["MAPE"]) / m_persist["MAPE"] * 100

    def r2_delta(metrics):
        return metrics["R2"] - m_persist["R2"]

    lines = [
        "# Power Domain and Ablation Results (50/25/25, w=60, Transformer)\n",
        "Power estimated via Fan et al. (2007): P = n_gpus × (50 + 250 × gpu_util) / 1000",
        "The idle term is charged to every powered GPU; active-pod ratio is not used in the power conversion.\n",
        f"**Configuration**: 50/25/25 split, w={WINDOW}, seed={SEED}",
        "**Full model**: 27 features (GPU utilization and derived statistics + exogenous workload telemetry)",
        "**History-only model**: 10 features (GPU utilization, lags, rolling means/stds, rate of change, fractional differencing — no QPS, pod ratios, memory, or time-of-day)",
        "**Persistence**: predict current value for all future steps\n",
        "| Method | Feature Set | Features | MAE (kW) | RMSE (kW) | R² | MAPE (%) | vs Persistence MAE | vs Persistence R² |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| **Transformer (full)** | GPU history + workload telemetry | 27 | **{m_model['MAE']:.4f}** | "
        f"**{m_model['RMSE']:.4f}** | **{m_model['R2']:.4f}** | **{m_model['MAPE']:.2f}** | "
        f"**{mae_improvement(m_model):+.1f}%** | **{r2_delta(m_model):+.4f}** |",
        f"| Transformer (history only) | GPU history only | 10 | {history_power['MAE']:.4f} | "
        f"{history_power['RMSE']:.4f} | {history_power['R2']:.4f} | {history_power['MAPE']:.2f} | "
        f"{mae_improvement(history_power):+.1f}% | {r2_delta(history_power):+.4f} |",
        f"| Persistence | Repeat last value | — | {m_persist['MAE']:.4f} | {m_persist['RMSE']:.4f} | "
        f"{m_persist['R2']:.4f} | {m_persist['MAPE']:.2f} | — | — |",
        "",
        "## Relative Improvements over Persistence",
        "",
        "| Method | MAE | RMSE | R² | MAPE |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Transformer (full) | {mae_improvement(m_model):+.1f}% | {rmse_improvement(m_model):+.1f}% | "
        f"{r2_delta(m_model):+.4f} | {mape_improvement(m_model):+.1f}% |",
        f"| Transformer (history only) | {mae_improvement(history_power):+.1f}% | {rmse_improvement(history_power):+.1f}% | "
        f"{r2_delta(history_power):+.4f} | {mape_improvement(history_power):+.1f}% |",
        "",
    ]
    (R / "power_table.md").write_text("\n".join(lines))
    print("  → power_table.md")
    print()
    print("  " + "\n  ".join(lines))


if __name__ == "__main__":
    main()
