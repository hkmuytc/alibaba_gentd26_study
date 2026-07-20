#!/usr/bin/env python3
"""
Split strategy study: compare 50/25/25, 60/20/20, 70/15/15
across all models and window sizes. All metrics reported in POWER DOMAIN (kW).
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.pipeline import (load_raw_signals, aggregate_to_cluster, engineer_features,
                      estimate_power_kw, prepare_data)
from core.models import build_model
from core.train import train_model, get_device, compute_metrics

R = Path(__file__).resolve().parent.parent / "results"
SEED = 0

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 10, "font.family": "serif",
    "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "#fafafa",
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

SPLITS = [(0.50, 0.25), (0.60, 0.20), (0.70, 0.15)]
SPLIT_LABELS = ["50/25/25", "60/20/20", "70/15/15"]
WINDOWS = [15, 30, 60]
MODELS = ["lstm", "gru", "transformer", "cnn_lstm"]
MODEL_LABELS = {"lstm": "LSTM", "gru": "GRU", "transformer": "Transformer", "cnn_lstm": "CNN-LSTM"}
MODEL_COLORS = {"lstm": "#1565C0", "gru": "#2E7D32", "transformer": "#E65100", "cnn_lstm": "#6A1B9A"}

def main():
    print(f"=== SPLIT STRATEGY STUDY — POWER DOMAIN (seed={SEED}) ===\n")
    results = run_all()

    print("\n[Figures]")
    fig_mae_by_split(results);          print("  fig_split_study_mae.png")
    fig_r2_by_split(results);           print("  fig_split_study_r2.png")

    print("\n[Tables]")
    generate_tables(results);            print("  split_study_tables.md")

    print("\n Done.")

def evaluate_power(model, data, feat_df, n_gpus, window_size, train_ratio, val_ratio):
    """Evaluate model in the power domain (kW)."""
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

    # Compute test target indices matching prepare_data exactly
    n_feat = len(feat_df)
    val_end = int(n_feat * (train_ratio + val_ratio))
    # Test targets are at indices [max(val_end, window_size), n_feat)
    test_start = max(val_end, window_size)
    n_test = data["n_test"]
    target_indices = list(range(test_start, test_start + n_test))
    prev_indices = [i - 1 for i in target_indices]
    apr_pred = feat_df["active_pod_ratio"].values[prev_indices]
    apr_true = feat_df["active_pod_ratio"].values[target_indices]

    pred_kw = estimate_power_kw(pred_gpu, apr_pred, n_gpus)
    true_kw = estimate_power_kw(true_gpu, apr_true, n_gpus)
    persist_kw = estimate_power_kw(persist_gpu, apr_pred, n_gpus)

    m_model = compute_metrics(true_kw, pred_kw)
    m_persist = compute_metrics(true_kw, persist_kw)
    return m_model, m_persist


def run_all():
    """Run all split x model x window combinations. Metrics in power domain (kW).
    Uses model cache to avoid retraining when possible."""
    from core.model_cache import ModelCache
    cache = ModelCache(Path(__file__).resolve().parent.parent / "models" / "onestep")

    gpu, gmem, qps = load_raw_signals()
    agg = aggregate_to_cluster(gpu, gmem, qps)
    feat = engineer_features(agg)
    feat["power_kw"] = estimate_power_kw(feat["gpu_util"].values, feat["active_pod_ratio"].values,
                                          int(agg["gpu_n_pods"].median()))  # for feature df only
    model_df = feat.drop(columns=["power_kw"])

    results = {}
    n_loaded = 0
    n_trained = 0

    for train_r, val_r in SPLITS:
        label = f"{int(train_r*100)}/{int(val_r*100)}/{int((1-train_r-val_r)*100)}"
        cache_label = f"{int(train_r*100)}-{int(val_r*100)}-{int((1-train_r-val_r)*100)}"
        n_gpus = int(agg.iloc[:int(len(agg) * train_r)]["gpu_n_pods"].median())
        results[label] = {}

        for window in WINDOWS:
            data = prepare_data(model_df, target_col="gpu_util", window_size=window,
                                train_ratio=train_r, val_ratio=val_r,
                                batch_size=32, residual_mode=True)

            results[label][window] = {
                "n_train": data["n_train"],
                "n_val": data["n_val"],
                "n_test": data["n_test"],
                "models": {},
            }

            for name in MODELS:
                cache_name = f"{name}_w{window}_{cache_label}"

                # Try loading from cache
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                model = build_model(name, len(data["feature_cols"]), hidden_dim=32)
                cached = cache.load(cache_name, model)

                if cached is not None:
                    model, _, cached_metrics = cached
                    m_model = cached_metrics
                    n_loaded += 1
                    src = "cached"
                else:
                    model, hist = train_model(model, data["train_loader"], data["val_loader"],
                                               epochs=200, patience=20, lr=1e-3, weight_decay=1e-4)
                    m_model, _ = evaluate_power(model, data, feat, n_gpus, window, train_r, val_r)
                    m_model["epochs"] = hist["epochs_trained"]

                    scalers = {"feat_scaler": data["feat_scaler"], "tgt_scaler": data["tgt_scaler"]}
                    cache.save(cache_name, model, scalers, m_model)
                    n_trained += 1
                    src = "trained"

                m_persist = evaluate_power(model, data, feat, n_gpus, window, train_r, val_r)[1]

                results[label][window]["persistence"] = m_persist
                results[label][window]["models"][name] = m_model

                beat = "✓" if m_model["MAE"] < m_persist["MAE"] else " "
                print(f"  {label} w={window:>2} {name:>12} [{src:>6}]: "
                      f"MAE={m_model['MAE']:.4f} kW  R2={m_model['R2']:.4f}  "
                      f"| Persist MAE={m_persist['MAE']:.4f} kW  R2={m_persist['R2']:.4f}  {beat}")

    print(f"\n  Cache: {n_loaded} loaded, {n_trained} trained ({n_loaded + n_trained} total)")
    return results


# =====================================================================
# FIGURES
# =====================================================================

def fig_mae_by_split(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for si, split_label in enumerate(SPLIT_LABELS):
        ax = axes[si]
        x = np.arange(len(WINDOWS))
        width = 0.15
        methods = ["persistence"] + MODELS
        labels = ["Persistence"] + [MODEL_LABELS[m] for m in MODELS]
        colors = ["#888888"] + [MODEL_COLORS[m] for m in MODELS]

        for mi, (method, label, color) in enumerate(zip(methods, labels, colors)):
            vals = []
            for w in WINDOWS:
                if method == "persistence":
                    vals.append(results[split_label][w]["persistence"]["MAE"])
                else:
                    vals.append(results[split_label][w]["models"][method]["MAE"])
            ax.bar(x + mi * width, vals, width, label=label, color=color,
                   alpha=0.85, edgecolor="white", linewidth=0.5)

            for j, v in enumerate(vals):
                all_v = [results[split_label][WINDOWS[j]]["persistence"]["MAE"]] + \
                        [results[split_label][WINDOWS[j]]["models"][m]["MAE"] for m in MODELS]
                if v == min(all_v):
                    ax.plot(x[j] + mi * width, v + 0.0005, "*", color=color,
                            markersize=10, markeredgewidth=0.5, markeredgecolor="black")

        ax.set_title(f"Split: {split_label}", fontsize=12)
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels([f"w={w}" for w in WINDOWS])
        ax.set_ylim(bottom=0)
        if si == 0:
            ax.set_ylabel("MAE (kW)")
            ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)

    fig.suptitle("One-Step Power Prediction MAE (kW) by Model, Window, and Split", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(R / "fig_onestep_split_mae.png", bbox_inches="tight")
    plt.close(fig)


def fig_r2_by_split(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for si, split_label in enumerate(SPLIT_LABELS):
        ax = axes[si]
        x = np.arange(len(WINDOWS))
        width = 0.15
        methods = ["persistence"] + MODELS
        labels = ["Persistence"] + [MODEL_LABELS[m] for m in MODELS]
        colors = ["#888888"] + [MODEL_COLORS[m] for m in MODELS]

        for mi, (method, label, color) in enumerate(zip(methods, labels, colors)):
            vals = []
            for w in WINDOWS:
                if method == "persistence":
                    vals.append(results[split_label][w]["persistence"]["R2"])
                else:
                    vals.append(results[split_label][w]["models"][method]["R2"])
            ax.bar(x + mi * width, vals, width, label=label, color=color,
                   alpha=0.85, edgecolor="white", linewidth=0.5)

            for j, v in enumerate(vals):
                all_v = [results[split_label][WINDOWS[j]]["persistence"]["R2"]] + \
                        [results[split_label][WINDOWS[j]]["models"][m]["R2"] for m in MODELS]
                if v == max(all_v):
                    ax.plot(x[j] + mi * width, v + 0.005, "*", color=color,
                            markersize=10, markeredgewidth=0.5, markeredgecolor="black")

        ax.set_title(f"Split: {split_label}", fontsize=12)
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels([f"w={w}" for w in WINDOWS])
        # Dynamic y-axis: find min R² across all methods and windows for this split
        all_r2 = []
        for w in WINDOWS:
            all_r2.append(results[split_label][w]["persistence"]["R2"])
            for m in MODELS:
                all_r2.append(results[split_label][w]["models"][m]["R2"])
        y_min = min(0, min(all_r2) - 0.05)
        ax.set_ylim(y_min, 1.0)
        if si == 0:
            ax.set_ylabel("$R^2$ (Power, kW)")
            ax.legend(loc="lower right", fontsize=7, ncol=2, framealpha=0.9)

    fig.suptitle("One-Step Power Prediction $R^2$ by Model, Window, and Split", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(R / "fig_onestep_split_r2.png", bbox_inches="tight")
    plt.close(fig)




# =====================================================================
# TABLES
# =====================================================================

def generate_tables(results):
    lines = []
    lines.append("## Split Strategy Comparison — Power Domain (kW)\n")
    lines.append("All metrics computed in the power domain via Fan et al. (2007) linear model.\n")

    for sl in SPLIT_LABELS:
        n_tr = results[sl][WINDOWS[0]]["n_train"]
        n_va = results[sl][WINDOWS[0]]["n_val"]
        n_te = results[sl][WINDOWS[0]]["n_test"]
        lines.append(f"### Split {sl} (train={n_tr}, val={n_va}, test={n_te})\n")

        for metric, unit in [("MAE", " (kW)"), ("R2", "")]:
            mkey = "R2" if metric == "R2" else metric
            lines.append(f"#### {metric}{unit}\n")
            lines.append("| Method | w=15 | w=30 | w=60 |")
            lines.append("| --- | ---: | ---: | ---: |")

            for method_label, method_key in [("Persistence", "persistence")] + \
                                             [(MODEL_LABELS[m], m) for m in MODELS]:
                cells = []
                for w in WINDOWS:
                    if method_key == "persistence":
                        v = results[sl][w]["persistence"][mkey]
                    else:
                        v = results[sl][w]["models"][method_key][mkey]
                    all_v = [results[sl][w]["persistence"][mkey]] + \
                            [results[sl][w]["models"][m][mkey] for m in MODELS]
                    is_best = (v == min(all_v)) if mkey == "MAE" else (v == max(all_v))
                    fmt = ".4f" if mkey == "MAE" else ".3f"
                    s = f"**{v:{fmt}}**" if is_best else f"{v:{fmt}}"
                    cells.append(s)
                lines.append(f"| {method_label} | " + " | ".join(cells) + " |")

            deltas = []
            for w in WINDOWS:
                p = results[sl][w]["persistence"][mkey]
                best_vals = [results[sl][w]["models"][m][mkey] for m in MODELS]
                best = min(best_vals) if mkey == "MAE" else max(best_vals)
                if mkey == "MAE":
                    d = (p - best) / p * 100
                    deltas.append(f"{'+' if d > 0 else ''}{d:.1f}%")
                else:
                    d = best - p
                    deltas.append(f"{'+' if d > 0 else ''}{d:.3f}")
            lines.append(f"| Δ vs Persist. | " + " | ".join(deltas) + " |")
            lines.append("")

    # Best configuration summary
    lines.append("---\n")
    lines.append("## Best Configuration Summary (Power Domain)\n")
    lines.append("| Split | Window | Best Model | MAE (kW) | Persist MAE (kW) | Δ MAE | R² | Persist R² | Δ R² |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for sl in SPLIT_LABELS:
        best_mae = float("inf")
        best_info = None
        for w in WINDOWS:
            for m in MODELS:
                v = results[sl][w]["models"][m]["MAE"]
                if v < best_mae:
                    best_mae = v
                    p = results[sl][w]["persistence"]
                    best_info = (sl, w, MODEL_LABELS[m], v, p["MAE"],
                                 results[sl][w]["models"][m]["R2"], p["R2"])
        sl_, w_, m_, mae, pmae, r2, pr2 = best_info
        dm = (pmae - mae) / pmae * 100
        dr = r2 - pr2
        lines.append(f"| {sl_} | {w_} | {m_} | {mae:.4f} | {pmae:.4f} | "
                     f"{'+' if dm > 0 else ''}{dm:.1f}% | {r2:.3f} | {pr2:.3f} | "
                     f"{'+' if dr > 0 else ''}{dr:.3f} |")
    lines.append("")

    (R / "split_study_tables.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()