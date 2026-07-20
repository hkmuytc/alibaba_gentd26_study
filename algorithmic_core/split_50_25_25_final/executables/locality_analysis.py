#!/usr/bin/env python3
"""
Locality analysis for multi-step averaged power forecasting.

Answers: "Under what temporal regimes does the model's power prediction
improvement over persistence emerge or break down?"

Focus: h=12 (representative medium-range horizon), power domain (kW).

Figures:
  1. Power error growth across horizons (overview)
  2. Per-sample power advantage at h=12 (when does the model win?)
  3. Power error by data regime at h=12 (quantified breakdown)
  4. Best/worst prediction cases at h=12 (qualitative examples)
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.pipeline import (load_raw_signals, aggregate_to_cluster, engineer_features,
                      estimate_power_kw)
from core.models import build_model
from core.train import train_model, get_device

R = Path(__file__).resolve().parent.parent / "results"
SEED = 0
WINDOW = 60
TRAIN_R, VAL_R = 0.50, 0.25
HORIZONS = [1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 24]
H_FOCUS = 12

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 10, "font.family": "serif",
    "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc", "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

def main():
    print(f"=== LOCALITY ANALYSIS — POWER DOMAIN (50/25/25, w={WINDOW}) ===\n")
    print(f"Focus horizon: h={H_FOCUS}")
    print("Training models for each horizon...")

    (results, feat, y_all, gpu_roc, gpu_vol,
     power_kw, power_roc, power_vol, n_gpus) = run_all()

    print("\nGenerating figures...")
    fig_power_error_growth(results, feat, n_gpus)
    fig_per_sample_advantage(results, power_kw)
    fig_best_worst(results, y_all, gpu_roc, gpu_vol, power_kw, n_gpus)

    # Print summary statistics for discussion
    res = results[H_FOCUS]
    idx = res["test_idx"]
    pred_idx = np.array([i - 1 for i in idx])
    advantage = res["pwr_advantage"]

    print(f"\n{'='*60}")
    print(f"SUMMARY: h={H_FOCUS} Power Domain Locality")
    print(f"{'='*60}")
    print(f"  Overall: Model wins {np.sum(advantage > 0)}/{len(advantage)} "
          f"({np.mean(advantage > 0)*100:.0f}%), "
          f"mean advantage = {np.mean(advantage):+.4f} kW")

    regimes = [
        ("Calm (σ < 0.01)", gpu_vol[pred_idx] < 0.01),
        ("Moderate vol (0.01–0.03)", (gpu_vol[pred_idx] >= 0.01) & (gpu_vol[pred_idx] < 0.03)),
        ("Volatile (σ ≥ 0.03)", gpu_vol[pred_idx] >= 0.03),
        ("Stable trend (|Δ₅| < 0.01)", np.abs(y_all[pred_idx] - y_all[np.maximum(pred_idx - 5, 0)]) < 0.01),
        ("Rising trend (Δ₅ > 0.01)", (y_all[pred_idx] - y_all[np.maximum(pred_idx - 5, 0)]) > 0.01),
        ("Falling trend (Δ₅ < −0.01)", (y_all[pred_idx] - y_all[np.maximum(pred_idx - 5, 0)]) < -0.01),
    ]

    for name, mask in regimes:
        if mask.sum() > 0:
            wr = np.mean(advantage[mask] > 0) * 100
            ma = np.mean(advantage[mask])
            print(f"  {name:<35} n={mask.sum():>3}  Win rate={wr:>5.1f}%  "
                  f"Mean advantage={ma:+.4f} kW")

    # Write locality table
    _write_locality_table(results, y_all, gpu_vol, regimes)

    print(f"\n✓ All locality figures and tables generated.")

# =====================================================================
# Data preparation and training
# =====================================================================

def forward_mean(y, h):
    n = len(y)
    result = np.full(n, np.nan)
    for t in range(n - h):
        result[t] = np.mean(y[t + 1: t + h + 1])
    return result


def rolling_std(arr, window):
    result = np.full(len(arr), np.nan)
    for i in range(window, len(arr)):
        result[i] = np.std(arr[i - window:i], ddof=1)
    return result


def train_for_horizon(feat_df, feature_cols, horizon):
    """Train model and return per-sample test results in original scale."""
    y_all = feat_df["gpu_util"].values.astype(np.float32)
    X_all = feat_df[feature_cols].values
    y_fwd = forward_mean(y_all, horizon)
    valid = ~np.isnan(y_fwd) & ~np.any(np.isnan(X_all), axis=1)
    valid_indices = np.where(valid)[0]
    valid_indices = valid_indices[valid_indices >= WINDOW]

    n_valid = len(valid_indices)
    n_train = int(n_valid * TRAIN_R)
    n_val = int(n_valid * VAL_R)
    train_idx = valid_indices[:n_train]
    val_idx = valid_indices[n_train:n_train + n_val]
    test_idx = valid_indices[n_train + n_val:]

    def make_set(indices):
        Xs = np.array([X_all[i - WINDOW:i] for i in indices], dtype=np.float32)
        return Xs, y_fwd[indices - 1].astype(np.float32), y_all[indices - 1].astype(np.float32)

    X_tr, yt_tr, yc_tr = make_set(train_idx)
    X_va, yt_va, yc_va = make_set(val_idx)
    X_te, yt_te, yc_te = make_set(test_idx)

    fsc = StandardScaler()
    fsc.fit(X_tr.reshape(-1, X_tr.shape[2]))
    X_tr_s = fsc.transform(X_tr.reshape(-1, X_tr.shape[2])).reshape(X_tr.shape)
    X_va_s = fsc.transform(X_va.reshape(-1, X_va.shape[2])).reshape(X_va.shape)
    X_te_s = fsc.transform(X_te.reshape(-1, X_te.shape[2])).reshape(X_te.shape)

    tsc = StandardScaler()
    tsc.fit((yt_tr - yc_tr).reshape(-1, 1))

    def to_scaled(yt, yc):
        return tsc.transform((yt - yc).reshape(-1, 1)).flatten()

    class TSD(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]

    train_loader = DataLoader(TSD(X_tr_s, to_scaled(yt_tr, yc_tr)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TSD(X_va_s, to_scaled(yt_va, yc_va)), batch_size=32, shuffle=False)
    test_loader = DataLoader(TSD(X_te_s, to_scaled(yt_te, yc_te)), batch_size=32, shuffle=False)

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = build_model("transformer", len(feature_cols), hidden_dim=32)
    model, _ = train_model(model, train_loader, val_loader,
                            epochs=200, patience=20, lr=1e-3, weight_decay=1e-4)

    device = get_device()
    model = model.to(device).eval()
    preds_s, tgts_s = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds_s.append(model(Xb.to(device)).cpu().numpy())
            tgts_s.append(yb.numpy())
    preds_s = np.concatenate(preds_s)
    tgts_s = np.concatenate(tgts_s)

    pred_deltas = tsc.inverse_transform(preds_s.reshape(-1, 1)).flatten()
    pred_avg = yc_te + pred_deltas
    true_avg = yt_te

    return {
        "test_idx": test_idx,
        "pred_avg": pred_avg,
        "true_avg": true_avg,
        "y_current": yc_te,
        "model_gpu_err": np.abs(pred_avg - true_avg),
        "persist_gpu_err": np.abs(yc_te - true_avg),
    }


def add_power_errors(res, feat_df, n_gpus, horizon):
    """Add power-domain errors to results dict."""
    apr = feat_df["active_pod_ratio"].values
    idx = res["test_idx"]
    prev_idx = [i - 1 for i in idx]
    apr_pred = apr[prev_idx]

    # Ground truth: average of instantaneous power over the horizon
    # This correctly accounts for E[gpu*apr] ≠ E[gpu]*E[apr] when correlated
    all_true_power = estimate_power_kw(
        feat_df["gpu_util"].values, feat_df["active_pod_ratio"].values, n_gpus)
    true_kw = np.array([np.mean(all_true_power[i:i + horizon]) for i in idx])

    pred_kw = estimate_power_kw(res["pred_avg"], apr_pred, n_gpus)
    persist_kw = estimate_power_kw(res["y_current"], apr_pred, n_gpus)

    res["apr_pred"] = apr_pred
    res["model_pwr_err"] = np.abs(pred_kw - true_kw)
    res["persist_pwr_err"] = np.abs(persist_kw - true_kw)
    res["pwr_advantage"] = res["persist_pwr_err"] - res["model_pwr_err"]
    return res


def run_all():
    gpu, gmem, qps = load_raw_signals()
    agg = aggregate_to_cluster(gpu, gmem, qps, bin_sec=60)
    feat = engineer_features(agg)
    n_gpus = int(agg.iloc[:int(len(agg) * TRAIN_R)]["gpu_n_pods"].median())
    feat["power_kw"] = estimate_power_kw(
        feat["gpu_util"].values, feat["active_pod_ratio"].values, n_gpus)
    model_df = feat.drop(columns=["power_kw"])
    feature_cols = list(model_df.columns)

    y_all = feat["gpu_util"].values
    gpu_roc = np.zeros_like(y_all)
    gpu_roc[1:] = np.diff(y_all)
    gpu_vol = rolling_std(y_all, 15)
    power_kw = feat["power_kw"].values
    power_roc = np.zeros_like(power_kw)
    power_roc[1:] = np.diff(power_kw)
    power_vol = rolling_std(power_kw, 15)

    from core.model_cache import ModelCache
    # Use the multistep cache (shared with multistep_averaged.py)
    cache = ModelCache(Path(__file__).resolve().parent.parent / "models" / "multistep")
    cache_label = f"{int(TRAIN_R*100)}-{int(VAL_R*100)}-{int((1-TRAIN_R-VAL_R)*100)}"

    results = {}
    n_loaded = 0
    n_trained = 0

    for h in HORIZONS:
        print(f"  h={h:>2}...", end=" ", flush=True)
        cache_name = f"transformer_h{h}_{cache_label}"

        # Prepare data (always needed for evaluation even if model is cached)
        y_all_raw = model_df["gpu_util"].values.astype(np.float32)
        X_all_raw = model_df[feature_cols].values
        y_fwd = forward_mean(y_all_raw, h)
        valid = ~np.isnan(y_fwd) & ~np.any(np.isnan(X_all_raw), axis=1)
        valid_indices = np.where(valid)[0]
        valid_indices = valid_indices[valid_indices >= WINDOW]
        n_valid = len(valid_indices)
        n_train_idx = int(n_valid * TRAIN_R)
        n_val_idx = int(n_valid * VAL_R)
        test_idx = valid_indices[n_train_idx + n_val_idx:]

        # Build model and try cache
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = build_model("transformer", len(feature_cols), hidden_dim=32)
        cached = cache.load(cache_name, model)

        if cached is not None:
            model, _, _ = cached
            res = _evaluate_cached(model, model_df, feature_cols, feat, h, n_gpus)
            n_loaded += 1
            src = "cached"
        else:
            res = train_for_horizon(model_df, feature_cols, h)
            n_trained += 1
            src = "trained"

        res = add_power_errors(res, feat, n_gpus, h)
        results[h] = res
        m_mae = float(np.mean(res["model_pwr_err"]))
        p_mae = float(np.mean(res["persist_pwr_err"]))
        print(f"[{src:>6}]  Power MAE: Model={m_mae:.4f} vs Persist={p_mae:.4f} kW  "
              f"({'✓' if m_mae < p_mae else '✗'})")

    print(f"\n  Cache: {n_loaded} loaded, {n_trained} trained ({n_loaded + n_trained} total)")
    return results, feat, y_all, gpu_roc, gpu_vol, power_kw, power_roc, power_vol, n_gpus


def _evaluate_cached(model, model_df, feature_cols, feat, horizon, n_gpus):
    """Evaluate a cached model on the test set for a given horizon."""
    y_all = model_df["gpu_util"].values.astype(np.float32)
    X_all = model_df[feature_cols].values
    y_fwd = forward_mean(y_all, horizon)
    valid = ~np.isnan(y_fwd) & ~np.any(np.isnan(X_all), axis=1)
    valid_indices = np.where(valid)[0]
    valid_indices = valid_indices[valid_indices >= WINDOW]
    n_valid = len(valid_indices)
    n_train = int(n_valid * TRAIN_R)
    n_val = int(n_valid * VAL_R)
    train_idx = valid_indices[:n_train]
    test_idx = valid_indices[n_train + n_val:]

    def make_set(indices):
        Xs = np.array([X_all[i - WINDOW:i] for i in indices], dtype=np.float32)
        return Xs, y_fwd[indices - 1].astype(np.float32), y_all[indices - 1].astype(np.float32)

    X_tr, _, _ = make_set(train_idx)
    X_te, yt_te, yc_te = make_set(test_idx)

    # Fit scalers on training data (deterministic)
    fsc = StandardScaler()
    fsc.fit(X_tr.reshape(-1, X_tr.shape[2]))
    X_te_s = fsc.transform(X_te.reshape(-1, X_te.shape[2])).reshape(X_te.shape)

    tsc = StandardScaler()
    _, yt_tr, yc_tr = make_set(train_idx)
    tsc.fit((yt_tr - yc_tr).reshape(-1, 1))

    # Evaluate
    device = get_device()
    model = model.to(device).eval()
    X_tensor = torch.tensor(X_te_s, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds_s = model(X_tensor).cpu().numpy()

    pred_deltas = tsc.inverse_transform(preds_s.reshape(-1, 1)).flatten()
    pred_avg = yc_te + pred_deltas
    true_avg = yt_te

    return {
        "test_idx": test_idx,
        "pred_avg": pred_avg,
        "true_avg": true_avg,
        "y_current": yc_te,
        "model_gpu_err": np.abs(pred_avg - true_avg),
        "persist_gpu_err": np.abs(yc_te - true_avg),
    }


# =====================================================================
# FIGURE 1: Multi-step power prediction — MAE, improvement, and R²
# =====================================================================

def fig_power_error_growth(results, feat_df, n_gpus):
    print("\n[Fig 1] Multi-step power prediction overview...")
    from sklearn.metrics import r2_score

    horizons = sorted(results.keys())
    model_maes = [float(np.mean(results[h]["model_pwr_err"])) for h in horizons]
    persist_maes = [float(np.mean(results[h]["persist_pwr_err"])) for h in horizons]
    improvements = [(p - m) / p * 100 if p > 0 else 0
                    for m, p in zip(model_maes, persist_maes)]

    # Compute R² for each horizon using instantaneous power ground truth
    all_true_power = estimate_power_kw(
        feat_df["gpu_util"].values, feat_df["active_pod_ratio"].values, n_gpus)
    model_r2 = []
    persist_r2 = []
    for h in horizons:
        res = results[h]
        idx = res["test_idx"]
        true_kw = np.array([np.mean(all_true_power[i:i + h]) for i in idx])
        pred_kw = estimate_power_kw(res["pred_avg"], res["apr_pred"], n_gpus)
        persist_kw = estimate_power_kw(res["y_current"], res["apr_pred"], n_gpus)
        model_r2.append(float(r2_score(true_kw, pred_kw)))
        persist_r2.append(float(r2_score(true_kw, persist_kw)))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1.2, 1.2]})

    # Panel 1: MAE curves
    ax1.plot(horizons, persist_maes, "o-", color="#888", lw=2, ms=8, label="Persistence")
    ax1.plot(horizons, model_maes, "s-", color="#E65100", lw=2, ms=8, label="Transformer")
    ax1.fill_between(horizons, model_maes, persist_maes, alpha=0.15, color="#2E7D32",
                     where=[m < p for m, p in zip(model_maes, persist_maes)])
    ax1.set_ylabel("Power MAE (kW)")
    ax1.set_title("Multi-Step Window-Averaged Power Prediction Across Forecast Horizons\n"
                  "(50/25/25 Split, Transformer, w=60)")
    ax1.legend(loc="upper left")

    # Panel 2: Improvement (stem plot to avoid bar overlap at close horizons)
    colors = ["#2E7D32" if imp > 0 else "#C62828" for imp in improvements]
    ax2.axhline(y=0, color="black", lw=0.8)
    for h, imp, c in zip(horizons, improvements, colors):
        ax2.plot([h, h], [0, imp], color=c, lw=2)
        ax2.plot(h, imp, "o", color=c, ms=6)
        ax2.text(h, imp + (1.5 if imp > 0 else -3.5), f"{imp:+.0f}%",
                ha="center", va="bottom" if imp > 0 else "top", fontsize=7)
    ax2.set_ylabel("MAE Improvement (%)")

    # Panel 3: R²
    ax3.plot(horizons, persist_r2, "o-", color="#888", lw=2, ms=8, label="Persistence")
    ax3.plot(horizons, model_r2, "s-", color="#E65100", lw=2, ms=8, label="Transformer")
    ax3.axhline(y=0, color="black", lw=0.5, ls="--", alpha=0.3)
    ax3.set_ylabel("$R^2$")
    ax3.set_xlabel("Forecast Horizon h (minutes)")
    ax3.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(R / "fig_multistep_power_growth.png", bbox_inches="tight")
    plt.close(fig)
    print("  → fig_multistep_power_growth.png")


# =====================================================================
# FIGURE 2: Per-sample power advantage at h=12 (combined view)
# =====================================================================

def fig_per_sample_advantage(results, power_kw):
    print("[Fig 2] Per-sample power advantage at h=12...")
    res = results[H_FOCUS]
    idx = res["test_idx"]
    advantage = res["pwr_advantage"]  # persist_err - model_err; positive = model wins
    t = np.arange(len(idx))

    fig, ax = plt.subplots(figsize=(14, 5))

    # Background: power trajectory (light gray, secondary axis)
    ax2 = ax.twinx()
    ax2.fill_between(t, power_kw[idx], alpha=0.07, color="#1565C0")
    ax2.plot(t, power_kw[idx], color="#1565C0", lw=0.6, alpha=0.35, label="Power (kW)")
    ax2.set_ylabel("Cluster Power (kW)", color="#1565C0", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#1565C0", labelsize=8)

    # Foreground: advantage as colored bars from zero
    colors = ["#2E7D32" if a > 0 else "#C62828" for a in advantage]
    ax.bar(t, advantage, color=colors, alpha=0.5, width=1.0, edgecolor="none")

    # Rolling average
    rw = max(5, len(t) // 20)
    kernel = np.ones(rw) / rw
    smooth = np.convolve(advantage, kernel, mode="same")
    ax.plot(t, smooth, color="#E65100", lw=2.5, label=f"Rolling avg ({rw}-sample)")
    ax.axhline(y=0, color="black", lw=0.8)

    ax.set_ylabel("Power Advantage over Persistence (kW)", fontsize=10)
    ax.set_xlabel("Test Sample Index")

    wins = int(np.sum(advantage > 0))
    total = len(advantage)
    mean_adv = float(np.mean(advantage))
    ax.set_title(
        f"Per-Sample Power Prediction Advantage at h={H_FOCUS} min (50/25/25)\n"
        f"Green bars = model outperforms persistence  |  Red bars = persistence outperforms  |  "
        f"Model wins {wins}/{total} samples, mean advantage {mean_adv:+.4f} kW",
        fontsize=11)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(R / "fig_locality_per_sample_advantage.png", bbox_inches="tight")
    plt.close(fig)
    print("  → fig_locality_per_sample_advantage.png")


# =====================================================================
# FIGURE 3: Best/worst cases at h=12 (ranked by model advantage)
# =====================================================================

def fig_best_worst(results, y_all, gpu_roc, gpu_vol, power_kw, n_gpus):
    print("[Fig 3] Best/worst cases at h=12...")
    res = results[H_FOCUS]
    idx = res["test_idx"]
    advantage = res["pwr_advantage"]  # persist_err - model_err

    # Rank by advantage: most positive = model outperforms most
    best_local = np.argsort(advantage)[-3:][::-1]   # top 3 advantage
    worst_local = np.argsort(advantage)[:3]          # bottom 3 (most negative)

    cases = (list(zip(best_local, ["Best", "2nd Best", "3rd Best"])) +
             list(zip(worst_local, ["Worst", "2nd Worst", "3rd Worst"])))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row in range(2):
        for col in range(3):
            ci = row * 3 + col
            local_idx, label = cases[ci]
            ax = axes[row][col]
            global_idx = idx[local_idx]
            pred_time = global_idx - 1

            # Context: 40 min before prediction + h=12 future + 5 min after
            ctx_start = max(0, pred_time - 40)
            ctx_end = min(len(y_all), global_idx + H_FOCUS + 5)
            t_ctx = np.arange(ctx_start, ctx_end)

            # Plot power trajectory
            pwr_ctx = power_kw[ctx_start:ctx_end]
            ax.plot(t_ctx, pwr_ctx, "#333", lw=1.5, label="Power (kW)")
            ax.axvline(x=pred_time, color="#1565C0", ls=":", lw=1.2, alpha=0.7,
                       label="Prediction time")

            # Shade future window
            future_t = np.arange(global_idx, global_idx + H_FOCUS)
            future_pwr = power_kw[global_idx:global_idx + H_FOCUS]
            ax.fill_between(future_t, min(pwr_ctx) * 0.95, max(pwr_ctx) * 1.05,
                           alpha=0.08, color="#2E7D32")
            ax.plot(future_t, future_pwr, "#2E7D32", lw=2, marker=".", ms=4,
                    label=f"Future {H_FOCUS} min")

            # Average power lines over the future window
            apr_p = res["apr_pred"][local_idx]
            pred_avg_pwr = float(estimate_power_kw(
                np.array([res["pred_avg"][local_idx]]), np.array([apr_p]), n_gpus)[0])
            persist_avg_pwr = float(estimate_power_kw(
                np.array([res["y_current"][local_idx]]), np.array([apr_p]), n_gpus)[0])
            # True avg power: average of instantaneous power over horizon
            all_pwr = estimate_power_kw(
                feat["gpu_util"].values, feat["active_pod_ratio"].values, n_gpus)
            gi = global_idx
            true_avg_pwr = float(np.mean(all_pwr[gi:gi + H_FOCUS]))

            ax.axhline(y=pred_avg_pwr, color="#E65100", ls="--", lw=2,
                        xmin=0.55, xmax=0.92, label=f"Model: {pred_avg_pwr:.2f} kW")
            ax.axhline(y=persist_avg_pwr, color="#888", ls="--", lw=1.2,
                        xmin=0.55, xmax=0.92, label=f"Persist: {persist_avg_pwr:.2f} kW")
            ax.axhline(y=true_avg_pwr, color="#2E7D32", ls="-", lw=1.5, alpha=0.5,
                        xmin=0.55, xmax=0.92, label=f"True avg: {true_avg_pwr:.2f} kW")

            # Annotate with data conditions
            gpu_level = y_all[pred_time]
            trend_5 = y_all[pred_time] - y_all[max(0, pred_time - 5)]
            vol = gpu_vol[pred_time] if not np.isnan(gpu_vol[pred_time]) else 0
            m_err = float(res["model_pwr_err"][local_idx])
            p_err = float(res["persist_pwr_err"][local_idx])
            adv = float(advantage[local_idx])

            trend_label = "Rising" if trend_5 > 0.01 else ("Falling" if trend_5 < -0.01 else "Stable")
            annotation = (f"Trend: {trend_label} (Δ₅={trend_5:+.3f})\n"
                          f"GPU={gpu_level:.3f}  σ₁₅={vol:.3f}\n"
                          f"Model err={m_err:.4f} kW\n"
                          f"Persist err={p_err:.4f} kW\n"
                          f"Advantage={adv:+.4f} kW")

            ax.text(0.02, 0.97, annotation, transform=ax.transAxes, fontsize=6.5,
                    va="top", ha="left", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.85, edgecolor="#ccc"))

            is_best = ci < 3
            color = "#2E7D32" if is_best else "#C62828"
            ax.set_title(f"{label}: advantage={adv:+.4f} kW", fontsize=10,
                        color=color, fontweight="bold")
            ax.set_xlabel("Time index")
            ax.set_ylabel("Power (kW)")
            if row == 0 and col == 0:
                ax.legend(fontsize=5.5, loc="lower left", framealpha=0.8)

    fig.suptitle(
        f"Where the Model Outperforms vs Underperforms Persistence at h={H_FOCUS} min\n"
        f"Ranked by advantage = |persist error| − |model error|  "
        f"(positive = model better, negative = persistence better)",
        fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(R / "fig_locality_best_worst.png", bbox_inches="tight")
    plt.close(fig)
    print("  → fig_locality_best_worst.png")


# =====================================================================
# TABLE GENERATION
# =====================================================================

def _write_locality_table(results, y_all, gpu_vol, regimes):
    """Generate markdown table for locality analysis at h=H_FOCUS."""
    res = results[H_FOCUS]
    idx = res["test_idx"]
    pred_idx = np.array([i - 1 for i in idx])
    advantage = res["pwr_advantage"]

    lines = []
    lines.append(f"## Locality Analysis: Power Prediction at h={H_FOCUS}")
    lines.append("")
    lines.append(f"**Configuration**: 50/25/25 split, w=60, Transformer, h={H_FOCUS}")
    lines.append(f"**Advantage** = |persistence error| − |model error| (positive = model better)")
    lines.append("")
    lines.append("### By Workload Volatility (15-min rolling σ)")
    lines.append("")
    lines.append("| Regime | Samples | Model Win Rate | Mean Advantage (kW) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name, mask in regimes[:3]:
        if mask.sum() > 0:
            wr = np.mean(advantage[mask] > 0) * 100
            ma = np.mean(advantage[mask])
            lines.append(f"| {name} | {mask.sum()} | {wr:.1f}% | {ma:+.4f} |")
    lines.append("")
    lines.append("### By Workload Trend (Δ over 5 min)")
    lines.append("")
    lines.append("| Regime | Samples | Model Win Rate | Mean Advantage (kW) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name, mask in regimes[3:]:
        if mask.sum() > 0:
            wr = np.mean(advantage[mask] > 0) * 100
            ma = np.mean(advantage[mask])
            lines.append(f"| {name} | {mask.sum()} | {wr:.1f}% | {ma:+.4f} |")
    lines.append("")
    lines.append("### Overall")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"| --- | --- |")
    lines.append(f"| Total test samples | {len(advantage)} |")
    lines.append(f"| Model wins | {np.sum(advantage > 0)}/{len(advantage)} ({np.mean(advantage > 0)*100:.0f}%) |")
    lines.append(f"| Mean advantage | {np.mean(advantage):+.4f} kW |")

    (R / "locality_table.md").write_text("\n".join(lines) + "\n")
    print(f"\n  → locality_table.md")


if __name__ == "__main__":
    main()
