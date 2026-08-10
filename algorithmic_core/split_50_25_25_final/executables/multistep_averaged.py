#!/usr/bin/env python3
"""
Window-averaged multi-step forecasting — CONSISTENT METHODOLOGY.

For each horizon h:
  - Target: mean(y[t+1..t+h]) — average GPU util over next h minutes
  - Persistence: y[t] — current value (same baseline as one-step)
  - Residual: target - y[t] = change from current to future average
  - Model learns: given features at time t, predict the change in average load

This is consistent with one-step where:
  - Target: y[t+1]
  - Persistence: y[t]
  - Residual: y[t+1] - y[t]
"""

import sys
import numpy as np
import torch
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.pipeline import (load_raw_signals, aggregate_to_cluster, engineer_features,
                      estimate_power_kw)
from core.models import build_model
from core.train import train_model, get_device, compute_metrics

R = Path(__file__).resolve().parent.parent / "results"

SEED = 0
WINDOW = 60
SPLITS = [(0.50, 0.25), (0.60, 0.20), (0.70, 0.15)]
HORIZONS = [1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 24]


def forward_mean(y, h):
    """mean(y[t+1..t+h]) for each t."""
    n = len(y)
    result = np.full(n, np.nan)
    for t in range(n - h):
        result[t] = np.mean(y[t + 1: t + h + 1])
    return result


def prepare_averaged_data(feat_df, feature_cols, window_size, horizon,
                          train_ratio, val_ratio):
    y_all = feat_df["gpu_util"].values.astype(np.float32)
    X_all = feat_df[feature_cols].values

    # Target: forward mean = mean(y[t+1..t+h])
    y_fwd = forward_mean(y_all, horizon)

    # Valid: forward mean defined, no NaN in features, enough history
    valid = ~np.isnan(y_fwd) & ~np.any(np.isnan(X_all), axis=1)
    valid_indices = np.where(valid)[0]
    valid_indices = valid_indices[valid_indices >= window_size]

    n_valid = len(valid_indices)
    n_train = int(n_valid * train_ratio)
    n_val = int(n_valid * val_ratio)

    train_idx = valid_indices[:n_train]
    val_idx = valid_indices[n_train:n_train + n_val]
    test_idx = valid_indices[n_train + n_val:]

    def make_set(indices):
        Xs = np.array([X_all[i - window_size:i] for i in indices], dtype=np.float32)
        y_targets = y_fwd[indices - 1].astype(np.float32)  # forward mean starting at t+1
        # Persistence: y[t] — current value (consistent with one-step)
        y_current = y_all[indices - 1].astype(np.float32)
        return Xs, y_targets, y_current

    X_tr, yt_tr, yc_tr = make_set(train_idx)
    X_va, yt_va, yc_va = make_set(val_idx)
    X_te, yt_te, yc_te = make_set(test_idx)

    # Scale features
    fsc = StandardScaler()
    n_feat = X_tr.shape[2]
    fsc.fit(X_tr.reshape(-1, n_feat))
    X_tr_s = fsc.transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
    X_va_s = fsc.transform(X_va.reshape(-1, n_feat)).reshape(X_va.shape)
    X_te_s = fsc.transform(X_te.reshape(-1, n_feat)).reshape(X_te.shape)

    # Scale RESIDUALS: target - y[t]
    residuals_tr = yt_tr - yc_tr
    tsc = StandardScaler()
    tsc.fit(residuals_tr.reshape(-1, 1))

    def to_scaled(yt, yc):
        r = yt - yc
        return tsc.transform(r.reshape(-1, 1)).flatten()

    yt_tr_s = to_scaled(yt_tr, yc_tr)
    yt_va_s = to_scaled(yt_va, yc_va)
    yt_te_s = to_scaled(yt_te, yc_te)

    class TSD(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]

    return {
        "train_loader": DataLoader(TSD(X_tr_s, yt_tr_s), batch_size=32, shuffle=True),
        "val_loader": DataLoader(TSD(X_va_s, yt_va_s), batch_size=32, shuffle=False),
        "test_loader": DataLoader(TSD(X_te_s, yt_te_s), batch_size=32, shuffle=False),
        "feat_scaler": fsc, "tgt_scaler": tsc,
        "yc_test": yc_te,   # current values (persistence)
        "yt_test": yt_te,   # actual forward means
        "test_indices": test_idx,
        "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te),
    }


def _write_multistep_table(all_results):
    """Generate markdown table for 50/25/25 split."""
    split = all_results.get("50/25/25", {})
    if not split:
        return

    lines = []
    lines.append("# Multi-Step Window-Averaged Power Prediction Results")
    lines.append("")
    lines.append("**Configuration**: 50/25/25 split, w=60, Transformer, residual prediction")
    lines.append("**Target**: mean(y[t+1..t+h]) — average GPU utilization over next h minutes")
    lines.append("**Persistence baseline**: y[t] — current value")
    lines.append("**Power**: strict Fan et al. utilization-to-power conversion; active-pod ratio is not used")
    lines.append("")
    lines.append("| h (min) | Model Power MAE (kW) | Persist Power MAE (kW) | Improvement | Model Power R² | Persist Power R² | Model Wins |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | --- |")

    for h_str in sorted(split.keys(), key=int):
        r = split[h_str]
        wins = "✓" if r["model_wins_power"] else "✗"
        lines.append(
            f"| {h_str} | {r['power_mae_kw']:.4f} | {r['power_persist_mae_kw']:.4f} "
            f"| {r['power_improvement_pct']:+.1f}% | {r['power_r2']:.4f} "
            f"| {r['power_persist_r2']:.4f} | {wins} |")

    wins_total = sum(1 for h in split.values() if h["model_wins_power"])
    lines.append("")
    lines.append(f"## Model wins: {wins_total}/{len(split)} horizons (power MAE)")

    (R / "multistep_power_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  → multistep_power_table.md")


def run_all():
    from core.model_cache import ModelCache
    cache = ModelCache(Path(__file__).resolve().parent.parent / "models" / "multistep")

    gpu, gmem, qps = load_raw_signals()
    agg = aggregate_to_cluster(gpu, gmem, qps, bin_sec=60)
    feat = engineer_features(agg)
    feat["power_kw"] = estimate_power_kw(
        feat["gpu_util"].values, feat["active_pod_ratio"].values,
        int(agg["gpu_n_pods"].median()))  # for feature df only
    model_df = feat.drop(columns=["power_kw"])
    feature_cols = list(model_df.columns)

    all_results = {}
    n_loaded = 0
    n_trained = 0

    for train_r, val_r in SPLITS:
        label = f"{int(train_r*100)}/{int(val_r*100)}/{int((1-train_r-val_r)*100)}"
        cache_label = f"{int(train_r*100)}-{int(val_r*100)}-{int((1-train_r-val_r)*100)}"
        train_idx = feat.index[:int(len(feat) * train_r)]
        n_gpus = int(agg.loc[train_idx, "gpu_n_pods"].median())
        all_results[label] = {}
        print(f"\n{'='*60}")
        print(f"  Split: {label}")
        print(f"{'='*60}")

        for h in HORIZONS:
            print(f"  h={h:>2}...", end=" ", flush=True)
            data = prepare_averaged_data(
                model_df, feature_cols, WINDOW, h, train_r, val_r)

            cache_name = f"transformer_h{h}_{cache_label}"

            torch.manual_seed(SEED); np.random.seed(SEED)
            model = build_model("transformer", len(feature_cols), hidden_dim=32)
            cached = cache.load(cache_name, model)

            if cached is not None:
                model, _, _ = cached
                n_loaded += 1
                src = "cached"
            else:
                model, hist = train_model(
                    model, data["train_loader"], data["val_loader"],
                    epochs=200, patience=20, lr=1e-3, weight_decay=1e-4)
                n_trained += 1
                src = "trained"

            # Evaluate
            device = get_device()
            model = model.to(device).eval()
            preds_s, tgts_s = [], []
            with torch.no_grad():
                for Xb, yb in data["test_loader"]:
                    preds_s.append(model(Xb.to(device)).cpu().numpy())
                    tgts_s.append(yb.numpy())
            preds_s = np.concatenate(preds_s)
            tgts_s = np.concatenate(tgts_s)

            # Inverse transform residuals
            pred_deltas = data["tgt_scaler"].inverse_transform(
                preds_s.reshape(-1, 1)).flatten()
            true_deltas = data["tgt_scaler"].inverse_transform(
                tgts_s.reshape(-1, 1)).flatten()

            # Reconstruct levels
            y_current = data["yc_test"]
            pred_avg = y_current + pred_deltas
            true_avg = y_current + true_deltas

            # Persistence: y[t] (current value)
            persist_avg = y_current

            # GPU util metrics
            m_mae = float(np.mean(np.abs(pred_avg - true_avg)))
            p_mae = float(np.mean(np.abs(persist_avg - true_avg)))
            m_r2 = float(r2_score(true_avg, pred_avg))
            p_r2 = float(r2_score(true_avg, persist_avg))

            # Power metrics
            test_indices = data["test_indices"]
            prev_indices = [i - 1 for i in test_indices]
            apr_pred = feat["active_pod_ratio"].values[prev_indices]

            # Ground truth: average of instantaneous strict Fan power over the horizon.
            all_true_power = estimate_power_kw(
                feat["gpu_util"].values, feat["active_pod_ratio"].values, n_gpus)
            true_kw = np.array([np.mean(all_true_power[i:i + h]) for i in test_indices])

            pred_kw = estimate_power_kw(pred_avg, apr_pred, n_gpus)
            persist_kw = estimate_power_kw(persist_avg, apr_pred, n_gpus)
            m_power = compute_metrics(true_kw, pred_kw)
            p_power = compute_metrics(true_kw, persist_kw)

            wins = m_mae < p_mae
            wins_power = float(m_power["MAE"]) < float(p_power["MAE"])
            imp = (p_mae - m_mae) / p_mae * 100 if p_mae > 0 else 0
            pwr_imp = (float(p_power["MAE"]) - float(m_power["MAE"])) / float(p_power["MAE"]) * 100 if float(p_power["MAE"]) > 0 else 0

            all_results[label][str(h)] = {
                "gpu_mae": round(m_mae, 6),
                "gpu_persist_mae": round(p_mae, 6),
                "gpu_r2": round(m_r2, 4),
                "gpu_persist_r2": round(p_r2, 4),
                "power_mae_kw": round(float(m_power["MAE"]), 4),
                "power_persist_mae_kw": round(float(p_power["MAE"]), 4),
                "power_r2": round(float(m_power["R2"]), 4),
                "power_persist_r2": round(float(p_power["R2"]), 4),
                "mae_improvement_pct": round(imp, 1),
                "power_improvement_pct": round(pwr_imp, 1),
                "model_wins": wins,
                "model_wins_power": wins_power,
            }
            # Save to cache if newly trained
            if src == "trained":
                scalers = {"feat_scaler": data["feat_scaler"], "tgt_scaler": data["tgt_scaler"]}
                cache.save(cache_name, model, scalers, {
                    "gpu_mae": m_mae, "power_mae_kw": float(m_power["MAE"]),
                    "gpu_r2": m_r2, "power_r2": float(m_power["R2"])})

            gpu_tag = "✓" if wins else "✗"
            pwr_tag = "✓" if wins_power else "✗"
            print(f"[{src:>6}]  Model MAE={m_mae:.4f} vs Persist MAE={p_mae:.4f} ({imp:+.1f}%) [{gpu_tag}]  "
                  f"Power MAE={m_power['MAE']:.4f} vs {p_power['MAE']:.4f} kW ({pwr_imp:+.1f}%) [{pwr_tag}]")

    print(f"\n  Cache: {n_loaded} loaded, {n_trained} trained ({n_loaded + n_trained} total)")

    # Save JSON
    with open(R / "averaged_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Generate markdown table
    _write_multistep_table(all_results)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Window-Averaged Multi-Step (y[t] persistence)")
    print("  Target: mean(y[t+1..t+h])")
    print("  Persistence: y[t] (current value, same as one-step)")
    print("  Residual: change from current to future average")
    print(f"{'='*70}")
    for sl, horizons in all_results.items():
        wins_gpu = sum(1 for h in horizons.values() if h["model_wins"])
        wins_pwr = sum(1 for h in horizons.values() if h["model_wins_power"])
        print(f"\n  Split {sl}: Model wins {wins_gpu}/{len(horizons)} GPU MAE, "
              f"{wins_pwr}/{len(horizons)} Power MAE")
        print(f"  {'h':>4} {'Model MAE':>10} {'Persist MAE':>12} {'Δ MAE':>8}  "
              f"{'Pwr MAE':>9} {'Pwr Persist':>11} {'Δ Pwr':>8}  {'GPU':>4} {'Pwr':>4}")
        print(f"  {'-'*78}")
        for h_str in sorted(horizons.keys(), key=int):
            r = horizons[h_str]
            gpu_tag = "✓" if r["model_wins"] else ""
            pwr_tag = "✓" if r["model_wins_power"] else ""
            print(f"  {h_str:>4} {r['gpu_mae']:>10.4f} {r['gpu_persist_mae']:>12.4f} "
                  f"{r['mae_improvement_pct']:>+7.1f}%  "
                  f"{r['power_mae_kw']:>9.4f} {r['power_persist_mae_kw']:>11.4f} "
                  f"{r['power_improvement_pct']:>+7.1f}%  {gpu_tag:>4} {pwr_tag:>4}")

    print(f"\n✓ Results saved to {R / 'averaged_results.json'}")


if __name__ == "__main__":
    run_all()
