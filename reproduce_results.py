import sys
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.pipeline import build_genai_power_series
from src.evaluation.trainer import prepare_data, train_model, evaluate_model
from src.models.architectures import get_model
from src.models.inference import autoregressive_forecast

MODELS_DIR     = PROJECT_ROOT / "models" / "saved"
TARGET_COL     = "power_total_kw"
WINDOW_SIZE    = 24     # 24 steps × 5 min = 2 hours of history
TRAIN_RATIO    = 0.8
REPLAY_HORIZON = 12     # 12 steps × 5 min = 60-min forecast horizon
REPLAY_N_CUTOFFS = 43
MODEL_NAMES    = ["LSTM", "GRU", "Transformer"]

# Best saved weights per model (genai dataset, selected by R²)
BEST_WEIGHTS = {
    "GRU":         "GRU_genai_w24_20260521_104305_weights.pt",           # R²=0.587
    "LSTM":        "LSTM_genai_w24_20260521_104257_weights.pt",          # best available LSTM
    "Transformer": "Transformer_genai_w24_20260521_104306_weights.pt",   # R²=0.394
}


def main():
    print("=" * 60)
    print("  Reproducing Interim Report Results")
    print("=" * 60)

    df   = load_processed_data()
    data = split_and_scale(df)

    one_step_results  = evaluate_one_step(data)
    replay_results    = evaluate_replay(df, data, one_step_results)

    print_results(one_step_results, replay_results)


def load_processed_data():
    print("\n[1/4] Loading GenTD26 power series...")
    cached = PROJECT_ROOT / "data" / "processed" / "genai_300s.csv"
    if cached.exists():
        import pandas as pd
        df = pd.read_csv(cached)
        print(f"  Loaded cached CSV: {len(df)} rows")
    else:
        df = build_genai_power_series()
        print(f"  Built from raw traces: {len(df)} rows")
    return df

def split_and_scale(df):
    print("\n[2/4] Preparing 80/20 train/test split...")
    data = prepare_data(df, target_col=TARGET_COL, window_size=WINDOW_SIZE, train_ratio=TRAIN_RATIO)
    print(f"  Features: {data['input_dim']}  |  "
          f"Train: {len(data['train_loader'].dataset)}  |  "
          f"Test:  {len(data['test_loader'].dataset)}")
    return data


def evaluate_one_step(data):
    print("\n[3/4] One-step forecasting...")
    results = {}
    for name in MODEL_NAMES:
        model = load_or_train(name, data)
        preds, actuals, metrics = evaluate_model(model, data["test_loader"], data["target_scaler"])
        results[name] = {"model": model, "metrics": metrics, "preds": preds, "actuals": actuals}
        print(f"  {name:<12} MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  "
              f"MAPE={metrics['MAPE']:.2f}%  R²={metrics['R2']:.3f}")

    results["baseline"] = {"metrics": persistence_one_step_metrics(data["raw_targets_test"])}
    m = results["baseline"]["metrics"]
    print(f"  {'Baseline':<12} MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  "
          f"MAPE={m['MAPE']:.2f}%  R²={m['R2']:.3f}")
    return results

def persistence_one_step_metrics(raw_actuals):
    preds   = raw_actuals[:-1]
    actuals = raw_actuals[1:]
    return compute_metrics(actuals, preds)


def evaluate_replay(df, data, one_step_results):
    print("\n[4/4] 12-step autoregressive replay (60-min horizon)...")
    cutoff_indices = build_cutoff_indices(df, data["split_idx"])
    raw_power      = df[TARGET_COL].values.astype(np.float32)
    feature_names  = list(data["feature_names"])
    results        = {}

    for name in MODEL_NAMES:
        model = one_step_results[name]["model"]
        window_results = run_replay_windows(
            model, df, feature_names, raw_power, data, cutoff_indices
        )
        if not window_results:
            print(f"  {name}: no valid windows")
            continue
        results[name] = summarise_replay_windows(window_results)
        r = results[name]
        print(f"  {name:<12} MAE={r['avg_mae']:.3f}  Baseline MAE={r['baseline_mae']:.3f}  "
              f"Improvement={r['improvement']:.1f}%  Win rate={r['win_rate']:.1f}%  "
              f"({r['n_windows']} windows)")

    return results

def build_cutoff_indices(df, split_idx):
    first = split_idx + WINDOW_SIZE
    last  = len(df) - REPLAY_HORIZON
    return np.linspace(first, last, REPLAY_N_CUTOFFS, dtype=int)

def run_replay_windows(model, df, feature_names, raw_power, data, cutoff_indices):
    window_results = []
    for cutoff in cutoff_indices:
        if cutoff + REPLAY_HORIZON > len(df):
            continue
        result = evaluate_single_replay_window(model, df, feature_names, raw_power, data, cutoff)
        if result is not None:
            window_results.append(result)
    return window_results

def evaluate_single_replay_window(model, df, feature_names, raw_power, data, cutoff):
    try:
        preds = autoregressive_forecast(
            model=model,
            df=df,
            feature_names=feature_names,
            target_col=TARGET_COL,
            scaler=data["scaler"],
            target_scaler=data["target_scaler"],
            window_size=WINDOW_SIZE,
            cutoff_idx=int(cutoff),
            horizon=REPLAY_HORIZON,
        )
    except Exception as e:
        print(f"    Replay error at cutoff {cutoff}: {e}")
        return None

    truths   = raw_power[cutoff: cutoff + REPLAY_HORIZON]
    n        = min(len(truths), len(preds))
    mae_model    = float(mean_absolute_error(truths[:n], preds[:n]))
    mae_baseline = float(mean_absolute_error(truths[:n], np.full(n, raw_power[cutoff - 1])))
    return {"mae_model": mae_model, "mae_baseline": mae_baseline}

def summarise_replay_windows(window_results):
    model_maes    = [r["mae_model"]    for r in window_results]
    baseline_maes = [r["mae_baseline"] for r in window_results]
    wins          = sum(r["mae_model"] < r["mae_baseline"] for r in window_results)
    avg_mae       = float(np.mean(model_maes))
    avg_base      = float(np.mean(baseline_maes))
    return {
        "avg_mae":      round(avg_mae, 3),
        "baseline_mae": round(avg_base, 3),
        "improvement":  round((avg_base - avg_mae) / avg_base * 100, 1),
        "win_rate":     round(wins / len(window_results) * 100, 1),
        "n_windows":    len(window_results),
    }


def print_results(one_step_results, replay_results):
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print_one_step_table(one_step_results)
    print_replay_table(replay_results)
    print("\nDone.")

def print_one_step_table(results):
    bm = results["baseline"]["metrics"]
    rows = [("Repeat-last baseline", f"{bm['MAE']:.3f}", f"{bm['RMSE']:.3f}",
             f"{bm['MAPE']:.2f}", f"{bm['R2']:.3f}")]
    for name in MODEL_NAMES:
        m = results[name]["metrics"]
        rows.append((name, f"{m['MAE']:.3f}", f"{m['RMSE']:.3f}",
                     f"{m['MAPE']:.2f}", f"{m['R2']:.3f}"))
    print_table("Table 1 — One-Step Forecasting Results", rows,
                ["Model", "MAE", "RMSE", "MAPE (%)", "R²"])

def print_replay_table(results):
    rows = []
    for name in MODEL_NAMES:
        if name not in results:
            continue
        r = results[name]
        rows.append((name, f"{r['avg_mae']:.3f}", f"{r['baseline_mae']:.3f}",
                     f"{r['improvement']:.1f}%", f"{r['win_rate']:.1f}%"))
    print_table("Table 2 — 12-Step Replay Results (60-min horizon)", rows,
                ["Model", "12-step MAE", "Baseline MAE", "MAE Improvement", "Win Rate"])


def load_or_train(model_name, data):
    model = load_saved_weights(model_name, data)
    if model is not None:
        return model
    print(f"  No saved weights for {model_name} — training from scratch...")
    model = get_model(model_name, input_dim=data["input_dim"])
    train_model(model, data["train_loader"], val_loader=data["test_loader"],
                epochs=100, device="cpu")
    return model

def load_saved_weights(model_name, data):
    if model_name not in BEST_WEIGHTS:
        return None
    weights_path = MODELS_DIR / BEST_WEIGHTS[model_name]
    if not weights_path.exists():
        return None

    input_dim    = read_input_dim_from_manifest(weights_path, fallback=data["input_dim"])
    model        = get_model(model_name, input_dim=input_dim)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    print(f"  {model_name:<12} Loaded: {weights_path.name}")
    return model

def read_input_dim_from_manifest(weights_path, fallback):
    manifest_path = Path(str(weights_path).replace("_weights.pt", "_manifest.json"))
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f).get("input_dim", fallback)
    return fallback

def compute_metrics(actuals, preds):
    actuals, preds = np.asarray(actuals), np.asarray(preds)
    mae  = float(mean_absolute_error(actuals, preds))
    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mask = actuals != 0
    mape = float(np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100) if mask.sum() > 0 else float("inf")
    r2   = float(r2_score(actuals, preds))
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE": round(mape, 2), "R2": round(r2, 3)}

def print_table(title, rows, headers):
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)
    print(f"\n{title}")
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


if __name__ == "__main__":
    main()

