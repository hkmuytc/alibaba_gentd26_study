"""Model inference utilities used by the Forecast Replay page.

Provides:
  - load_model_bundle   : find the latest saved weights for a model/dataset pair
                          and return a ready-to-use bundle dict.
  - autoregressive_forecast : roll a trained model forward step-by-step.
  - forecast_metrics    : compute MAE/RMSE/MAPE/R² for a forecast window.
  - format_model_bundle_caption : human-readable summary of a loaded bundle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Locate the saved-models directory relative to this file.
# File lives at:  src/capstone_data_visualization/src/models/inference.py
# Models dir at:  src/capstone_data_visualization/models/saved/
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "saved"


def _power_roll_mean(window: np.ndarray, width: int) -> float:
    tail = window[-min(len(window), width):]
    return float(np.mean(tail))


def _power_roll_std(window: np.ndarray, width: int) -> float:
    tail = window[-min(len(window), width):]
    return float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0


def _advance_time_features(row: np.ndarray, feature_index: dict[str, int], next_ts: float) -> None:
    dt = np.datetime64(int(next_ts), "s")
    ts = np.datetime_as_string(dt, unit="s")
    from datetime import datetime

    parsed = datetime.fromisoformat(ts)
    seconds_in_day = parsed.hour * 3600 + parsed.minute * 60 + parsed.second
    day_of_week = parsed.weekday()

    if "hour_sin" in feature_index:
        row[feature_index["hour_sin"]] = np.sin(2 * np.pi * seconds_in_day / 86400)
    if "hour_cos" in feature_index:
        row[feature_index["hour_cos"]] = np.cos(2 * np.pi * seconds_in_day / 86400)
    if "dow_sin" in feature_index:
        row[feature_index["dow_sin"]] = np.sin(2 * np.pi * day_of_week / 7)
    if "dow_cos" in feature_index:
        row[feature_index["dow_cos"]] = np.cos(2 * np.pi * day_of_week / 7)
    if "hour" in feature_index:
        row[feature_index["hour"]] = float(parsed.hour)


def _update_power_features(
    row: np.ndarray,
    feature_index: dict[str, int],
    power_history_scaled: list[float],
    pred_scaled: float,
) -> None:
    power_history_scaled.append(float(pred_scaled))

    if "power_total_kw" in feature_index:
        row[feature_index["power_total_kw"]] = float(pred_scaled)
    if "power_total_kw_roc" in feature_index:
        prev_val = power_history_scaled[-2] if len(power_history_scaled) >= 2 else power_history_scaled[-1]
        row[feature_index["power_total_kw_roc"]] = float(pred_scaled - prev_val)
    if "power_total_kw_roll_mean_12" in feature_index:
        row[feature_index["power_total_kw_roll_mean_12"]] = _power_roll_mean(
            np.asarray(power_history_scaled, dtype=np.float64), 12
        )
    if "power_total_kw_roll_std_12" in feature_index:
        row[feature_index["power_total_kw_roll_std_12"]] = _power_roll_std(
            np.asarray(power_history_scaled, dtype=np.float64), 12
        )
    if "power_total_kw_roll_mean_72" in feature_index:
        row[feature_index["power_total_kw_roll_mean_72"]] = _power_roll_mean(
            np.asarray(power_history_scaled, dtype=np.float64), 72
        )
    if "power_total_kw_roll_std_72" in feature_index:
        row[feature_index["power_total_kw_roll_std_72"]] = _power_roll_std(
            np.asarray(power_history_scaled, dtype=np.float64), 72
        )


# --------------------------------------------------------------------------- #
# load_model_bundle                                                            #
# --------------------------------------------------------------------------- #

def load_model_bundle(
    model_name: str,
    dataset_key: str,
    df,
    target_col: str,
    window_size: int,
) -> Optional[dict]:
    """Find the latest saved weights for *model_name* on *dataset_key* and return
    a bundle ready for inference.

    The bundle dict contains:
        {
          "model":          nn.Module (eval mode, CPU),
          "feature_names":  list[str],
          "scaler":         StandardScaler fitted on training features,
          "target_scaler":  StandardScaler fitted on training target,
          "manifest":       dict  (raw manifest JSON),
          "weights_path":   Path,
        }

    Returns None if no matching weights file is found.
    """
    from src.models.architectures import get_model  # local import to avoid circular
    from src.evaluation.trainer import prepare_data  # noqa: F401 – refit scalers

    if not _MODELS_DIR.exists():
        return None

    # Prefer a timestamped file; fall back to the "latest" shortcut.
    ds_short = _dataset_short(dataset_key)
    candidates = sorted(_MODELS_DIR.glob(f"{model_name}_{ds_short}_w{window_size}_*_weights.pt"))
    if not candidates:
        # Try the "latest" shortcut written by save_model()
        shortcut = _MODELS_DIR / f"{model_name}_{ds_short}_weights.pt"
        if shortcut.exists():
            candidates = [shortcut]
    if not candidates:
        return None

    weights_path = candidates[-1]  # latest timestamp

    # Load manifest if it exists (gives us input_dim, window_size, etc.)
    manifest: dict = {}
    manifest_path = Path(str(weights_path).replace("_weights.pt", "_manifest.json"))
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    # Refit scalers on the training portion of *df* so they match what was used
    # when the model was trained.  We use the same prepare_data() logic.
    data = prepare_data(df, target_col=target_col, window_size=window_size, train_ratio=0.8)
    feature_names: list[str] = list(data["feature_names"])
    manifest_features = manifest.get("feature_names")
    compat_warning = None
    compat_error = None
    if manifest_features:
        feature_names = [name for name in manifest_features if name in df.columns]
        missing = [name for name in manifest_features if name not in df.columns]
        if missing:
            compat_warning = (
                "Saved model expects feature columns missing from the current dataset: "
                + ", ".join(missing)
            )
        if target_col not in manifest_features:
            compat_error = (
                f"Saved model was trained without '{target_col}' as an explicit input feature. "
                "Retrain the model with the current recursive-power schema before using Forecast Replay."
            )
    elif manifest:
        compat_error = (
            "Saved model predates the current feature-schema manifest. "
            "Retrain the model before using Forecast Replay."
        )

    # Prefer input_dim from the manifest so the architecture exactly matches the
    # saved weights (the current df may have a different number of features).
    manifest_dim: Optional[int] = manifest.get("input_dim")
    input_dim: int = manifest_dim if manifest_dim else data["input_dim"]

    # If the manifest dim differs from current features, we can still load the
    # model, but inference will need to pad/trim features to match.  Store a flag
    # in the bundle so callers can warn the user.
    dim_mismatch = manifest_dim is not None and manifest_dim != data["input_dim"]

    model = get_model(model_name, input_dim=input_dim)
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    return {
        "model": model,
        "feature_names": feature_names,
        "scaler": data["scaler"],
        "target_scaler": data["target_scaler"],
        "manifest": manifest,
        "weights_path": weights_path,
        "dim_mismatch": dim_mismatch,
        "input_dim": input_dim,
        "compat_warning": compat_warning,
        "compat_error": compat_error,
    }


# --------------------------------------------------------------------------- #
# autoregressive_forecast                                                      #
# --------------------------------------------------------------------------- #

def autoregressive_forecast(
    model,
    df,
    feature_names: list[str],
    target_col: str,
    scaler: StandardScaler,
    target_scaler: StandardScaler,
    window_size: int,
    cutoff_idx: int,
    horizon: int,
    device: str = "cpu",
) -> np.ndarray:
    """Roll *model* forward *horizon* steps starting at *cutoff_idx*.

    At each step the model predicts y[t+1] from the last *window_size* rows of
    features.  The predicted value is inserted back into the feature matrix
    wherever *target_col* appears (typically as a lagged feature), so errors
    compound — exactly like real deployment.

    Returns
    -------
    np.ndarray of shape (horizon,) with predictions in the original kW scale.
    """
    model = model.to(device)
    model.eval()

    # Build the full scaled feature matrix
    features = df[feature_names].values.astype(np.float64)
    features_scaled = scaler.transform(features)

    # If the model was trained with a different input_dim (e.g. different dataset
    # version), pad or trim the feature matrix so it matches the model weights.
    model_input_dim = next(iter(model.parameters())).shape[-1]  # first weight's last dim
    if features_scaled.shape[1] != model_input_dim:
        if features_scaled.shape[1] < model_input_dim:
            pad = np.zeros((features_scaled.shape[0], model_input_dim - features_scaled.shape[1]))
            features_scaled = np.hstack([features_scaled, pad])
        else:
            features_scaled = features_scaled[:, :model_input_dim]

    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    if target_col not in feature_index:
        raise ValueError(
            f"Feature set must include '{target_col}' for recursive replay. Retrain the model with the new schema."
        )

    # Seed the sliding window from the data that precedes the cutoff
    window = features_scaled[cutoff_idx - window_size : cutoff_idx].copy()  # (W, F)
    power_history_scaled = window[:, feature_index[target_col]].astype(np.float64).tolist()

    timestamps = df["timestamp"].values if "timestamp" in df.columns else None
    step_seconds = 300
    if timestamps is not None and len(timestamps) > 1:
        diffs = np.diff(timestamps.astype(np.float64))
        positive = diffs[diffs > 0]
        if len(positive) > 0:
            step_seconds = int(np.median(positive))

    preds_scaled: list[float] = []
    for step_idx in range(horizon):
        x = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(x)  # (1, 1) or (1,)
        pred_scaled = float(out.squeeze().cpu().item())
        preds_scaled.append(pred_scaled)

        # Shift window and append the new predicted step
        new_row = window[-1].copy()
        _update_power_features(new_row, feature_index, power_history_scaled, pred_scaled)
        if timestamps is not None:
            last_known_ts = float(df["timestamp"].iloc[cutoff_idx - 1])
            next_ts = last_known_ts + step_seconds * (step_idx + 1)
            _advance_time_features(new_row, feature_index, next_ts)
        window = np.vstack([window[1:], new_row])

    preds_raw = np.array(preds_scaled).reshape(-1, 1)
    return target_scaler.inverse_transform(preds_raw).flatten()


# --------------------------------------------------------------------------- #
# forecast_metrics                                                             #
# --------------------------------------------------------------------------- #

def forecast_metrics(actuals: np.ndarray, preds: np.ndarray) -> dict:
    """Return MAE, RMSE, MAPE, R² for a forecast window."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    actuals = np.asarray(actuals, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    # Trim to common length in case they differ by ±1
    n = min(len(actuals), len(preds))
    actuals, preds = actuals[:n], preds[:n]

    mask = actuals != 0
    mape = (
        float(np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100)
        if mask.any()
        else float("inf")
    )
    return {
        "MAE":  float(mean_absolute_error(actuals, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(actuals, preds))),
        "MAPE": mape,
        "R2":   float(r2_score(actuals, preds)) if n > 1 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# format_model_bundle_caption                                                  #
# --------------------------------------------------------------------------- #

def format_model_bundle_caption(bundle: dict) -> str:
    """Return a short markdown string describing a loaded model bundle."""
    if not bundle:
        return ""
    m = bundle.get("manifest", {})
    parts: list[str] = []
    if m.get("model_name"):
        parts.append(f"**{m['model_name']}**")
    if m.get("dataset"):
        parts.append(f"dataset `{m['dataset']}`")
    if m.get("window_size"):
        parts.append(f"window {m['window_size']}")
    if m.get("created"):
        parts.append(f"saved {m['created']}")
    if m.get("metrics"):
        met = m["metrics"]
        mae = met.get("MAE") or met.get("mae")
        if mae is not None:
            parts.append(f"train MAE {mae:.4f} kW")
    wp = bundle.get("weights_path")
    if wp:
        parts.append(f"`{Path(wp).name}`")
    return " · ".join(parts)


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _dataset_short(dataset_key: str) -> str:
    """Normalise a dataset label to the short form used in filenames.

    Handles labels like 'genai (saved)', 'GenTD26 (Primary)', 'gpu_v2020', etc.
    """
    dk = dataset_key.lower().replace(" (saved)", "").replace(" (primary)", "").replace(" (supplementary)", "")
    if "genai" in dk or "gentd" in dk:
        return "genai"
    if "v2020" in dk or "gpu_v2020" in dk:
        return "gpu_v2020"
    return dk.strip()
