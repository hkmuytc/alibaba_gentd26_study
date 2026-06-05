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
    import pandas as pd

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

    # We also need the *target* column index inside feature_names if it's there
    # (some pipelines include the target as a feature for its own lags).
    target_in_feats = target_col in feature_names
    target_feat_idx = feature_names.index(target_col) if target_in_feats else None

    # Seed the sliding window from the data that precedes the cutoff
    window = features_scaled[cutoff_idx - window_size : cutoff_idx].copy()  # (W, F)

    preds_scaled: list[float] = []
    for _ in range(horizon):
        x = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(x)  # (1, 1) or (1,)
        pred_scaled = float(out.squeeze().cpu().item())
        preds_scaled.append(pred_scaled)

        # Shift window and append the new predicted step
        new_row = window[-1].copy()
        if target_feat_idx is not None:
            new_row[target_feat_idx] = pred_scaled
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
