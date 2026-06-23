"""Model persistence helpers for the window sweep workflow."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import torch

from src.models.architectures import build_model_kwargs
from src.models.architectures import get_model

from .window_sweep_config import DATASET_NAME, DROPOUT, HIDDEN_DIM, NUM_LAYERS, PROJECT_ROOT


MODELS_DIR = PROJECT_ROOT / "models" / "saved"


def save_trained_model_window(
    model,
    model_name: str,
    metrics: dict,
    history: dict,
    input_dim: int,
    feature_names: list[str],
    window_size: int,
) -> None:
    """Persist one trained model as a window-specific saved artifact set."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{model_name}_{DATASET_NAME}_w{window_size}_{timestamp}"
    weights_path = MODELS_DIR / f"{prefix}_weights.pt"
    metrics_path = MODELS_DIR / f"{prefix}_metrics.json"
    history_path = MODELS_DIR / f"{prefix}_history.json"
    manifest_path = MODELS_DIR / f"{prefix}_manifest.json"

    torch.save(model.state_dict(), weights_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    manifest = {
        "model_name": model_name,
        "dataset": DATASET_NAME,
        "window_size": window_size,
        "train_ratio": 0.8,
        "input_dim": input_dim,
        "feature_names": list(feature_names),
        "created": timestamp,
        "epochs_trained": len(history.get("train_loss", [])),
        "metrics": metrics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_window_model(model_name: str, input_dim: int):
    """Construct one model instance with the workflow's fixed architecture defaults."""
    kwargs = build_model_kwargs(model_name, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    return get_model(model_name, input_dim=input_dim, **kwargs)


def load_saved_model_window(model_name: str, input_dim: int, feature_names: list[str], window_size: int):
    """Load the latest saved model run for one architecture/window pair if available."""
    manifest_path = _latest_window_manifest_path(model_name, window_size)
    if manifest_path is None:
        return None

    with open(manifest_path) as handle:
        manifest = json.load(handle)

    manifest_features = manifest.get("feature_names")
    if manifest.get("input_dim") != input_dim or list(manifest_features or []) != list(feature_names):
        return None

    weights_path = Path(str(manifest_path).replace("_manifest.json", "_weights.pt"))
    metrics_path = Path(str(manifest_path).replace("_manifest.json", "_metrics.json"))
    history_path = Path(str(manifest_path).replace("_manifest.json", "_history.json"))
    if not weights_path.exists():
        return None

    model = build_window_model(model_name, input_dim)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    history = json.loads(history_path.read_text()) if history_path.exists() else {"train_loss": [], "val_loss": []}
    return {
        "model": model,
        "metrics": metrics,
        "history": history,
        "manifest": manifest,
        "weights_path": weights_path,
    }


def trained_model_label(model_name: str, window_size: int) -> str:
    """Return a short label for logs around one trained model artifact."""
    return f"{model_name} @ w={window_size}"


def _latest_window_manifest_path(model_name: str, window_size: int) -> Path | None:
    """Return the latest timestamped manifest for one model/window pair."""
    pattern = f"{model_name}_{DATASET_NAME}_w{window_size}_*_manifest.json"
    candidates = sorted(MODELS_DIR.glob(pattern))
    return candidates[-1] if candidates else None