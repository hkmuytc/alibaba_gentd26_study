"""Model persistence helpers for the window sweep workflow."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.trainer import load_model, save_model
from src.models.architectures import build_model_kwargs

from .window_sweep_config import DATASET_NAME, DROPOUT, HIDDEN_DIM, NUM_LAYERS


def save_trained_model_window(
    model,
    model_name: str,
    metrics: dict,
    history: dict,
    input_dim: int,
    feature_names: list[str],
    window_size: int,
) -> None:
    """Persist one trained model using the shared trainer save format."""
    save_model(
        model,
        model_name,
        DATASET_NAME,
        metrics,
        history,
        window_size=window_size,
        train_ratio=0.8,
        input_dim=input_dim,
        feature_names=feature_names,
    )


def load_saved_model_window(model_name: str, input_dim: int, window_size: int):
    """Load the latest saved model for one architecture/window pair if available."""
    kwargs = build_model_kwargs(model_name, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    try:
        model, metrics = load_model(model_name, DATASET_NAME, input_dim, **kwargs)
    except FileNotFoundError:
        return None, None
    except Exception:
        return None, None
    return model, metrics


def trained_model_label(model_name: str, window_size: int) -> str:
    """Return a short label for logs around one trained model artifact."""
    return f"{model_name} @ w={window_size}"