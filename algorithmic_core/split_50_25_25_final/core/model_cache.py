"""
Model cache: save, load, and compare trained models with their scalers and metrics.

Usage:
    from model_cache import ModelCache

    cache = ModelCache("models/onestep")

    # Try loading
    cached = cache.load("transformer_w60_70-15-15")
    if cached is not None:
        model, scalers, meta = cached
    else:
        # Train...
        cache.save("transformer_w60_70-15-15", model, scalers, {"mae": 0.028, "r2": 0.89})

    # Compare: returns True if new is better
    if cache.is_better("transformer_w60_70-15-15", new_metric=0.027, key="mae"):
        # overwrite results
"""

import hashlib
import json
import torch
import joblib
from pathlib import Path


def _json_safe(value):
    """Convert numpy scalar metrics to plain JSON values."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


class ModelCache:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _model_path(self, name):
        return self.cache_dir / f"{name}.pt"

    def _scaler_path(self, name):
        return self.cache_dir / f"{name}_scalers.joblib"

    def _meta_path(self):
        return self.cache_dir / "metadata.json"

    def _load_meta(self):
        p = self._meta_path()
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_meta(self, meta):
        with open(self._meta_path(), "w") as f:
            json.dump(meta, f, indent=2)

    def save(self, name, model, scalers, metrics):
        """Save model weights, scalers, and metrics."""
        torch.save(model.state_dict(), self._model_path(name))
        joblib.dump(scalers, self._scaler_path(name))

        meta = self._load_meta()
        meta[name] = _json_safe(metrics)
        self._save_meta(meta)

    def load(self, name, model):
        """Load model weights and scalers. Returns (model, scalers, metrics) or None."""
        model_path = self._model_path(name)
        scaler_path = self._scaler_path(name)
        if not model_path.exists() or not scaler_path.exists():
            return None

        meta = self._load_meta()
        if name not in meta:
            return None

        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        scalers = joblib.load(scaler_path)
        return model, scalers, meta.get(name, {})

    def is_better(self, name, new_metric, key="mae", lower_is_better=True):
        """Check if new_metric is better than the cached metric."""
        meta = self._load_meta()
        if name not in meta or key not in meta[name]:
            return True  # no previous result → always save
        old_metric = meta[name][key]
        if lower_is_better:
            return new_metric < old_metric
        return new_metric > old_metric

    def get_metric(self, name, key="mae"):
        """Get a cached metric for a model."""
        meta = self._load_meta()
        if name in meta and key in meta[name]:
            return meta[name][key]
        return None

    def exists(self, name):
        """Check if a cached model exists."""
        return self._model_path(name).exists() and self._scaler_path(name).exists()


def code_hash(filepath):
    """Compute MD5 hash of a Python file for change detection."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
