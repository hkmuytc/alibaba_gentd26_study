from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys

def load_v3_model_bundle(model_name: str, horizon: int = 1) -> Optional[dict]:
    # Resolve the path to algorithmic_core cleanly
    v3_path = Path(__file__).resolve().parent.parent / "algorithmic_core" / "split_50_25_25_final"
    if str(v3_path) not in sys.path:
        sys.path.insert(0, str(v3_path))
        
    from core.models import build_model
    from core.model_cache import ModelCache
    
    target_model_name = model_name.lower().replace('-', '_') 
    v3_window = 60
    
    if horizon == 1:
        cache_dir = v3_path / "models" / "onestep"
        cache_name = f"{target_model_name}_w{v3_window}_50-25-25"
    else:
        cache_dir = v3_path / "models" / "multistep"
        cache_name = f"{target_model_name}_h{horizon}_50-25-25"
        
    cache = ModelCache(cache_dir)
    
    try:
        model = build_model(target_model_name, 27, hidden_dim=32)
        cached = cache.load(cache_name, model)
        if cached is None:
            return None
        model, scalers, meta = cached
        
        return {
            "model": model,
            "scaler": scalers["feat_scaler"],
            "target_scaler": scalers["tgt_scaler"],
            "meta": meta,
            "weights_path": cache._model_path(cache_name),
            "input_dim": 27,
            "horizon": horizon
        }
    except Exception as e:
        print("Error loading model:", e)
        return None