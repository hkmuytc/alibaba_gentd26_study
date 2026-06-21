"""Shared configuration for the window sweep experiment runner."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME: str = "genai"

WINDOW_SIZES: list[int] = [4, 8, 12, 16, 24, 36, 48, 72]
FIXED_WINDOW: int = 24
MAX_HORIZON: int = 24
N_CUTOFFS: int = 50

HIDDEN_DIM: int = 64
NUM_LAYERS: int = 2
DROPOUT: float = 0.2

TARGET_COL: str = "power_total_kw"
TRAIN_RATIO: float = 0.8
STEP_MINUTES: int = 5

DEFAULT_EPOCHS: int = 100
DEFAULT_PATIENCE: int = 10
QUICK_EPOCHS: int = 30
QUICK_PATIENCE: int = 5

OUTPUT_DIR: Path = PROJECT_ROOT / "results" / "window_sweep"

MODEL_COLORS: dict[str, str] = {
    "LSTM": "#1f77b4",
    "GRU": "#ff7f0e",
    "Transformer": "#2ca02c",
    "Persistence": "#333333",
}
MODEL_NAMES: list[str] = ["LSTM", "GRU", "Transformer"]
