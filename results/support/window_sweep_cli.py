"""CLI-specific helpers for the window sweep entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from .window_sweep_config import (
    DATASET_NAME,
    DEFAULT_EPOCHS,
    DEFAULT_PATIENCE,
    FIXED_WINDOW,
    MAX_HORIZON,
    N_CUTOFFS,
    OUTPUT_DIR,
    STEP_MINUTES,
    WINDOW_SIZES,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the results entry point."""
    parser = argparse.ArgumentParser(
        description="Window sweep study - run the full GenAI workflow from raw data to sweep results",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training; regenerate figures from saved JSON",
    )
    parser.add_argument(
        "--cluster-units",
        type=int,
        default=100,
        metavar="N",
        help="Cluster size used during power estimation in the raw-data processing stage (default: 100)",
    )
    parser.add_argument(
        "--rebuild-study-results",
        action="store_true",
        help="Ignore lookback/horizon JSON caches and rerun the sweep studies from scratch",
    )
    return parser.parse_args()


def resolve_device() -> str:
    """Return the best available accelerator for the fixed workflow."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_training_budget() -> tuple[int, int]:
    """Return the fixed training budget for the workflow."""
    return DEFAULT_EPOCHS, DEFAULT_PATIENCE


def print_run_configuration(
    device: str,
    epochs: int,
    patience: int,
) -> None:
    """Print a compact run summary before any training starts."""
    print("=" * 60)
    print("  Window Sweep Study")
    print("=" * 60)
    print(f"  Dataset      : {DATASET_NAME}")
    print(f"  Device       : {device}")
    print(f"  Epochs/run   : {epochs}  (early-stop patience={patience})")
    print(f"  Window sizes : {WINDOW_SIZES}")
    print(f"  Max horizon  : {MAX_HORIZON} steps ({MAX_HORIZON * STEP_MINUTES} min)")
    print(f"  Fixed window : {FIXED_WINDOW} steps (used for horizon sweep)")
    print(f"  Cutoffs      : {N_CUTOFFS}")
    print(f"  Output dir   : {OUTPUT_DIR.relative_to(OUTPUT_DIR.parents[1])}")
    print()


def print_data_and_cache_policy() -> None:
    """Print the data boundary and cache policy for this CLI."""
    print("[Data Boundary]")
    print(f"  Dataset is fixed to: {DATASET_NAME}")
    print("  Raw trace loading, aggregation, power estimation, and feature engineering live in src/data_processing/pipeline.py")
    print("  This entry script calls that pipeline path directly on every run before model training")
    print()
    print("[Cache Policy]")
    print("  Processed-data stage: always rebuilt from raw traces in this workflow")
    print("  Processed CSV snapshot: saved to data/processed for inspection, not reused by this workflow")
    print("  Lookback cache: results/window_sweep/lookback_results.json -> skips completed model/window runs")
    print("  Horizon cache: results/window_sweep/horizon_results.json -> skips completed fixed-window horizon runs")
    print("  Default sweep studies train fresh models whenever a JSON cache entry is missing")
    print("  Use --rebuild-study-results to ignore the JSON study caches and retrain/recompute the sweeps")
    print("  Default sweep studies do not reuse saved model checkpoints from models/saved")
    print("  Report mode may reuse fixed saved checkpoints when the feature schema matches")
    print("  Additional analysis trains fresh fixed-window models for loss curves and feature importance")
    print()


def load_cached_results(cache_path: Path) -> Optional[dict]:
    """Load cached JSON results from disk when they exist."""
    if not cache_path.exists():
        return None
    with open(cache_path) as handle:
        return json.load(handle)