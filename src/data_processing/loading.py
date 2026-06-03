"""Shared data loading helpers for dashboard and evaluation flows."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")
_IDENTIFIER_EXACT_NAMES = {"::auto_unique_id::"}


def read_csv_portable(csv_path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """Read CSVs across platforms, falling back for legacy single-byte files."""
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(csv_path, encoding=encoding, **read_csv_kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error  # latin-1 should make this unreachable.


def is_processed_power_frame(df: pd.DataFrame) -> bool:
    return "power_total_kw" in df.columns and "timestamp" in df.columns


def looks_like_identifier_column(name: str) -> bool:
    normalized = str(name).strip().lower()
    if normalized in _IDENTIFIER_EXACT_NAMES:
        return True
    if normalized == "id" or normalized.endswith("_id"):
        return True
    if normalized.startswith("::") and normalized.endswith("::") and "id" in normalized:
        return True
    return False


def clean_processed_power_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in df.columns if looks_like_identifier_column(col)]
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols)


def dataset_short_name(dataset_key: str) -> str:
    key = str(dataset_key).lower()
    if "genai" in key or "gentd" in key:
        return "genai"
    if "v2020" in key:
        return "gpu_v2020"
    return key.replace(" (saved)", "").replace("_300s", "").split()[0]


def load_processed_datasets(
    session_data: Mapping[str, pd.DataFrame] | None = None,
    processed_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Return processed datasets from session state plus saved CSVs on disk."""
    available: dict[str, pd.DataFrame] = {}
    for key, value in (session_data or {}).items():
        if value is not None and len(value) > 0:
            available[key] = clean_processed_power_frame(value)

    source_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_DIR
    if not source_dir.exists():
        return available

    for csv_path in sorted(source_dir.glob("*_300s.csv")):
        label = csv_path.stem.replace("_300s", "") + " (saved)"
        base = csv_path.stem.split("_")[0]
        already_loaded = label in available or any(base in str(k) for k in available)
        if already_loaded:
            continue

        loaded = clean_processed_power_frame(read_csv_portable(csv_path))
        if is_processed_power_frame(loaded):
            available[label] = loaded

    return available