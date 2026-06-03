"""High-level symbolic-regression workflow helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_processing.loading import looks_like_identifier_column

from .symbolic import (
    build_lagged_design_matrix,
    build_multi_horizon_design_matrix,
    evaluate_multistep,
    fit_symbolic,
)


DEFAULT_SYMBOLIC_EXOGENOUS = (
    "gpu_util",
    "qps",
    "mem_util",
    "machine_cpu",
    "machine_gpu",
    "hour_sin",
    "hour_cos",
)


@dataclass(frozen=True)
class SymbolicRunConfig:
    target_col: str = "power_total_kw"
    lags: tuple[int, ...] = (1, 2, 3, 6, 12)
    rolling_means: tuple[int, ...] = (6, 12)
    extra_features: tuple[str, ...] = ()
    train_ratio: float = 0.8
    generations: int = 20
    population_size: int = 500
    parsimony: float = 0.001
    function_set: tuple[str, ...] = ("add", "sub", "mul", "div", "sin")
    predict_residual: bool = True
    optimization_horizon: int = 12
    train_across_horizon_window: bool = True


def parse_positive_ints(text: str, label: str, allow_empty: bool = False) -> tuple[int, ...]:
    values = []
    for part in text.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        value = int(cleaned)
        if value <= 0:
            raise ValueError(f"{label} values must be positive integers.")
        values.append(value)

    if not values and not allow_empty:
        raise ValueError(f"Provide at least one {label} value.")
    return tuple(values)


def symbolic_exogenous_candidates(
    df: pd.DataFrame,
    target_col: str = "power_total_kw",
    limit: int = 30,
) -> list[str]:
    candidates = [
        col for col in df.columns
        if col not in ("timestamp", target_col)
        and target_col not in col
        and not looks_like_identifier_column(col)
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    return candidates[:limit]


def default_symbolic_exogenous(candidates: list[str]) -> list[str]:
    return [col for col in DEFAULT_SYMBOLIC_EXOGENOUS if col in candidates]


def run_symbolic_regression(df: pd.DataFrame, config: SymbolicRunConfig) -> dict:
    if config.predict_residual and 1 not in config.lags:
        raise ValueError("Residual mode needs `1` in target lags for the y[t-1] baseline.")
    if not config.function_set:
        raise ValueError("Select at least one symbolic operator.")

    split_idx = int(len(df) * config.train_ratio)
    horizon_col = "horizon_steps"
    train_mask = None
    horizon_mode = "single-step"

    if config.train_across_horizon_window and config.optimization_horizon > 1:
        horizons = tuple(range(1, config.optimization_horizon + 1))
        X, y, meta = build_multi_horizon_design_matrix(
            df,
            target_col=config.target_col,
            horizons=horizons,
            lags=config.lags,
            rolling_means=config.rolling_means,
            extra_features=config.extra_features,
            horizon_col=horizon_col,
        )
        target_indices = meta["source_index"] + meta["horizon"] - 1
        train_mask = target_indices < split_idx
        train_end = int(train_mask.sum())
        horizon_mode = f"direct horizon window 1..{config.optimization_horizon}"
    else:
        X, y, valid = build_lagged_design_matrix(
            df,
            target_col=config.target_col,
            lags=config.lags,
            rolling_means=config.rolling_means,
            extra_features=config.extra_features,
            forecast_horizon=config.optimization_horizon,
        )
        source_indices = np.flatnonzero(valid).astype(int)
        meta = {
            "source_index": source_indices,
            "horizon": np.full(len(source_indices), int(config.optimization_horizon), dtype=int),
            "horizon_col": horizon_col,
        }
        train_end = int(len(X) * config.train_ratio)
        horizon_mode = f"direct horizon {config.optimization_horizon}"

    meta["target_index"] = meta["source_index"] + meta["horizon"] - 1

    if train_end <= 0 or train_end >= len(X):
        raise ValueError("Train/test split produced an empty side; adjust the train ratio or horizon.")

    result = fit_symbolic(
        X,
        y,
        train_end,
        population_size=config.population_size,
        generations=config.generations,
        parsimony=config.parsimony,
        function_set=config.function_set,
        predict_residual=config.predict_residual,
        baseline_col="y_lag_1",
        train_mask=train_mask,
        horizon_col=horizon_col,
    )
    result["horizon_mode"] = horizon_mode
    result["optimization_horizon"] = config.optimization_horizon
    return {
        "res": result,
        "train_end": train_end,
        "X": X,
        "y": y,
        "config": config,
        "meta": meta,
        "split_idx": split_idx,
    }


def run_symbolic_multistep(cache: dict, horizon: int, n_starts: int) -> dict:
    return evaluate_multistep(
        cache["res"],
        cache["X"],
        cache["y"],
        cache["train_end"],
        horizon=horizon,
        n_starts=n_starts,
    )


def symbolic_mode_text(result: dict) -> str:
    return "delta residual" if result.get("predict_residual") else "direct value"