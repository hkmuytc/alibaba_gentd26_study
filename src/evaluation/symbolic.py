"""
Symbolic regression for short-term power forecasting.

Goal: discover a compact algebraic expression
    y[t] = f(y[t-1], y[t-2], ..., exogenous features)
that competes with deep models on short horizons. This addresses the
"unified algebraic expression" novelty angle.

Backend: gplearn (pure Python, no Julia dependency). PySR can be swapped
in by replacing the fit() call.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_lagged_design_matrix(df: pd.DataFrame, target_col: str,
                                lags=(1, 2, 3, 6, 12),
                                extra_features=None,
                                rolling_means=(6, 12),
                                forecast_horizon: int = 1):
    """Build a feature matrix from lags of the target and current exogenous
    features, suitable for symbolic regression."""
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be >= 1.")

    X_parts = {}
    for L in lags:
        X_parts[f"y_lag_{L}"] = df[target_col].shift(L).values
    for w in rolling_means:
        X_parts[f"y_rmean_{w}"] = df[target_col].shift(1).rolling(w, min_periods=1).mean().values
    if extra_features:
        for c in extra_features:
            if c in df.columns:
                X_parts[c] = df[c].values
    X = pd.DataFrame(X_parts)
    y = df[target_col].shift(-(forecast_horizon - 1)).values
    valid = ~X.isna().any(axis=1) & ~pd.isna(y)
    return X[valid].reset_index(drop=True), y[valid.values], np.array(valid)


def build_multi_horizon_design_matrix(df: pd.DataFrame, target_col: str,
                                      horizons=(1, 2, 3, 6, 12),
                                      lags=(1, 2, 3, 6, 12),
                                      extra_features=None,
                                      rolling_means=(6, 12),
                                      horizon_col: str = "horizon_steps"):
    """Build one symbolic-regression dataset across multiple future horizons.

    A single expression can learn y[t+h] from lagged history plus `h`, which is
    a stronger objective than repeatedly rewarding one-step persistence.
    """
    clean_horizons = tuple(sorted({int(h) for h in horizons if int(h) >= 1}))
    if not clean_horizons:
        raise ValueError("Provide at least one positive horizon.")

    base_X, _, valid = build_lagged_design_matrix(
        df,
        target_col=target_col,
        lags=lags,
        extra_features=extra_features,
        rolling_means=rolling_means,
        forecast_horizon=1,
    )
    base_indices = np.flatnonzero(valid)
    rows = []
    targets = []
    source_indices = []
    horizon_values = []

    for row_pos, source_idx in enumerate(base_indices):
        base_row = base_X.iloc[row_pos]
        for horizon in clean_horizons:
            target_idx = source_idx + horizon - 1
            if target_idx >= len(df):
                continue
            row = base_row.copy()
            row[horizon_col] = float(horizon)
            rows.append(row)
            targets.append(float(df[target_col].iloc[target_idx]))
            source_indices.append(int(source_idx))
            horizon_values.append(int(horizon))

    X = pd.DataFrame(rows).reset_index(drop=True)
    y = np.asarray(targets, dtype=np.float64)
    meta = {
        "source_index": np.asarray(source_indices, dtype=int),
        "horizon": np.asarray(horizon_values, dtype=int),
        "horizon_col": horizon_col,
    }
    return X, y, meta


def fit_symbolic(X: pd.DataFrame, y: np.ndarray, train_end: int,
                  population_size: int = 500, generations: int = 20,
                  parsimony: float = 0.001,
                  function_set=("add", "sub", "mul", "div", "sin", "log"),
                  random_state: int = 42,
                  verbose: int = 0,
                  predict_residual: bool = False,
                  baseline_col: str = "y_lag_1",
                  train_mask: np.ndarray | None = None,
                  horizon_col: str = "horizon_steps"):
    """Fit gplearn SymbolicRegressor on train and evaluate on test.

    If `predict_residual=True`, the target becomes Δy = y - X[baseline_col]
    (typically y[t] - y[t-1]). The final prediction is then
        ŷ = baseline + symbolic_expression(features)
    This forces the engine to find what explains *movement*, not just
    re-discover persistence.

    When other features are available, `baseline_col` is kept as the reported
    persistence baseline but removed from the symbolic search itself. That
    prevents direct mode from collapsing to the trivial identity `ŷ = y[t-1]`
    and keeps residual mode from reusing the same baseline twice.
    """
    try:
        from gplearn.genetic import SymbolicRegressor
    except ImportError as e:
        raise ImportError("gplearn required. pip install gplearn") from e

    y = np.asarray(y, dtype=np.float64)
    if train_mask is None:
        train_mask = np.arange(len(X)) < train_end
    else:
        train_mask = np.asarray(train_mask, dtype=bool)
    test_mask = ~train_mask
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Symbolic train/test split produced an empty side.")

    target = y.copy().astype(np.float64)
    baseline_train = baseline_test = None
    if predict_residual:
        if baseline_col not in X.columns:
            raise ValueError(
                f"predict_residual=True requires column '{baseline_col}' in X."
            )
        baseline_train = X.loc[train_mask, baseline_col].values.astype(np.float64)
        baseline_test = X.loc[test_mask, baseline_col].values.astype(np.float64)
        target = y - X[baseline_col].values

    search_cols = list(X.columns)
    if baseline_col in search_cols:
        non_baseline_cols = [col for col in search_cols if col != baseline_col]
        if predict_residual:
            if not non_baseline_cols:
                raise ValueError(
                    "Residual mode needs at least one symbolic feature beyond the baseline column."
                )
            search_cols = non_baseline_cols
        elif non_baseline_cols:
            search_cols = non_baseline_cols

    search_X = X.loc[:, search_cols]

    sr = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        function_set=list(function_set),
        parsimony_coefficient=parsimony,
        max_samples=0.9,
        verbose=verbose,
        random_state=random_state,
        n_jobs=1,
    )
    sr.fit(search_X.loc[train_mask].values, target[train_mask])

    raw_train = sr.predict(search_X.loc[train_mask].values)
    raw_test = sr.predict(search_X.loc[test_mask].values)

    if predict_residual:
        preds_train = baseline_train + raw_train
        preds_test = baseline_test + raw_test
    else:
        preds_train = raw_train
        preds_test = raw_test

    actuals_train = y[train_mask]
    actuals_test = y[test_mask]

    def _m(a, p):
        a = np.asarray(a); p = np.asarray(p)
        mask = a != 0
        return {
            "MAE": float(mean_absolute_error(a, p)),
            "RMSE": float(np.sqrt(mean_squared_error(a, p))),
            "MAPE": float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.any() else float("inf"),
            "R2": float(r2_score(a, p)) if len(a) > 1 else float("nan"),
        }

    persistence_train = persistence_test = None
    metrics_persistence_train = metrics_persistence_test = None
    if baseline_col in X.columns:
        persistence_train = X.loc[train_mask, baseline_col].values.astype(np.float64)
        persistence_test = X.loc[test_mask, baseline_col].values.astype(np.float64)
        metrics_persistence_train = _m(actuals_train, persistence_train)
        metrics_persistence_test = _m(actuals_test, persistence_test)

    metrics_by_horizon = None
    persistence_by_horizon = None
    if horizon_col in X.columns:
        metrics_by_horizon = {}
        persistence_by_horizon = {}
        test_horizons = X.loc[test_mask, horizon_col].values.astype(int)
        for horizon in sorted(np.unique(test_horizons)):
            h_mask = test_horizons == horizon
            metrics_by_horizon[int(horizon)] = _m(actuals_test[h_mask], preds_test[h_mask])
            if persistence_test is not None:
                persistence_by_horizon[int(horizon)] = _m(actuals_test[h_mask], persistence_test[h_mask])

    # Extract fitness curve and top-K expressions from final population
    run_details = getattr(sr, "run_details_", {})
    fitness_curve = {
        "generation": list(run_details.get("generation", [])),
        "best_fitness": list(run_details.get("best_fitness", [])),
        "average_fitness": list(run_details.get("average_fitness", [])),
        "best_length": list(run_details.get("best_length", [])),
    }

    top_expressions = []
    last_pop = sr._programs[-1] if getattr(sr, "_programs", None) else []
    seen = set()
    sorted_pop = sorted(
        [p for p in last_pop if p is not None],
        key=lambda p: (p.fitness_, p.length_),
    )
    for prog in sorted_pop:
        s = str(prog)
        if s in seen:
            continue
        seen.add(s)
        top_expressions.append({
            "expression": s,
            "complexity": int(prog.length_),
            "fitness": float(prog.fitness_),
        })
        if len(top_expressions) >= 10:
            break

    return {
        "model": sr,
        "expression": str(sr._program),
        "complexity": int(sr._program.length_),
        "preds_test": preds_test,
        "actuals_test": actuals_test,
        "preds_train": preds_train,
        "persistence_test": persistence_test,
        "persistence_train": persistence_train,
        "metrics_test": _m(actuals_test, preds_test),
        "metrics_train": _m(actuals_train, preds_train),
        "metrics_persistence_test": metrics_persistence_test,
        "metrics_persistence_train": metrics_persistence_train,
        "metrics_by_horizon": metrics_by_horizon,
        "persistence_by_horizon": persistence_by_horizon,
        "feature_names": list(search_X.columns),
        "predict_residual": predict_residual,
        "baseline_col": baseline_col if predict_residual else None,
        "fitness_curve": fitness_curve,
        "top_expressions": top_expressions,
    }


_SYMBOLIC_TOKEN_RE = re.compile(r"\bX(\d+)\b")
_UNARY_FUNCS = {"sin", "cos", "log", "sqrt", "abs", "neg"}
_BINARY_FUNCS = {"add", "sub", "mul", "div", "max", "min", "pow"}


def _humanize_feature_name(name: str, fmt: str) -> str:
    if name.startswith("y_lag_"):
        lag = name.split("_")[-1]
        return rf"y_{{t-{lag}}}" if fmt == "latex" else f"y[t-{lag}]"
    if name.startswith("y_rmean_"):
        window = name.split("_")[-1]
        return rf"\overline{{y}}_{{{window}}}" if fmt == "latex" else f"mean_{window}(y)"
    if name == "horizon_steps":
        return "h"
    return name.replace("_", r"\_") if fmt == "latex" else name


def _resolve_feature_name(token: str, feature_names) -> str | None:
    match = _SYMBOLIC_TOKEN_RE.fullmatch(token)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < len(feature_names):
            return feature_names[idx]
    if token in feature_names:
        return token
    return None


def _feature_token_name(token: str, feature_names, fmt: str) -> str:
    resolved = _resolve_feature_name(token, feature_names)
    if resolved is not None:
        return _humanize_feature_name(resolved, fmt)

    return token


def _is_number_token(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _split_call(expr: str) -> tuple[str, list[str]] | None:
    expr = expr.strip()
    first_paren = expr.find("(")
    if first_paren <= 0 or not expr.endswith(")"):
        return None

    func = expr[:first_paren].strip()
    body = expr[first_paren + 1:-1]
    args = []
    current = []
    depth = 0
    for ch in body:
        if ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        current.append(ch)
    if current:
        args.append("".join(current).strip())
    return func, args


def _format_number_text(token: str) -> str:
    try:
        value = float(token)
    except ValueError:
        return token
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


def _format_number_latex(token: str) -> str:
    return _format_number_text(token)


def _strip_outer_group(expr: str) -> str:
    expr = expr.strip()
    if len(expr) < 2 or expr[0] != "(" or expr[-1] != ")":
        return expr
    depth = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and i < len(expr) - 1:
            return expr
    return expr[1:-1]


def _render_symbolic(expr: str, feature_names, fmt: str = "text") -> str:
    expr = expr.strip()
    call = _split_call(expr)
    if call is None:
        token = _feature_token_name(expr, feature_names, fmt)
        if _is_number_token(token):
            return _format_number_latex(token) if fmt == "latex" else _format_number_text(token)
        return token

    func, args = call
    rendered_args = [_render_symbolic(arg, feature_names, fmt=fmt) for arg in args]

    if func in _BINARY_FUNCS and len(rendered_args) == 2:
        left, right = rendered_args
        if func == "add":
            return f"({left} + {right})"
        if func == "sub":
            return f"({left} - {right})"
        if func == "mul":
            if fmt == "latex":
                return f"({left} \\cdot {right})"
            return f"({left} * {right})"
        if func == "div":
            if fmt == "latex":
                return rf"\frac{{{left}}}{{{right}}}"
            return f"({left} / {right})"
        if func == "pow":
            if fmt == "latex":
                return rf"({left})^{{{right}}}"
            return f"({left} ^ {right})"
        if func == "max":
            return rf"\max\left({left}, {right}\right)" if fmt == "latex" else f"max({left}, {right})"
        if func == "min":
            return rf"\min\left({left}, {right}\right)" if fmt == "latex" else f"min({left}, {right})"

    if func in _UNARY_FUNCS and len(rendered_args) == 1:
        arg = rendered_args[0]
        if func == "neg":
            return f"(-{arg})"
        if func == "abs":
            return rf"\left|{arg}\right|" if fmt == "latex" else f"abs({arg})"
        if func == "sqrt":
            return rf"\sqrt{{{arg}}}" if fmt == "latex" else f"sqrt({arg})"
        if func == "log":
            return rf"\log\left({arg}\right)" if fmt == "latex" else f"log({arg})"
        if func == "sin":
            return rf"\sin\left({arg}\right)" if fmt == "latex" else f"sin({arg})"
        if func == "cos":
            return rf"\cos\left({arg}\right)" if fmt == "latex" else f"cos({arg})"

    # Fallback for unexpected operators: still replace feature tokens and preserve structure.
    if fmt == "latex":
        return rf"\mathrm{{{func}}}\left({', '.join(rendered_args)}\right)"
    return f"{func}({', '.join(rendered_args)})"


def pretty_expression(expr: str, feature_names) -> str:
    """Render gplearn prefix expressions as readable infix math."""
    try:
        return _strip_outer_group(_render_symbolic(expr, feature_names, fmt="text"))
    except Exception:
        out = expr
        for i in reversed(range(len(feature_names))):
            out = out.replace(f"X{i}", feature_names[i])
        return out


def latex_expression(expr: str, feature_names) -> str:
    """Render gplearn prefix expressions as LaTeX for Streamlit/KaTeX display."""
    return _strip_outer_group(_render_symbolic(expr, feature_names, fmt="latex"))


def expression_feature_names(expr: str, feature_names) -> list[str]:
    """Return the ordered feature names referenced by an expression."""
    seen = []

    def _walk(part: str) -> None:
        part = part.strip()
        call = _split_call(part)
        if call is None:
            resolved = _resolve_feature_name(part, feature_names)
            if resolved is not None and resolved not in seen:
                seen.append(resolved)
            return
        _func, args = call
        for arg in args:
            _walk(arg)

    _walk(expr)
    return seen


def raw_expression_mapping(expr: str, feature_names) -> list[dict[str, str]]:
    """Map raw gplearn placeholders like X11 to their actual feature names."""
    rows = []
    seen = set()
    for match in _SYMBOLIC_TOKEN_RE.finditer(expr):
        token = match.group(0)
        if token in seen:
            continue
        seen.add(token)
        idx = int(match.group(1))
        if 0 <= idx < len(feature_names):
            feature_name = feature_names[idx]
            rows.append({
                "Placeholder": token,
                "Dataset feature": feature_name,
                "Readable symbol": _humanize_feature_name(feature_name, "text"),
            })
    return rows


def describe_symbolic_feature(name: str, step_minutes: int = 5) -> dict[str, str]:
    """Plain-English description for formula variables shown in the UI."""
    symbol = _humanize_feature_name(name, "text")

    if name.startswith("y_lag_"):
        lag = int(name.split("_")[-1])
        minutes = lag * step_minutes
        return {
            "Symbol": symbol,
            "Meaning": f"Cluster power {lag} step(s) earlier ({minutes} minutes earlier).",
            "Source": "Past target value",
        }
    if name.startswith("y_rmean_"):
        window = int(name.split("_")[-1])
        minutes = window * step_minutes
        return {
            "Symbol": symbol,
            "Meaning": f"Average cluster power over the previous {window} step(s) ({minutes} minutes).",
            "Source": "Engineered target history",
        }
    if name == "hour_sin":
        return {
            "Symbol": symbol,
            "Meaning": "Time-of-day signal derived from the timestamp using sine.",
            "Source": "Engineered time feature",
        }
    if name == "hour_cos":
        return {
            "Symbol": symbol,
            "Meaning": "Time-of-day signal derived from the timestamp using cosine.",
            "Source": "Engineered time feature",
        }
    if name == "dow_sin":
        return {
            "Symbol": symbol,
            "Meaning": "Day-of-week signal derived from the timestamp using sine.",
            "Source": "Engineered time feature",
        }
    if name == "dow_cos":
        return {
            "Symbol": symbol,
            "Meaning": "Day-of-week signal derived from the timestamp using cosine.",
            "Source": "Engineered time feature",
        }
    if name == "horizon_steps":
        return {
            "Symbol": "h",
            "Meaning": "How many steps ahead into the future the formula is predicting.",
            "Source": "Forecast horizon index",
        }

    common_raw = {
        "gpu_util": "GPU utilization level.",
        "qps": "Query / request rate.",
        "mem_util": "Memory utilization level.",
        "gpu_mem_bytes": "GPU memory usage in bytes.",
        "machine_cpu": "Machine CPU utilization.",
        "machine_gpu": "Machine GPU utilization.",
    }
    if name in common_raw:
        return {
            "Symbol": symbol,
            "Meaning": common_raw[name],
            "Source": "Dataset feature",
        }

    return {
        "Symbol": symbol,
        "Meaning": f"Input feature `{name}` used by the symbolic model.",
        "Source": "Dataset / engineered feature",
    }


def _parse_feature_meta(feature_names):
    """Inspect feature names to figure out lag indices, rolling-mean windows,
    and which columns are exogenous (passed straight through)."""
    lag_cols = {}        # name -> int lag
    rmean_cols = {}      # name -> int window
    exo_cols = []        # exogenous (read from future)
    for name in feature_names:
        if name.startswith("y_lag_"):
            try:
                lag_cols[name] = int(name.split("_")[-1])
            except ValueError:
                exo_cols.append(name)
        elif name.startswith("y_rmean_"):
            try:
                rmean_cols[name] = int(name.split("_")[-1])
            except ValueError:
                exo_cols.append(name)
        else:
            exo_cols.append(name)
    return lag_cols, rmean_cols, exo_cols


def _is_known_future_feature(name: str) -> bool:
    """Return True for deterministic future features that are safe to advance.

    Calendar-derived signals and the explicit horizon index are known once the
    forecast timestamp is chosen, so using their future values does not leak the
    hidden test target or hidden future workload measurements.
    """
    if name == "horizon_steps":
        return True
    if name == "hour" or name.startswith("hour_"):
        return True
    if name.startswith("dow_"):
        return True
    return False


def replay_forecast_from_frame(
    sr_result: dict,
    df: pd.DataFrame,
    start_idx: int,
    horizon: int,
    target_col: str = "power_total_kw",
) -> dict:
    """Roll the learned symbolic formula forward on the original time series.

    This is the symbolic analogue of forecast replay: start from one cutoff in
    the original dataframe, keep unknown future exogenous inputs frozen at the
    last known value, let deterministic calendar features advance, and update
    lag/rolling target features with the model's own predictions.
    """
    sr = sr_result["model"]
    feature_names = sr_result["feature_names"]
    predict_residual = sr_result.get("predict_residual", False)

    if target_col not in df.columns:
        raise ValueError(f"Dataframe must include target column '{target_col}'.")
    if start_idx < 0 or start_idx >= len(df):
        raise ValueError("start_idx is outside the dataframe bounds.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    lag_cols, rmean_cols, exo_cols = _parse_feature_meta(feature_names)
    max_lookback = max(list(lag_cols.values()) + list(rmean_cols.values()) + [1])

    target_values = df[target_col].to_numpy(dtype=np.float64)
    buffer = list(target_values[max(0, start_idx - max_lookback):start_idx])
    if not buffer:
        raise ValueError("Not enough history before the cutoff for symbolic replay.")

    seed_row_idx = max(0, start_idx - 1)
    carried_exogenous = {
        name: float(df[name].iloc[seed_row_idx])
        for name in exo_cols
        if name in df.columns and not _is_known_future_feature(name)
    }

    preds = []
    actuals = []
    persistence = []
    target_indices = []
    last_known = float(target_values[seed_row_idx])

    for h in range(horizon):
        row_idx = start_idx + h
        if row_idx >= len(df):
            break

        feats = np.empty(len(feature_names), dtype=np.float64)
        for j, name in enumerate(feature_names):
            if name in lag_cols:
                lag = lag_cols[name]
                if len(buffer) >= lag:
                    feats[j] = buffer[-lag]
                else:
                    feats[j] = last_known
            elif name in rmean_cols:
                window = rmean_cols[name]
                tail = buffer[-window:] if buffer else [last_known]
                feats[j] = float(np.mean(tail))
            elif name == "horizon_steps":
                feats[j] = float(h + 1)
            elif _is_known_future_feature(name) and name in df.columns:
                feats[j] = float(df[name].iloc[row_idx])
            else:
                feats[j] = carried_exogenous.get(
                    name,
                    float(df[name].iloc[seed_row_idx]) if name in df.columns else 0.0,
                )

        raw_pred = float(sr.predict(feats.reshape(1, -1))[0])
        baseline_val = buffer[-1]
        yhat = baseline_val + raw_pred if predict_residual else raw_pred

        preds.append(yhat)
        actuals.append(float(target_values[row_idx]))
        persistence.append(last_known)
        target_indices.append(int(row_idx))
        buffer.append(yhat)
        if len(buffer) > max_lookback + 5:
            buffer = buffer[-(max_lookback + 5):]

    return {
        "preds": np.asarray(preds, dtype=np.float64),
        "actuals": np.asarray(actuals, dtype=np.float64),
        "persistence": np.asarray(persistence, dtype=np.float64),
        "target_indices": np.asarray(target_indices, dtype=int),
    }


def recursive_forecast(sr_result: dict, X: pd.DataFrame, y: np.ndarray,
                        start_idx: int, horizon: int) -> np.ndarray:
    """Roll the symbolic model forward `horizon` steps starting at `start_idx`.

    At each step:
      - y_lag_L is fed from the prediction made L steps ago (or, if L > current
        rollout depth, from the historical actuals).
      - y_rmean_W is recomputed from a sliding buffer of the last W values
        (mixing actuals and predictions as the rollout proceeds).
            - Deterministic calendar features (for example hour/day encodings) are
                allowed to advance with the forecast timestamp.
            - Other exogenous inputs are carried forward from the last known row so
                the rollout does not peek at hidden future workload values.
    """
    sr = sr_result["model"]
    feature_names = sr_result["feature_names"]
    predict_residual = sr_result.get("predict_residual", False)
    baseline_col = sr_result.get("baseline_col") or "y_lag_1"

    lag_cols, rmean_cols, exo_cols = _parse_feature_meta(feature_names)

    # Buffer of the last max_lag + max_window y-values (actuals + own preds)
    max_lookback = max(
        list(lag_cols.values()) + list(rmean_cols.values()) + [1]
    )
    # Seed the buffer with actuals up to (but not including) start_idx.
    # The X DataFrame already aligns row i with target y[i], so we need the
    # source y array. We approximate using X[y_lag_1].shift if needed; but
    # the safest is to rebuild from y itself.
    buffer = list(y[max(0, start_idx - max_lookback):start_idx])
    seed_row_idx = max(0, start_idx - 1)
    carried_exogenous = {
        name: float(X[name].iloc[seed_row_idx])
        for name in exo_cols
        if name in X.columns and not _is_known_future_feature(name)
    }

    preds = []
    for h in range(horizon):
        row_idx = start_idx + h
        if row_idx >= len(X):
            break
        feats = np.empty(len(feature_names), dtype=np.float64)
        for j, name in enumerate(feature_names):
            if name in lag_cols:
                L = lag_cols[name]
                if len(buffer) >= L:
                    feats[j] = buffer[-L]
                else:
                    feats[j] = X[name].iloc[row_idx]
            elif name in rmean_cols:
                W = rmean_cols[name]
                # rolling mean of last W values (target shifted by 1 = exclude current)
                tail = buffer[-W:] if len(buffer) >= 1 else [0.0]
                feats[j] = float(np.mean(tail))
            else:
                if name == "horizon_steps":
                    feats[j] = float(h + 1)
                elif _is_known_future_feature(name) and name in X.columns:
                    feats[j] = float(X[name].iloc[row_idx])
                else:
                    feats[j] = carried_exogenous.get(
                        name,
                        float(X[name].iloc[seed_row_idx]),
                    )

        raw = float(sr.predict(feats.reshape(1, -1))[0])
        if predict_residual:
            # baseline is y[t-1] -- that's the last value in our buffer
            baseline_val = buffer[-1] if buffer else 0.0
            yhat = baseline_val + raw
        else:
            yhat = raw

        preds.append(yhat)
        buffer.append(yhat)
        if len(buffer) > max_lookback + 5:
            buffer = buffer[-(max_lookback + 5):]

    return np.array(preds)


def evaluate_multistep(sr_result: dict, X: pd.DataFrame, y: np.ndarray,
                        train_end: int, horizon: int,
                        n_starts: int = 30) -> dict:
    """Run recursive_forecast from many starting points within the test region
    and aggregate per-horizon error.

    Returns:
        {
          "horizon": [1..H],
          "mae_per_h": np.ndarray (H,),
          "rmse_per_h": np.ndarray (H,),
          "persistence_mae_per_h": np.ndarray (H,),  # baseline: flat-line forecast
          "sample_rollout": {"start": idx, "preds": ..., "actuals": ...,
                              "persistence": ...},
        }
    """
    test_len = len(y) - train_end
    if test_len < horizon + 5:
        # Not enough room — shrink horizon
        horizon = max(1, test_len - 2)

    valid_starts = list(range(train_end, len(y) - horizon))
    if len(valid_starts) > n_starts:
        idx_sel = np.linspace(0, len(valid_starts) - 1, n_starts).astype(int)
        starts = [valid_starts[i] for i in idx_sel]
    else:
        starts = valid_starts

    sym_errs = np.zeros((len(starts), horizon))
    per_errs = np.zeros((len(starts), horizon))
    sym_sq = np.zeros((len(starts), horizon))
    per_sq = np.zeros((len(starts), horizon))

    for i, s in enumerate(starts):
        actuals = y[s:s + horizon]
        sym_p = recursive_forecast(sr_result, X, y, s, horizon)
        # Pad if shorter
        if len(sym_p) < horizon:
            sym_p = np.concatenate([sym_p, np.full(horizon - len(sym_p), np.nan)])
        # Persistence: predict y[s-1] for all h steps (flat line)
        last_known = y[s - 1] if s > 0 else y[s]
        per_p = np.full(horizon, last_known)

        sym_errs[i] = np.abs(sym_p - actuals)
        per_errs[i] = np.abs(per_p - actuals)
        sym_sq[i] = (sym_p - actuals) ** 2
        per_sq[i] = (per_p - actuals) ** 2

    sample_start = starts[len(starts) // 2]
    sample_pred = recursive_forecast(sr_result, X, y, sample_start, horizon)
    sample_actual = y[sample_start:sample_start + horizon]
    sample_pers = np.full(len(sample_actual), y[sample_start - 1])

    return {
        "horizon": list(range(1, horizon + 1)),
        "mae_per_h": np.nanmean(sym_errs, axis=0),
        "rmse_per_h": np.sqrt(np.nanmean(sym_sq, axis=0)),
        "persistence_mae_per_h": np.nanmean(per_errs, axis=0),
        "persistence_rmse_per_h": np.sqrt(np.nanmean(per_sq, axis=0)),
        "n_starts": len(starts),
        "rollout_exogenous_policy": (
            "No-peek rollout: unknown future exogenous inputs are carried "
            "forward from the last known step; deterministic calendar "
            "features and horizon_steps are allowed to advance."
        ),
        "sample_rollout": {
            "start": int(sample_start),
            "preds": sample_pred,
            "actuals": sample_actual,
            "persistence": sample_pers,
        },
    }
