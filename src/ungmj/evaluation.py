"""Chronological model and feature ablation evaluation; results never auto-promote a model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import DataValidationError

FEATURE_GROUPS: dict[str, tuple[str, ...] | None] = {
    "all": None,
    "price_momentum": ("return_1", "momentum_2", "momentum_3", "momentum_5", "momentum_10", "momentum_20", "momentum_60", "overnight_gap"),
    "volatility_range": ("true_range_pct", "realized_vol_5", "realized_vol_20", "range_expansion", "close_location"),
    "volume": ("volume_surprise_20",),
    "trend": ("trend_distance_10", "trend_distance_20", "trend_distance_60"),
    "henry_hub": ("henry_hub_spot", "henry_hub_return_1"),
}


def _model(alpha: float = 10.0) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def evaluate_strategies_and_features(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    horizon_sessions: int,
    min_train_rows: int = 250,
    step: int = 20,
    rolling_window: int = 756,
    half_life: int = 756,
    alpha: float = 10.0,
) -> pd.DataFrame:
    """Compare expanding, rolling, recency-weighted and feature ablations out of sample.

    Test folds are chronological; training labels are purged so their forward outcome window
    ends before each test timestamp. Results are evidence for review, never automatic promotion.
    """
    if horizon_sessions not in (1, 2, 3, 5) or step < 1:
        raise ValueError("Unsupported horizon or step")
    joined = features.join(targets, how="inner")
    labels = ["forward_return", "forward_high_return", "forward_low_return"]
    feature_columns = list(features.columns)
    groups = {
        name: list(columns) if columns is not None else feature_columns
        for name, columns in FEATURE_GROUPS.items()
    }
    groups = {name: [col for col in cols if col in feature_columns] for name, cols in groups.items()}
    rows: list[dict[str, object]] = []
    for test_i in range(min_train_rows + horizon_sessions, len(joined), step):
        train = joined.iloc[:test_i - horizon_sessions].dropna(subset=labels)
        test = joined.iloc[test_i]
        if len(train) < min_train_rows or pd.isna(test["forward_return"]):
            continue
        test_x = features.loc[[joined.index[test_i]]]
        for strategy in ("expanding", "rolling", "recency_weighted"):
            strategy_train = train.tail(rolling_window) if strategy == "rolling" else train
            weights = None
            if strategy == "recency_weighted":
                age = np.arange(len(strategy_train) - 1, -1, -1, dtype=float)
                weights = np.exp(np.log(0.5) * age / max(1, half_life))
            for group, cols in groups.items():
                if not cols:
                    continue
                model = _model(alpha)
                fit_kwargs = {"ridge__sample_weight": weights} if weights is not None else {}
                model.fit(strategy_train[cols], strategy_train["forward_return"].to_numpy(dtype=float), **fit_kwargs)
                prediction = float(model.predict(test_x[cols])[0])
                actual = float(test["forward_return"])
                rows.append({
                    "session": joined.index[test_i].isoformat(),
                    "horizon_sessions": horizon_sessions,
                    "strategy": strategy,
                    "feature_group": group,
                    "predicted_return": prediction,
                    "actual_return": actual,
                    "predicted_up": prediction > 0,
                    "actual_up": actual > 0,
                    "absolute_error": abs(prediction - actual),
                })
    if not rows:
        raise DataValidationError(
            f"Insufficient history for evaluation: need at least {min_train_rows + horizon_sessions + 1} rows"
        )
    return pd.DataFrame(rows)


def summarize_evaluation(results: pd.DataFrame) -> list[dict[str, object]]:
    if results.empty:
        return []
    summaries: list[dict[str, object]] = []
    grouping = ["strategy", "feature_group"]
    if "horizon_sessions" in results.columns:
        grouping = ["horizon_sessions", *grouping]
    for keys, frame in results.groupby(grouping, sort=True):
        if len(grouping) == 3:
            horizon, strategy, group = keys
        else:
            strategy, group = keys
            horizon = None
        summaries.append({
            "horizon_sessions": int(horizon) if horizon is not None else None,
            "strategy": str(strategy),
            "feature_group": str(group),
            "n": int(len(frame)),
            "directional_accuracy": float((frame["predicted_up"] == frame["actual_up"]).mean()),
            "mae": float(frame["absolute_error"].mean()),
            "mean_predicted_return": float(frame["predicted_return"].mean()),
            "mean_actual_return": float(frame["actual_return"].mean()),
        })
    return summaries
