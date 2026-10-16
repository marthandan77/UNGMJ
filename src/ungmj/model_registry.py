"""Independent model registry for shadow UNG forecasts.

Optional third-party boosters are loaded only when installed; unavailable models are
reported as skipped rather than silently substituted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import DataValidationError
from .model import ModelForecast, _empirical_probability


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    label: str
    description: str
    factory: Callable[[], object]
    requires: str | None = None


def _pipe(estimator: object, *, scale: bool = False) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("estimator", estimator))
    return Pipeline(steps)


def _lightgbm():
    from lightgbm import LGBMRegressor
    return _pipe(LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=15,
                              max_depth=-1, random_state=42, verbosity=-1))


def _xgboost():
    from xgboost import XGBRegressor
    return _pipe(XGBRegressor(n_estimators=250, max_depth=3, learning_rate=0.03,
                              subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror",
                              random_state=42, n_jobs=1, verbosity=0))


def _regime_aware():
    # Transparent regime-aware challenger: price/volatility regime features are explicit,
    # and the estimator remains a separate model rather than changing the Ridge baseline.
    return _pipe(HistGradientBoostingRegressor(max_iter=180, learning_rate=0.04,
                                                max_leaf_nodes=15, l2_regularization=2.0,
                                                random_state=42))


MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("ridge", "Ridge Regression", "Existing regularized linear benchmark.",
              lambda: _pipe(Ridge(alpha=10.0), scale=True)),
    ModelSpec("elastic_net", "Elastic Net", "Linear model with combined L1/L2 regularization.",
              lambda: _pipe(ElasticNet(alpha=0.001, l1_ratio=0.25, max_iter=10000), scale=True)),
    ModelSpec("random_forest", "Random Forest", "Bagged trees for nonlinear relationships.",
              lambda: _pipe(RandomForestRegressor(n_estimators=250, min_samples_leaf=5,
                                                  max_features=0.8, random_state=42, n_jobs=1))),
    ModelSpec("extra_trees", "Extra Trees", "Randomized tree ensemble challenger.",
              lambda: _pipe(ExtraTreesRegressor(n_estimators=250, min_samples_leaf=5,
                                                max_features=0.8, random_state=42, n_jobs=1))),
    ModelSpec("hist_gradient_boosting", "Histogram Gradient Boosting",
              "Native scikit-learn gradient boosting.", _regime_aware),
    ModelSpec("lightgbm", "LightGBM", "Efficient gradient-boosted trees; optional dependency.",
              _lightgbm, "lightgbm"),
    ModelSpec("xgboost", "XGBoost", "Gradient-boosted trees; optional dependency.",
              _xgboost, "xgboost"),
    ModelSpec("sgd_online", "SGD Online-Learning Challenger",
              "Incremental-capable linear model retrained from eligible history each run.",
              lambda: _pipe(SGDRegressor(loss="squared_error", penalty="elasticnet", alpha=0.001,
                                         max_iter=3000, tol=1e-4, random_state=42), scale=True)),
)


def fit_model_forecast(
    model_key: str, features: pd.DataFrame, targets: pd.DataFrame, reference_close: float,
    *, min_train_rows: int = 100, calibration_residuals: np.ndarray | None = None,
    calibration_low_errors: np.ndarray | None = None,
    calibration_high_errors: np.ndarray | None = None,
) -> ModelForecast:
    spec = next((item for item in MODEL_REGISTRY if item.key == model_key), None)
    if spec is None:
        raise DataValidationError(f"Unknown model key: {model_key}")
    aligned = features.join(targets, how="inner").dropna(subset=["forward_return"])
    label_cols = ["forward_return", "forward_high_return", "forward_low_return"]
    aligned = aligned.dropna(subset=label_cols)
    if len(aligned) < min_train_rows:
        raise DataValidationError(f"Need at least {min_train_rows} eligible rows; have {len(aligned)}")
    x = aligned[list(features.columns)]
    y = aligned["forward_return"].to_numpy(dtype=float)
    model = spec.factory()
    model.fit(x, y)
    predicted = float(model.predict(features.iloc[[-1]])[0])
    fitted = np.asarray(model.predict(x), dtype=float)
    residuals = y - fitted
    calibration = np.asarray(calibration_residuals, dtype=float) if calibration_residuals is not None else np.array([])
    calibration = calibration[np.isfinite(calibration)]
    source = calibration if len(calibration) >= 50 else residuals
    lower, upper = np.quantile(source, [0.1, 0.9])
    probability = (float(np.clip(np.mean(predicted + source > 0), 0.01, 0.99))
                   if len(source) >= 20 else _empirical_probability(predicted, source))

    # Predict the actual future low and high directly; do not substitute unconditional
    # historical quantiles for a model-specific five-session range.
    low_model, high_model = spec.factory(), spec.factory()
    low_model.fit(x, aligned["forward_low_return"].to_numpy(dtype=float))
    high_model.fit(x, aligned["forward_high_return"].to_numpy(dtype=float))
    low_return = float(low_model.predict(features.iloc[[-1]])[0])
    high_return = float(high_model.predict(features.iloc[[-1]])[0])
    low_errors = np.asarray(calibration_low_errors, dtype=float) if calibration_low_errors is not None else np.array([])
    high_errors = np.asarray(calibration_high_errors, dtype=float) if calibration_high_errors is not None else np.array([])
    low_errors, high_errors = low_errors[np.isfinite(low_errors)], high_errors[np.isfinite(high_errors)]
    range_n = min(len(low_errors), len(high_errors))
    if range_n >= 30:
        # Errors are actual minus forecast returns from previously matured, unseen vintages.
        low_return += float(np.quantile(low_errors, 0.10))
        high_return += float(np.quantile(high_errors, 0.90))
        range_method = "matured_out_of_sample_range_error_quantiles"
    else:
        range_method = "direct_low_high_models_uncalibrated"
    low_price = max(0.01, reference_close * (1 + low_return))
    high_price = max(0.02, reference_close * (1 + high_return))
    # Keep the interval structurally valid even when the two independently fitted heads cross.
    predicted_low, predicted_high = min(low_price, high_price), max(low_price, high_price)
    return ModelForecast(
        probability_up=probability, expected_return=predicted,
        expected_close=reference_close * (1 + predicted),
        predicted_low=predicted_low, predicted_high=predicted_high,
        lower_return_quantile=float(lower), upper_return_quantile=float(upper),
        model_name=model_key, model_version="0.2.0",
        calibration_sample_count=range_n,
        calibration_method=range_method,
    )
