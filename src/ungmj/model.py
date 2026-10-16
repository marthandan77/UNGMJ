"""Conservative Ridge baseline and expanding-window walk-forward evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import DataValidationError


@dataclass(frozen=True, slots=True)
class ModelForecast:
    probability_up: float
    expected_return: float
    expected_close: float
    predicted_low: float
    predicted_high: float
    lower_return_quantile: float
    upper_return_quantile: float
    model_name: str = "ridge"
    model_version: str = "0.1.0"
    calibration_sample_count: int = 0
    calibration_method: str = "in_sample_fallback"


def _pipeline(alpha: float = 10.0) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def _empirical_probability(predicted_return: float, residuals: np.ndarray) -> float:
    if len(residuals) < 20:
        return float(np.clip(0.5 + predicted_return * 5.0, 0.05, 0.95))
    adjusted = residuals + predicted_return
    return float(np.clip(np.mean(adjusted > 0), 0.01, 0.99))


def fit_and_forecast(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    reference_close: float,
    *,
    min_train_rows: int = 100,
    alpha: float = 10.0,
    calibration_residuals: np.ndarray | None = None,
) -> ModelForecast:
    """Fit a Ridge point forecast; range quantiles use training residuals (not yet calibrated)."""
    if reference_close <= 0 or not np.isfinite(reference_close):
        raise DataValidationError("reference_close must be positive and finite")
    aligned = features.join(targets, how="inner").dropna(subset=["forward_return"])
    label_cols = ["forward_return", "forward_high_return", "forward_low_return"]
    feature_cols = list(features.columns)
    aligned = aligned.dropna(subset=label_cols)
    if len(aligned) < min_train_rows:
        raise DataValidationError(f"Need at least {min_train_rows} eligible training rows; have {len(aligned)}")
    x = aligned[feature_cols]
    y = aligned["forward_return"].to_numpy(dtype=float)
    pipeline = _pipeline(alpha)
    pipeline.fit(x, y)
    predicted = float(pipeline.predict(features.iloc[[-1]])[0])
    fitted = pipeline.predict(x)
    residuals = y - fitted
    # Prefer strictly out-of-sample residuals. In-sample residuals are fallback only and
    # must never be described as calibrated.
    calibration = np.asarray(calibration_residuals, dtype=float) if calibration_residuals is not None else np.array([])
    calibration = calibration[np.isfinite(calibration)]
    if len(calibration) >= 50:
        lower, upper = np.quantile(calibration, [0.1, 0.9])
        probability = float(np.clip(np.mean(predicted + calibration > 0), 0.01, 0.99))
    else:
        lower, upper = np.quantile(residuals, [0.1, 0.9])
        probability = _empirical_probability(predicted, residuals)
    low_return = float(np.quantile(aligned["forward_low_return"], 0.1))
    high_return = float(np.quantile(aligned["forward_high_return"], 0.9))
    return ModelForecast(
        probability_up=probability,
        expected_return=predicted,
        expected_close=reference_close * (1 + predicted),
        predicted_low=max(0.01, reference_close * (1 + low_return)),
        predicted_high=max(0.02, reference_close * (1 + high_return)),
        lower_return_quantile=float(lower),
        upper_return_quantile=float(upper),
        calibration_sample_count=len(calibration),
        calibration_method="purged_walk_forward_residuals" if len(calibration) >= 50 else "in_sample_fallback",
    )


def walk_forward_predictions(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    min_train_rows: int = 250,
    step: int = 5,
    horizon_sessions: int = 1,
    alpha: float = 10.0,
) -> pd.DataFrame:
    """Chronological expanding-window evaluation; each prediction uses only earlier rows."""
    if step < 1 or horizon_sessions < 1:
        raise ValueError("step and horizon_sessions must be positive")
    joined = features.join(targets, how="inner")
    label_cols = ["forward_return", "forward_high_return", "forward_low_return"]
    feature_cols = list(features.columns)
    rows: list[dict[str, object]] = []
    for i in range(min_train_rows, len(joined), step):
        # Purge training labels whose forward horizon overlaps the test period.
        train = joined.iloc[:max(0, i - horizon_sessions)].dropna(subset=label_cols)
        test = joined.iloc[i:i + step]
        if test.empty:
            break
        train = train.dropna(subset=label_cols)
        if len(train) < min_train_rows:
            continue
        model = _pipeline(alpha)
        model.fit(train[feature_cols], train["forward_return"].to_numpy(dtype=float))
        for timestamp, row in test.iterrows():
            if row[feature_cols].isna().all() or pd.isna(row["forward_return"]):
                continue
            prediction = float(model.predict(features.loc[[timestamp], feature_cols])[0])
            rows.append({
                "session": timestamp,
                "predicted_return": prediction,
                "actual_return": float(row["forward_return"]),
                "predicted_up": prediction > 0,
                "naive_predicted_up": float(features.loc[timestamp].get("return_1", 0.0)) > 0,
                "actual_up": float(row["forward_return"]) > 0,
                "absolute_error": abs(prediction - float(row["forward_return"])),
            })
    return pd.DataFrame(rows).set_index("session") if rows else pd.DataFrame(
        columns=["predicted_return", "actual_return", "predicted_up", "actual_up", "absolute_error"]
    )


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, float | int | None]:
    if predictions.empty:
        return {"n": 0, "directional_accuracy": None, "mae": None, "naive_directional_accuracy": None}
    return {
        "n": int(len(predictions)),
        "directional_accuracy": float((predictions["predicted_up"] == predictions["actual_up"]).mean()),
        "mae": float(predictions["absolute_error"].mean()),
        "naive_directional_accuracy": float(
            (predictions["naive_predicted_up"] == predictions["actual_up"]).mean()
        ) if "naive_predicted_up" in predictions else None,
    }
