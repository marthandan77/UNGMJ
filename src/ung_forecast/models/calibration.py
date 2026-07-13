"""Probability calibration fitted only on held-out validation predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ung_forecast.validation.metrics import CLASS_ORDER


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    method: str = "platt"

    def __post_init__(self) -> None:
        if self.method not in {"platt", "isotonic", "identity"}:
            raise ValueError("method must be 'platt', 'isotonic', or 'identity'")


class MulticlassProbabilityCalibrator:
    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self._platt_models: dict[str, LogisticRegression] = {}
        self._isotonic_models: dict[str, IsotonicRegression] = {}
        self._fitted = False

    @staticmethod
    def _validate_probabilities(raw_probabilities: pd.DataFrame) -> None:
        if tuple(raw_probabilities.columns) != CLASS_ORDER:
            raise ValueError("Calibration probabilities must use the canonical class order")
        if raw_probabilities.empty:
            raise ValueError("Calibration probabilities cannot be empty")
        if raw_probabilities.isna().any().any():
            raise ValueError("Calibration probabilities contain missing values")
        if (raw_probabilities < 0.0).any().any():
            raise ValueError("Calibration probabilities cannot be negative")
        if (raw_probabilities.sum(axis=1) <= 0.0).any():
            raise ValueError("Calibration probability rows must have positive mass")

    def fit(self, raw_probabilities: pd.DataFrame, target: pd.Series) -> None:
        if not raw_probabilities.index.equals(target.index):
            raise ValueError("Calibration probabilities and target must align")
        self._validate_probabilities(raw_probabilities)
        if self.config.method == "identity":
            self._fitted = True
            return
        for class_name in CLASS_ORDER:
            binary_target = target.astype(str).eq(class_name).astype(int)
            scores = raw_probabilities[class_name].astype(float).to_numpy()
            if binary_target.nunique() < 2:
                raise ValueError(f"Calibration target lacks both outcomes for {class_name}")
            if self.config.method == "platt":
                model = LogisticRegression(solver="lbfgs")
                model.fit(scores.reshape(-1, 1), binary_target.to_numpy())
                self._platt_models[class_name] = model
            else:
                model = IsotonicRegression(out_of_bounds="clip")
                model.fit(scores, binary_target.to_numpy())
                self._isotonic_models[class_name] = model
        self._fitted = True

    def transform(self, raw_probabilities: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fitted")
        self._validate_probabilities(raw_probabilities)
        if self.config.method == "identity":
            row_sum = raw_probabilities.sum(axis=1)
            return raw_probabilities.astype(float).div(row_sum, axis=0)
        calibrated = pd.DataFrame(index=raw_probabilities.index)
        for class_name in CLASS_ORDER:
            scores = raw_probabilities[class_name].astype(float).to_numpy()
            if self.config.method == "platt":
                model = self._platt_models[class_name]
                values = model.predict_proba(scores.reshape(-1, 1))[:, 1]
            else:
                model = self._isotonic_models[class_name]
                values = model.predict(scores)
            calibrated[class_name] = np.asarray(values, dtype=float)
        row_sum = calibrated.sum(axis=1).replace(0.0, np.nan)
        calibrated = calibrated.div(row_sum, axis=0)
        if calibrated.isna().any().any():
            raise ValueError("Calibration produced invalid probability rows")
        return calibrated
