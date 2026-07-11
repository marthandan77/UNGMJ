"""Probability calibration fitted only on held-out validation predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ung_forecast.schemas import OutcomeClass
from ung_forecast.validation.metrics import CLASS_ORDER


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    method: str = "platt"

    def __post_init__(self) -> None:
        if self.method not in {"platt", "isotonic"}:
            raise ValueError("method must be 'platt' or 'isotonic'")


class MulticlassProbabilityCalibrator:
    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self._models: dict[str, object] = {}
        self._fitted = False

    def fit(self, raw_probabilities: pd.DataFrame, target: pd.Series) -> None:
        if not raw_probabilities.index.equals(target.index):
            raise ValueError("Calibration probabilities and target must align")
        for class_name in CLASS_ORDER:
            binary_target = target.astype(str).eq(class_name).astype(int)
            scores = raw_probabilities[class_name].astype(float).to_numpy()
            if binary_target.nunique() < 2:
                raise ValueError(f"Calibration target lacks both outcomes for {class_name}")
            if self.config.method == "platt":
                model = LogisticRegression(solver="lbfgs")
                model.fit(scores.reshape(-1, 1), binary_target.to_numpy())
            else:
                model = IsotonicRegression(out_of_bounds="clip")
                model.fit(scores, binary_target.to_numpy())
            self._models[class_name] = model
        self._fitted = True

    def transform(self, raw_probabilities: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fitted")
        calibrated = pd.DataFrame(index=raw_probabilities.index)
        for class_name in CLASS_ORDER:
            scores = raw_probabilities[class_name].astype(float).to_numpy()
            model = self._models[class_name]
            if self.config.method == "platt":
                values = model.predict_proba(scores.reshape(-1, 1))[:, 1]
            else:
                values = model.predict(scores)
            calibrated[class_name] = np.asarray(values, dtype=float)
        row_sum = calibrated.sum(axis=1).replace(0.0, np.nan)
        calibrated = calibrated.div(row_sum, axis=0)
        if calibrated.isna().any().any():
            raise ValueError("Calibration produced invalid probability rows")
        return calibrated
