"""Chronological validation-only selection of raw or calibrated probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from ung_forecast.models.calibration import CalibrationConfig, MulticlassProbabilityCalibrator
from ung_forecast.validation.metrics import multiclass_brier_score

ProbabilityMode = Literal["raw", "calibrated"]


@dataclass(frozen=True, slots=True)
class CalibrationSelectionConfig:
    method: str = "platt"
    fit_fraction: float = 0.5
    minimum_fit_rows: int = 10
    minimum_selection_rows: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < self.fit_fraction < 1.0:
            raise ValueError("fit_fraction must be strictly between zero and one")
        if self.minimum_fit_rows <= 0 or self.minimum_selection_rows <= 0:
            raise ValueError("Calibration split minimum rows must be positive")
        CalibrationConfig(method=self.method)


@dataclass(frozen=True, slots=True)
class CalibrationModeSelection:
    mode: ProbabilityMode
    calibrator: MulticlassProbabilityCalibrator
    fit_index: pd.Index
    selection_index: pd.Index
    raw_brier: float
    calibrated_brier: float


def split_calibration_index(
    index: pd.Index,
    *,
    config: CalibrationSelectionConfig,
) -> tuple[pd.Index, pd.Index]:
    if len(index) < config.minimum_fit_rows + config.minimum_selection_rows:
        raise ValueError("Validation interval is too short for calibration fit and selection")
    proposed_fit_rows = int(len(index) * config.fit_fraction)
    fit_rows = max(config.minimum_fit_rows, proposed_fit_rows)
    fit_rows = min(fit_rows, len(index) - config.minimum_selection_rows)
    fit_index = index[:fit_rows]
    selection_index = index[fit_rows:]
    if len(fit_index) < config.minimum_fit_rows:
        raise ValueError("Calibration fit partition is below its minimum size")
    if len(selection_index) < config.minimum_selection_rows:
        raise ValueError("Calibration selection partition is below its minimum size")
    return fit_index, selection_index


def select_calibration_mode(
    validation_raw: pd.DataFrame,
    validation_target: pd.Series,
    *,
    config: CalibrationSelectionConfig,
) -> CalibrationModeSelection:
    if not validation_raw.index.equals(validation_target.index):
        raise ValueError("Validation probabilities and targets must align")
    fit_index, selection_index = split_calibration_index(validation_raw.index, config=config)
    calibrator = MulticlassProbabilityCalibrator(CalibrationConfig(method=config.method))
    calibrator.fit(validation_raw.loc[fit_index], validation_target.loc[fit_index])
    raw_selection = validation_raw.loc[selection_index]
    calibrated_selection = calibrator.transform(raw_selection)
    selection_target = validation_target.loc[selection_index]
    raw_brier = multiclass_brier_score(selection_target, raw_selection)
    calibrated_brier = multiclass_brier_score(selection_target, calibrated_selection)
    mode: ProbabilityMode = "calibrated" if calibrated_brier < raw_brier else "raw"
    return CalibrationModeSelection(
        mode=mode,
        calibrator=calibrator,
        fit_index=fit_index,
        selection_index=selection_index,
        raw_brier=raw_brier,
        calibrated_brier=calibrated_brier,
    )


def fit_deployment_calibrator(
    raw_probabilities: pd.DataFrame,
    target: pd.Series,
    *,
    mode: ProbabilityMode,
    calibrated_method: str,
) -> MulticlassProbabilityCalibrator:
    method = calibrated_method if mode == "calibrated" else "identity"
    calibrator = MulticlassProbabilityCalibrator(CalibrationConfig(method=method))
    calibrator.fit(raw_probabilities, target)
    return calibrator
