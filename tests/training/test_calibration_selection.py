from __future__ import annotations

import numpy as np
import pandas as pd

from ung_forecast.training.calibration_selection import (
    CalibrationSelectionConfig,
    fit_deployment_calibrator,
    select_calibration_mode,
    split_calibration_index,
)


def _probabilities(rows: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    labels = pd.Series(
        np.array(["LOWER_FIRST", "UPPER_FIRST", "NEITHER"])[np.arange(rows) % 3],
        index=index,
    )
    probabilities = pd.DataFrame(
        {
            "LOWER_FIRST": np.where(labels.eq("LOWER_FIRST"), 0.75, 0.125),
            "UPPER_FIRST": np.where(labels.eq("UPPER_FIRST"), 0.75, 0.125),
            "NEITHER": np.where(labels.eq("NEITHER"), 0.75, 0.125),
        },
        index=index,
    )
    return probabilities, labels


def test_split_is_chronological_and_non_overlapping() -> None:
    probabilities, _ = _probabilities()
    fit_index, selection_index = split_calibration_index(
        probabilities.index,
        config=CalibrationSelectionConfig(
            fit_fraction=0.5,
            minimum_fit_rows=10,
            minimum_selection_rows=10,
        ),
    )
    assert fit_index[-1] < selection_index[0]
    assert fit_index.intersection(selection_index).empty
    assert len(fit_index) + len(selection_index) == len(probabilities)


def test_selection_is_determined_only_by_validation_data() -> None:
    probabilities, target = _probabilities()
    first = select_calibration_mode(
        probabilities,
        target,
        config=CalibrationSelectionConfig(),
    )
    changed_external_target = target.copy()
    changed_external_target.iloc[-1] = "LOWER_FIRST"
    second = select_calibration_mode(
        probabilities,
        changed_external_target,
        config=CalibrationSelectionConfig(),
    )
    assert first.fit_index.equals(second.fit_index)
    assert first.selection_index.equals(second.selection_index)


def test_identity_deployment_calibrator_preserves_probabilities() -> None:
    probabilities, target = _probabilities()
    calibrator = fit_deployment_calibrator(
        probabilities,
        target,
        mode="raw",
        calibrated_method="platt",
    )
    transformed = calibrator.transform(probabilities)
    pd.testing.assert_frame_equal(transformed, probabilities)
