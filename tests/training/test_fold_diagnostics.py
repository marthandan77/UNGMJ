from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ung_forecast.training.fold_diagnostics import (
    coefficient_diagnostics,
    feature_drift_diagnostics,
)


def test_feature_drift_is_zero_for_identical_feature_distributions() -> None:
    frame = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0],
            "b": [1.0, 0.0, 1.0, 0.0],
        }
    )
    result = feature_drift_diagnostics(frame, frame.copy(), frame.copy())
    assert result.validation_mahalanobis == pytest.approx(0.0)
    assert result.test_mahalanobis == pytest.approx(0.0)


def test_feature_drift_detects_a_mean_shift() -> None:
    training = pd.DataFrame(
        {
            "a": [-1.0, 0.0, 1.0, 2.0],
            "b": [0.0, 1.0, 0.0, 1.0],
        }
    )
    validation = training + pd.Series({"a": 1.0, "b": 0.0})
    test = training + pd.Series({"a": 2.0, "b": 0.0})
    result = feature_drift_diagnostics(training, validation, test)
    assert result.validation_mahalanobis > 0.0
    assert result.test_mahalanobis > result.validation_mahalanobis


def test_coefficient_diagnostics_records_exact_norm_and_sparsity() -> None:
    coefficients = pd.DataFrame([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    result = coefficient_diagnostics(coefficients)
    assert result.l2_norm == pytest.approx(5.0)
    assert result.nonzero_count == 2
    assert result.total_count == 6


def test_diagnostics_reject_non_finite_values() -> None:
    frame = pd.DataFrame({"a": [0.0, np.nan], "b": [1.0, 2.0]})
    with pytest.raises(ValueError, match="finite"):
        feature_drift_diagnostics(frame, frame, frame)
