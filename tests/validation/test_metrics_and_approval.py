from __future__ import annotations

import pandas as pd

from ung_forecast.models.calibration import CalibrationConfig, MulticlassProbabilityCalibrator
from ung_forecast.schemas import OutcomeClass
from ung_forecast.validation.approval import (
    ApprovalCriteria,
    ValidationMetrics,
    evaluate_approval,
)
from ung_forecast.validation.metrics import (
    class_frequency_baseline,
    expected_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
)


def sample_target() -> pd.Series:
    values = [
        OutcomeClass.LOWER_FIRST.value,
        OutcomeClass.UPPER_FIRST.value,
        OutcomeClass.NEITHER.value,
    ] * 40
    return pd.Series(values, index=pd.RangeIndex(len(values)))


def sample_probabilities(target: pd.Series) -> pd.DataFrame:
    rows = []
    for value in target:
        row = {outcome.value: 0.1 for outcome in OutcomeClass}
        row[str(value)] = 0.8
        rows.append(row)
    return pd.DataFrame(rows, index=target.index)


def test_probabilistic_metrics_reward_correct_probabilities() -> None:
    target = sample_target()
    model = sample_probabilities(target)
    baseline = class_frequency_baseline(target)

    assert multiclass_brier_score(target, model) < multiclass_brier_score(target, baseline)
    assert multiclass_log_loss(target, model) < multiclass_log_loss(target, baseline)
    assert 0.0 <= expected_calibration_error(target, model) <= 1.0


def test_platt_calibrator_outputs_normalized_probabilities() -> None:
    target = sample_target()
    raw = sample_probabilities(target)
    calibrator = MulticlassProbabilityCalibrator(CalibrationConfig(method="platt"))
    calibrator.fit(raw, target)
    calibrated = calibrator.transform(raw)

    assert calibrated.index.equals(target.index)
    assert (calibrated >= 0.0).all().all()
    assert (calibrated <= 1.0).all().all()
    assert calibrated.sum(axis=1).round(10).eq(1.0).all()


def test_approval_requires_statistical_and_economic_improvement() -> None:
    baseline = ValidationMetrics(0.70, 1.20, 0.08, 0.0, 500)
    model = ValidationMetrics(0.60, 1.00, 0.06, 0.02, 500)
    decision = evaluate_approval(model, baseline, ApprovalCriteria(minimum_samples=250))
    assert decision.approved is True
    assert decision.reasons == ()


def test_approval_rejects_model_that_only_improves_accuracy_proxy() -> None:
    baseline = ValidationMetrics(0.70, 1.20, 0.08, 0.0, 500)
    model = ValidationMetrics(0.60, 1.00, 0.20, -0.01, 500)
    decision = evaluate_approval(model, baseline, ApprovalCriteria(minimum_samples=250))
    assert decision.approved is False
    assert "calibration_error_too_high" in decision.reasons
    assert "economic_value_not_positive" in decision.reasons
