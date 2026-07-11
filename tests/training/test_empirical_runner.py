from __future__ import annotations

import numpy as np
import pandas as pd

from ung_forecast.training.dataset import TrainingDataset
from ung_forecast.training.empirical import EmpiricalRunConfig, run_empirical_evaluation
from ung_forecast.validation.approval import ApprovalCriteria


def synthetic_dataset(rows: int = 360) -> TrainingDataset:
    index = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    phase = np.arange(rows)
    target_values = np.array(["LOWER_FIRST", "UPPER_FIRST", "NEITHER"])[phase % 3]
    features = pd.DataFrame(
        {
            "signal_lower": (target_values == "LOWER_FIRST").astype(float) + 0.05 * np.sin(phase),
            "signal_upper": (target_values == "UPPER_FIRST").astype(float) + 0.05 * np.cos(phase),
            "signal_neither": (target_values == "NEITHER").astype(float) + 0.02 * np.sin(phase / 3),
        },
        index=index,
    )
    target = pd.Series(target_values, index=index, name="target")
    label_end = pd.Series(index + pd.Timedelta(hours=2), index=index, name="label_end_time")
    metadata = pd.DataFrame(
        {
            "maximum_upward_excursion": np.where(target_values == "UPPER_FIRST", 0.2, 0.05),
            "maximum_downward_excursion": np.where(target_values == "LOWER_FIRST", 0.2, 0.05),
        },
        index=index,
    )
    return TrainingDataset(features, target, label_end, metadata)


def economic_value(target: pd.Series, probabilities: pd.DataFrame, metadata: pd.DataFrame) -> float:
    correct_probability = np.array(
        [probabilities.loc[idx, value] for idx, value in target.items()],
        dtype=float,
    )
    return float(correct_probability.mean() - 0.34)


def test_empirical_runner_stays_research_only_without_economic_evaluator() -> None:
    result = run_empirical_evaluation(
        synthetic_dataset(),
        run_config=EmpiricalRunConfig(120, 45, 45, 3, 3),
        approval_criteria=ApprovalCriteria(minimum_samples=30, maximum_calibration_error=0.5),
    )
    assert result.approved is False
    assert "economic_evaluator_not_configured" in result.approval_reasons


def test_empirical_runner_executes_walk_forward_with_economic_evidence() -> None:
    result = run_empirical_evaluation(
        synthetic_dataset(),
        run_config=EmpiricalRunConfig(120, 45, 45, 3, 3),
        approval_criteria=ApprovalCriteria(
            minimum_samples=30,
            maximum_calibration_error=0.5,
            minimum_economic_value=-1.0,
        ),
        economic_evaluator=economic_value,
    )
    assert len(result.folds) >= 1
    assert result.aggregate_model.sample_count > 0
    assert np.isfinite(result.aggregate_model.brier_score)
    assert np.isfinite(result.aggregate_model.log_loss)
