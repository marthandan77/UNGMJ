from __future__ import annotations

import numpy as np
import pandas as pd

from ung_forecast.training.barrier_selection import (
    BarrierCandidate,
    BarrierCandidateResult,
    BarrierSelectionResult,
)
from ung_forecast.training.dataset import TrainingDataset
from ung_forecast.training.runner_60m import SixtyMinuteFoldPlan, SixtyMinuteRunPlan
from ung_forecast.training.sixty_minute_evaluation import evaluate_sixty_minute_plan


def _plan() -> SixtyMinuteRunPlan:
    rows = 108
    index = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    phase = np.arange(rows)
    labels = np.array(["LOWER_FIRST", "UPPER_FIRST", "NEITHER"])[phase % 3]
    features = pd.DataFrame(
        {
            "lower_signal": (labels == "LOWER_FIRST").astype(float) + 0.02 * np.sin(phase),
            "upper_signal": (labels == "UPPER_FIRST").astype(float) + 0.02 * np.cos(phase),
            "neither_signal": (labels == "NEITHER").astype(float) + 0.01 * np.sin(phase / 2),
        },
        index=index,
    )
    target = pd.Series(labels, index=index, name="target")
    label_end = pd.Series(index + pd.Timedelta(minutes=60), index=index, name="label_end_time")
    metadata = pd.DataFrame(index=index)
    dataset = TrainingDataset(features, target, label_end, metadata)

    candidate = BarrierCandidate(1.0, 1.0)
    candidate_result = BarrierCandidateResult(
        candidate=candidate,
        accepted=True,
        validation_brier=0.5,
        training_rows=60,
        validation_rows=24,
        training_classes=("LOWER_FIRST", "NEITHER", "UPPER_FIRST"),
        validation_classes=("LOWER_FIRST", "NEITHER", "UPPER_FIRST"),
        rejection_reason=None,
    )
    selection = BarrierSelectionResult(selected=candidate, candidates=(candidate_result,))
    fold = SixtyMinuteFoldPlan(
        fold_number=1,
        selected_barrier=candidate,
        selection=selection,
        dataset=dataset,
        training_index=index[:60],
        validation_index=index[60:84],
        test_index=index[84:],
    )
    return SixtyMinuteRunPlan(folds=(fold,))


def test_evaluates_all_raw_calibrated_and_baseline_models() -> None:
    result = evaluate_sixty_minute_plan(_plan())
    metrics = result.aggregate
    assert len(result.folds) == 1
    assert metrics.unconditional.sample_count == 24
    assert metrics.recency_weighted.sample_count == 24
    assert metrics.plain_logistic_raw.sample_count == 24
    assert metrics.plain_logistic.sample_count == 24
    assert metrics.elastic_net_raw.sample_count == 24
    assert metrics.elastic_net.sample_count == 24
    assert np.isfinite(metrics.elastic_net.brier_score)
    assert np.isfinite(metrics.plain_logistic.brier_score)


def test_preserves_selected_barrier_and_fold_diagnostics() -> None:
    result = evaluate_sixty_minute_plan(_plan())
    fold = result.folds[0]
    assert fold.selected_lower_multiplier == 1.0
    assert fold.selected_upper_multiplier == 1.0
    assert fold.metrics.elastic_net.sample_count == 24
    assert fold.training_class_counts.total == 60
    assert fold.validation_class_counts.total == 24
    assert fold.test_class_counts.total == 24
    assert fold.training_class_counts.lower_first == 20
    assert fold.training_class_counts.upper_first == 20
    assert fold.training_class_counts.neither == 20
    assert fold.best_brier_model in {
        "unconditional",
        "recency_weighted",
        "plain_logistic_raw",
        "plain_logistic",
        "elastic_net_raw",
        "elastic_net",
    }
    assert fold.plain_calibration_brier_delta == (
        fold.metrics.plain_logistic.brier_score - fold.metrics.plain_logistic_raw.brier_score
    )
    assert fold.elastic_calibration_brier_delta == (
        fold.metrics.elastic_net.brier_score - fold.metrics.elastic_net_raw.brier_score
    )
