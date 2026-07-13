from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ung_forecast.training import sixty_minute_artifact
from ung_forecast.training.barrier_selection import (
    BarrierCandidate,
    BarrierCandidateResult,
    BarrierSelectionResult,
)
from ung_forecast.training.dataset import TrainingDataset
from ung_forecast.training.empirical import BenchmarkMetrics
from ung_forecast.training.runner_60m import SixtyMinuteFoldPlan, SixtyMinuteRunPlan
from ung_forecast.training.sixty_minute_artifact import (
    SixtyMinuteFinalFitConfig,
    SixtyMinuteModelBundle,
    fit_and_write_sixty_minute_artifact,
)
from ung_forecast.training.sixty_minute_report import SixtyMinuteResearchReport
from ung_forecast.validation.approval import ValidationMetrics


def _dataset(rows: int = 150) -> TrainingDataset:
    index = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    phase = np.arange(rows)
    labels = np.array(["LOWER_FIRST", "UPPER_FIRST", "NEITHER"])[phase % 3]
    features = pd.DataFrame(
        {
            "lower_signal": (labels == "LOWER_FIRST").astype(float) + 0.01 * np.sin(phase),
            "upper_signal": (labels == "UPPER_FIRST").astype(float) + 0.01 * np.cos(phase),
            "neither_signal": (labels == "NEITHER").astype(float),
        },
        index=index,
    )
    target = pd.Series(labels, index=index, name="target")
    label_end = pd.Series(index + pd.Timedelta(minutes=10), index=index, name="label_end_time")
    metadata = pd.DataFrame(index=index)
    return TrainingDataset(features, target, label_end, metadata)


def _plan(dataset: TrainingDataset) -> SixtyMinuteRunPlan:
    candidate = BarrierCandidate(1.0, 1.2)
    result = BarrierCandidateResult(
        candidate=candidate,
        accepted=True,
        validation_brier=0.5,
        training_rows=60,
        validation_rows=30,
        training_classes=("LOWER_FIRST", "NEITHER", "UPPER_FIRST"),
        validation_classes=("LOWER_FIRST", "NEITHER", "UPPER_FIRST"),
        rejection_reason=None,
    )
    selection = BarrierSelectionResult(selected=candidate, candidates=(result,))
    index = pd.DatetimeIndex(dataset.features.index)
    fold = SixtyMinuteFoldPlan(
        fold_number=1,
        selected_barrier=candidate,
        selection=selection,
        dataset=dataset,
        training_index=index[:60],
        validation_index=index[60:90],
        test_index=index[90:120],
    )
    return SixtyMinuteRunPlan(folds=(fold,))


def _metric(value: float = 0.2) -> ValidationMetrics:
    return ValidationMetrics(value, value, 0.05, float("nan"), 60)


def _report() -> SixtyMinuteResearchReport:
    metrics = BenchmarkMetrics(
        unconditional=_metric(0.7),
        recency_weighted=_metric(0.65),
        plain_logistic_raw=_metric(0.4),
        plain_logistic=_metric(0.35),
        elastic_net_raw=_metric(0.3),
        elastic_net=_metric(0.2),
    )
    return SixtyMinuteResearchReport(
        horizon_key="60m",
        fold_count=1,
        statistically_approved=True,
        approval_reasons=(),
        aggregate_best_brier_model="elastic_net",
        aggregate_metrics=metrics,
        folds=(),
    )


def test_writes_reloads_and_keeps_artifact_research_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    monkeypatch.setattr(sixty_minute_artifact, "build_training_dataset", lambda *a, **k: dataset)
    result = fit_and_write_sixty_minute_artifact(
        pd.DataFrame(index=dataset.features.index),
        dataset.features,
        pd.Series(1.0, index=dataset.features.index),
        plan=_plan(dataset),
        report=_report(),
        artifact_root=tmp_path,
        config=SixtyMinuteFinalFitConfig(calibration_rows=30, model_version="test-v1"),
    )
    assert isinstance(result.bundle, SixtyMinuteModelBundle)
    assert result.metadata.approved is False
    assert result.metadata.artifact_sha256
    assert result.training_rows == 118
    assert result.calibration_rows == 30
    assert (result.artifact_directory / "model.joblib").exists()
    assert (result.artifact_directory / "metadata.json").exists()


def test_rejects_calibration_tail_without_all_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    altered = dataset.target.copy()
    altered.iloc[-30:] = "NEITHER"
    incomplete = TrainingDataset(dataset.features, altered, dataset.label_end_time, dataset.metadata)
    monkeypatch.setattr(sixty_minute_artifact, "build_training_dataset", lambda *a, **k: incomplete)
    with pytest.raises(ValueError, match="calibration block"):
        fit_and_write_sixty_minute_artifact(
            pd.DataFrame(index=dataset.features.index),
            dataset.features,
            pd.Series(1.0, index=dataset.features.index),
            plan=_plan(dataset),
            report=_report(),
            artifact_root=tmp_path,
            config=SixtyMinuteFinalFitConfig(calibration_rows=30, model_version="bad-v1"),
        )
