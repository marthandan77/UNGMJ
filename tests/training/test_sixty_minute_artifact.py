from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ung_forecast.artifacts import ArtifactState, discover_horizon_artifacts
from ung_forecast.horizons import HorizonKey
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
    fit_and_write_sixty_minute_artifact,
)
from ung_forecast.training.sixty_minute_report import SixtyMinuteResearchReport
from ung_forecast.validation.approval import ValidationMetrics

CONFIGURATION_HASH = "12345678-test-configuration"


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


def _report(*, approved: bool = True) -> SixtyMinuteResearchReport:
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
        statistically_approved=approved,
        approval_reasons=() if approved else ("insufficient_samples",),
        aggregate_best_brier_model="elastic_net",
        aggregate_metrics=metrics,
        folds=(),
    )


def test_writes_runtime_manifest_and_validated_artifacts(
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
        config=SixtyMinuteFinalFitConfig(
            configuration_hash=CONFIGURATION_HASH,
            calibration_rows=30,
            model_version="test-v1",
        ),
    )
    assert result.manifest.statistical_approved is True
    assert result.training_rows == 118
    assert result.calibration_rows == 30
    assert result.selected_lower_multiplier == 1.0
    assert result.selected_upper_multiplier == 1.2
    assert (result.version_directory / "model.joblib").exists()
    assert (result.version_directory / "calibrator.joblib").exists()
    assert (result.artifact_directory / "manifest.json").exists()
    discovered = discover_horizon_artifacts(
        tmp_path,
        expected_configuration_hash=CONFIGURATION_HASH,
    )[HorizonKey.MINUTES_60]
    assert discovered.state is ArtifactState.VALIDATED


def test_writes_research_only_state_when_statistical_gate_fails(
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
        report=_report(approved=False),
        artifact_root=tmp_path,
        config=SixtyMinuteFinalFitConfig(
            configuration_hash=CONFIGURATION_HASH,
            calibration_rows=30,
            model_version="research-v1",
        ),
    )
    assert result.manifest.statistical_approved is False
    discovered = discover_horizon_artifacts(
        tmp_path,
        expected_configuration_hash=CONFIGURATION_HASH,
    )[HorizonKey.MINUTES_60]
    assert discovered.state is ArtifactState.RESEARCH_ONLY


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
            config=SixtyMinuteFinalFitConfig(
                configuration_hash=CONFIGURATION_HASH,
                calibration_rows=30,
                model_version="bad-v1",
            ),
        )
