from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from ung_forecast.training import pipeline_60m
from ung_forecast.training.barrier_selection import (
    BarrierCandidate,
    BarrierSelectionConfig,
)
from ung_forecast.training.pipeline_60m import (
    SixtyMinutePipelineConfig,
    run_sixty_minute_research_pipeline,
)
from ung_forecast.training.runner_60m import SixtyMinuteRunnerConfig
from ung_forecast.training.sixty_minute_artifact import SixtyMinuteFinalFitConfig


def _config() -> SixtyMinutePipelineConfig:
    return SixtyMinutePipelineConfig(
        runner=SixtyMinuteRunnerConfig(
            minimum_train_size=30,
            validation_size=10,
            test_size=10,
            purge_size=1,
            embargo_size=1,
            barrier_selection=BarrierSelectionConfig(
                candidates=(BarrierCandidate(1.0, 1.0),),
                minimum_training_rows=10,
                minimum_validation_rows=5,
                require_all_classes=False,
            ),
        ),
        final_fit=SixtyMinuteFinalFitConfig(
            configuration_hash="12345678-pipeline",
            calibration_rows=10,
            model_version="pipeline-v1",
        ),
    )


def test_runs_all_pipeline_stages_in_fixed_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    plan = cast(Any, SimpleNamespace())
    evaluation = cast(Any, SimpleNamespace())
    report = cast(Any, SimpleNamespace(statistically_approved=False))
    manifest = SimpleNamespace(
        statistical_approved=False,
        configuration_hash="12345678-pipeline",
    )
    artifact = cast(Any, SimpleNamespace(manifest=manifest))

    monkeypatch.setattr(
        pipeline_60m,
        "build_sixty_minute_run_plan",
        lambda *args, **kwargs: calls.append("plan") or plan,
    )
    monkeypatch.setattr(
        pipeline_60m,
        "evaluate_sixty_minute_plan",
        lambda *args, **kwargs: calls.append("evaluate") or evaluation,
    )
    monkeypatch.setattr(
        pipeline_60m,
        "build_sixty_minute_research_report",
        lambda *args, **kwargs: calls.append("report") or report,
    )
    monkeypatch.setattr(
        pipeline_60m,
        "fit_and_write_sixty_minute_artifact",
        lambda *args, **kwargs: calls.append("artifact") or artifact,
    )

    result = run_sixty_minute_research_pipeline(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.Series(dtype=float),
        artifact_root=tmp_path,
        config=_config(),
    )
    assert calls == ["plan", "evaluate", "report", "artifact"]
    assert result.report is report
    assert result.artifact is artifact


def test_rejects_artifact_approval_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = cast(Any, SimpleNamespace())
    evaluation = cast(Any, SimpleNamespace())
    report = cast(Any, SimpleNamespace(statistically_approved=True))
    artifact = cast(
        Any,
        SimpleNamespace(
            manifest=SimpleNamespace(
                statistical_approved=False,
                configuration_hash="12345678-pipeline",
            )
        ),
    )
    monkeypatch.setattr(pipeline_60m, "build_sixty_minute_run_plan", lambda *a, **k: plan)
    monkeypatch.setattr(pipeline_60m, "evaluate_sixty_minute_plan", lambda *a, **k: evaluation)
    monkeypatch.setattr(
        pipeline_60m,
        "build_sixty_minute_research_report",
        lambda *a, **k: report,
    )
    monkeypatch.setattr(
        pipeline_60m,
        "fit_and_write_sixty_minute_artifact",
        lambda *a, **k: artifact,
    )
    with pytest.raises(ValueError, match="approval state"):
        run_sixty_minute_research_pipeline(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.Series(dtype=float),
            artifact_root=tmp_path,
            config=_config(),
        )
