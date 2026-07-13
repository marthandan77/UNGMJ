"""End-to-end gated research pipeline for the 60-minute horizon."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ung_forecast.training.runner_60m import (
    SixtyMinuteRunnerConfig,
    SixtyMinuteRunPlan,
    build_sixty_minute_run_plan,
)
from ung_forecast.training.sixty_minute_artifact import (
    SixtyMinuteArtifactResult,
    SixtyMinuteFinalFitConfig,
    fit_and_write_sixty_minute_artifact,
)
from ung_forecast.training.sixty_minute_evaluation import (
    SixtyMinuteEvaluationConfig,
    SixtyMinuteEvaluationResult,
    evaluate_sixty_minute_plan,
)
from ung_forecast.training.sixty_minute_report import (
    SixtyMinuteResearchReport,
    build_sixty_minute_research_report,
)
from ung_forecast.validation.approval import StatisticalApprovalCriteria


@dataclass(frozen=True, slots=True)
class SixtyMinutePipelineConfig:
    runner: SixtyMinuteRunnerConfig
    final_fit: SixtyMinuteFinalFitConfig
    evaluation: SixtyMinuteEvaluationConfig = SixtyMinuteEvaluationConfig()
    approval: StatisticalApprovalCriteria = StatisticalApprovalCriteria()


@dataclass(frozen=True, slots=True)
class SixtyMinutePipelineResult:
    plan: SixtyMinuteRunPlan
    evaluation: SixtyMinuteEvaluationResult
    report: SixtyMinuteResearchReport
    artifact: SixtyMinuteArtifactResult


def run_sixty_minute_research_pipeline(
    market_data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    volatility: pd.Series,
    *,
    artifact_root: str | Path,
    config: SixtyMinutePipelineConfig,
) -> SixtyMinutePipelineResult:
    """Run every required gate in order and emit a runtime-compatible artifact."""

    plan = build_sixty_minute_run_plan(
        market_data,
        feature_frame,
        volatility,
        config=config.runner,
    )
    evaluation = evaluate_sixty_minute_plan(plan, config=config.evaluation)
    report = build_sixty_minute_research_report(evaluation, criteria=config.approval)
    artifact = fit_and_write_sixty_minute_artifact(
        market_data,
        feature_frame,
        volatility,
        plan=plan,
        report=report,
        artifact_root=artifact_root,
        config=config.final_fit,
    )
    if artifact.manifest.statistical_approved != report.statistically_approved:
        raise ValueError("Artifact approval state does not match the research report")
    if artifact.manifest.configuration_hash != config.final_fit.configuration_hash:
        raise ValueError("Artifact configuration hash does not match pipeline configuration")
    return SixtyMinutePipelineResult(
        plan=plan,
        evaluation=evaluation,
        report=report,
        artifact=artifact,
    )
