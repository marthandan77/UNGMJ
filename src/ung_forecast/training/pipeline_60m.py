"""End-to-end gated research pipeline for the 60-minute horizon."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from ung_forecast.models.elastic_net import ElasticNetConfig
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


def _selected_final_elastic_net_config(
    evaluation: SixtyMinuteEvaluationResult,
) -> ElasticNetConfig:
    """Choose the modal nested configuration with deterministic tie-breaking."""

    if not evaluation.folds:
        raise ValueError("Cannot select a final Elastic Net configuration without folds")
    counts: dict[tuple[float, float, int, int], int] = {}
    configs: dict[tuple[float, float, int, int], ElasticNetConfig] = {}
    for fold in evaluation.folds:
        config = fold.elastic_net_config
        key = (config.c, config.l1_ratio, config.max_iter, config.random_state)
        counts[key] = counts.get(key, 0) + 1
        configs[key] = config
    selected_key = min(
        counts,
        key=lambda key: (-counts[key], key[0], key[1], key[2], key[3]),
    )
    return configs[selected_key]


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
    final_elastic_net = _selected_final_elastic_net_config(evaluation)
    final_fit_config = replace(config.final_fit, elastic_net=final_elastic_net)
    artifact = fit_and_write_sixty_minute_artifact(
        market_data,
        feature_frame,
        volatility,
        plan=plan,
        report=report,
        artifact_root=artifact_root,
        config=final_fit_config,
    )
    if artifact.manifest.statistical_approved != report.statistically_approved:
        raise ValueError("Artifact approval state does not match the research report")
    if artifact.manifest.configuration_hash != config.final_fit.configuration_hash:
        raise ValueError("Artifact configuration hash does not match pipeline configuration")
    if artifact.elastic_net_config != final_elastic_net:
        raise ValueError("Artifact Elastic Net configuration does not match nested selection")
    return SixtyMinutePipelineResult(
        plan=plan,
        evaluation=evaluation,
        report=report,
        artifact=artifact,
    )
