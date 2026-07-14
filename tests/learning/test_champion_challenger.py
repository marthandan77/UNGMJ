from __future__ import annotations

from ung_forecast.horizons import HorizonKey
from ung_forecast.learning import CandidateEvaluation, ChampionChallengerManager
from ung_forecast.validation.approval import ApprovalDecision, ValidationMetrics


def metrics() -> ValidationMetrics:
    return ValidationMetrics(
        brier_score=0.5,
        log_loss=0.9,
        calibration_error=0.05,
        economic_value=0.02,
        sample_count=500,
    )


def test_challenger_cannot_promote_without_shadow_samples() -> None:
    candidate = CandidateEvaluation(
        horizon=HorizonKey.MINUTES_60,
        champion_version="champion-v1",
        challenger_version="challenger-v2",
        champion_metrics=metrics(),
        challenger_metrics=metrics(),
        approval=ApprovalDecision(approved=True, reasons=()),
        shadow_sample_count=20,
        rollback_version="champion-v1",
    )

    decision = ChampionChallengerManager(minimum_shadow_samples=100).evaluate(candidate)

    assert decision.promote is False
    assert "insufficient_shadow_samples" in decision.reasons


def test_challenger_cannot_promote_without_rollback() -> None:
    candidate = CandidateEvaluation(
        horizon=HorizonKey.DAY_1,
        champion_version="champion-v1",
        challenger_version="challenger-v2",
        champion_metrics=metrics(),
        challenger_metrics=metrics(),
        approval=ApprovalDecision(approved=True, reasons=()),
        shadow_sample_count=150,
        rollback_version=None,
    )

    decision = ChampionChallengerManager().evaluate(candidate)

    assert decision.promote is False
    assert "rollback_version_missing" in decision.reasons


def test_approved_shadowed_challenger_can_be_promoted() -> None:
    candidate = CandidateEvaluation(
        horizon=HorizonKey.DAYS_7,
        champion_version="champion-v1",
        challenger_version="challenger-v2",
        champion_metrics=metrics(),
        challenger_metrics=metrics(),
        approval=ApprovalDecision(approved=True, reasons=()),
        shadow_sample_count=150,
        rollback_version="champion-v1",
    )

    decision = ChampionChallengerManager().evaluate(candidate)

    assert decision.promote is True
    assert decision.new_champion_version == "challenger-v2"
    assert decision.rollback_version == "champion-v1"
