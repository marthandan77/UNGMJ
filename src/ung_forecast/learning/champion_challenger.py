"""Controlled champion-challenger evaluation and promotion metadata."""

from __future__ import annotations

from dataclasses import dataclass

from ung_forecast.horizons import HorizonKey
from ung_forecast.validation.approval import ApprovalDecision, ValidationMetrics


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    horizon: HorizonKey
    champion_version: str
    challenger_version: str
    champion_metrics: ValidationMetrics
    challenger_metrics: ValidationMetrics
    approval: ApprovalDecision
    shadow_sample_count: int
    rollback_version: str | None

    def __post_init__(self) -> None:
        if self.shadow_sample_count < 0:
            raise ValueError("shadow_sample_count cannot be negative")
        if self.champion_version == self.challenger_version:
            raise ValueError("Champion and challenger versions must differ")


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promote: bool
    reasons: tuple[str, ...]
    new_champion_version: str | None
    rollback_version: str | None


class ChampionChallengerManager:
    def __init__(self, *, minimum_shadow_samples: int = 100) -> None:
        if minimum_shadow_samples <= 0:
            raise ValueError("minimum_shadow_samples must be positive")
        self.minimum_shadow_samples = minimum_shadow_samples

    def evaluate(self, candidate: CandidateEvaluation) -> PromotionDecision:
        reasons: list[str] = list(candidate.approval.reasons)
        if not candidate.approval.approved:
            reasons.append("challenger_not_approved")
        if candidate.shadow_sample_count < self.minimum_shadow_samples:
            reasons.append("insufficient_shadow_samples")
        if candidate.rollback_version is None:
            reasons.append("rollback_version_missing")

        if reasons:
            return PromotionDecision(
                promote=False,
                reasons=tuple(dict.fromkeys(reasons)),
                new_champion_version=None,
                rollback_version=candidate.rollback_version,
            )

        return PromotionDecision(
            promote=True,
            reasons=(),
            new_champion_version=candidate.challenger_version,
            rollback_version=candidate.champion_version,
        )
