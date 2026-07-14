"""Confidence calculation from probability separation and model health."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConfidenceComponents:
    probability_margin: float
    distribution_similarity: float
    calibration_quality: float
    perturbation_stability: float

    def __post_init__(self) -> None:
        for name, value in (
            ("probability_margin", self.probability_margin),
            ("distribution_similarity", self.distribution_similarity),
            ("calibration_quality", self.calibration_quality),
            ("perturbation_stability", self.perturbation_stability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


def calculate_confidence(components: ConfidenceComponents) -> float:
    """Return multiplicative confidence in [0, 1]."""

    return (
        components.probability_margin
        * components.distribution_similarity
        * components.calibration_quality
        * components.perturbation_stability
    )
