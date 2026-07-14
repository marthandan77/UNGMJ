"""Summarize the forecast term structure without changing model outputs."""

from __future__ import annotations

from dataclasses import dataclass

from ung_forecast.horizons import HORIZON_SPECS, HorizonKey
from ung_forecast.schemas import HorizonForecast, ModelStatus


@dataclass(frozen=True, slots=True)
class HorizonDirection:
    horizon: HorizonKey
    directional_expectation: float


@dataclass(frozen=True, slots=True)
class CrossHorizonSummary:
    directions: tuple[HorizonDirection, ...]
    coherence_score: float
    transition_deltas: tuple[float, ...]
    validated_horizon_count: int
    structure_label: str


_ORDER = tuple(HORIZON_SPECS.keys())


def _structure_label(values: list[float]) -> str:
    if not values:
        return "insufficient_validated_horizons"
    if all(value > 0 for value in values):
        return "broad_upward_term_structure"
    if all(value < 0 for value in values):
        return "broad_downward_term_structure"
    if values[0] < 0 < values[-1]:
        return "short_term_weakness_longer_term_strength"
    if values[0] > 0 > values[-1]:
        return "short_term_strength_longer_term_weakness"
    return "mixed_term_structure"


def summarize_horizons(forecasts: dict[HorizonKey, HorizonForecast]) -> CrossHorizonSummary:
    ordered: list[HorizonDirection] = []
    for horizon in _ORDER:
        forecast = forecasts.get(horizon)
        if forecast is None or forecast.model_status is not ModelStatus.VALIDATED:
            continue
        expectation = forecast.probabilities.upper_first - forecast.probabilities.lower_first
        ordered.append(HorizonDirection(horizon=horizon, directional_expectation=expectation))

    values = [item.directional_expectation for item in ordered]
    deltas = tuple(values[index + 1] - values[index] for index in range(len(values) - 1))
    if len(values) <= 1:
        coherence = 0.0
    else:
        coherence = 1.0 - sum(abs(delta) / 2.0 for delta in deltas) / len(deltas)
        coherence = max(0.0, min(1.0, coherence))

    return CrossHorizonSummary(
        directions=tuple(ordered),
        coherence_score=coherence,
        transition_deltas=deltas,
        validated_horizon_count=len(ordered),
        structure_label=_structure_label(values),
    )
