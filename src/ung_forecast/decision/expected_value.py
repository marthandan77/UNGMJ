"""Economic decision formulas for buy and sell guidance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AdviceState(StrEnum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True, slots=True)
class DecisionEvaluation:
    expected_value: float
    lower_confidence_bound: float
    upper_confidence_bound: float
    advice_state: AdviceState


def _evaluate_interval(*, expected_value: float, uncertainty_std: float, z_value: float) -> DecisionEvaluation:
    if uncertainty_std < 0:
        raise ValueError("uncertainty_std cannot be negative")
    if z_value <= 0:
        raise ValueError("z_value must be positive")

    lower = expected_value - z_value * uncertainty_std
    upper = expected_value + z_value * uncertainty_std
    if lower > 0:
        state = AdviceState.POSITIVE
    elif upper < 0:
        state = AdviceState.NEGATIVE
    else:
        state = AdviceState.INDETERMINATE
    return DecisionEvaluation(expected_value, lower, upper, state)


def evaluate_sell(
    *,
    lower_probability: float,
    upper_probability: float,
    expected_decline: float,
    expected_upside: float,
    round_trip_cost: float,
    uncertainty_std: float,
    z_value: float,
) -> DecisionEvaluation:
    values = (
        lower_probability,
        upper_probability,
        expected_decline,
        expected_upside,
        round_trip_cost,
    )
    if any(value < 0 for value in values):
        raise ValueError("probabilities, movements, and costs cannot be negative")
    if lower_probability + upper_probability > 1.0 + 1e-12:
        raise ValueError("lower and upper probabilities cannot sum above one")

    expected_value = (
        lower_probability * expected_decline
        - upper_probability * expected_upside
        - round_trip_cost
    )
    return _evaluate_interval(
        expected_value=expected_value,
        uncertainty_std=uncertainty_std,
        z_value=z_value,
    )


def evaluate_buy(
    *,
    lower_probability: float,
    upper_probability: float,
    expected_decline: float,
    expected_upside: float,
    entry_cost: float,
    uncertainty_std: float,
    z_value: float,
) -> DecisionEvaluation:
    values = (
        lower_probability,
        upper_probability,
        expected_decline,
        expected_upside,
        entry_cost,
    )
    if any(value < 0 for value in values):
        raise ValueError("probabilities, movements, and costs cannot be negative")
    if lower_probability + upper_probability > 1.0 + 1e-12:
        raise ValueError("lower and upper probabilities cannot sum above one")

    expected_value = (
        upper_probability * expected_upside
        - lower_probability * expected_decline
        - entry_cost
    )
    return _evaluate_interval(
        expected_value=expected_value,
        uncertainty_std=uncertainty_std,
        z_value=z_value,
    )
