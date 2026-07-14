from __future__ import annotations

import pytest

from ung_forecast.confidence import ConfidenceComponents, calculate_confidence
from ung_forecast.decision import AdviceState, evaluate_buy, evaluate_sell


def test_sell_advice_requires_positive_conservative_bound() -> None:
    result = evaluate_sell(
        lower_probability=0.7,
        upper_probability=0.2,
        expected_decline=0.20,
        expected_upside=0.08,
        round_trip_cost=0.01,
        uncertainty_std=0.01,
        z_value=1.64,
    )

    assert result.expected_value == pytest.approx(0.114)
    assert result.lower_confidence_bound > 0
    assert result.advice_state is AdviceState.POSITIVE


def test_sell_advice_is_indeterminate_when_interval_crosses_zero() -> None:
    result = evaluate_sell(
        lower_probability=0.5,
        upper_probability=0.4,
        expected_decline=0.10,
        expected_upside=0.09,
        round_trip_cost=0.01,
        uncertainty_std=0.01,
        z_value=1.64,
    )

    assert result.lower_confidence_bound < 0 < result.upper_confidence_bound
    assert result.advice_state is AdviceState.INDETERMINATE


def test_buy_and_sell_formulas_are_directionally_symmetric() -> None:
    sell = evaluate_sell(
        lower_probability=0.6,
        upper_probability=0.3,
        expected_decline=0.15,
        expected_upside=0.10,
        round_trip_cost=0.01,
        uncertainty_std=0.0,
        z_value=1.0,
    )
    buy = evaluate_buy(
        lower_probability=0.6,
        upper_probability=0.3,
        expected_decline=0.15,
        expected_upside=0.10,
        entry_cost=0.01,
        uncertainty_std=0.0,
        z_value=1.0,
    )

    assert sell.expected_value == pytest.approx(0.05)
    assert buy.expected_value == pytest.approx(-0.07)
    assert sell.advice_state is AdviceState.POSITIVE
    assert buy.advice_state is AdviceState.NEGATIVE


def test_confidence_is_exact_product_of_evidence_components() -> None:
    components = ConfidenceComponents(
        probability_margin=0.4,
        distribution_similarity=0.8,
        calibration_quality=0.9,
        perturbation_stability=0.75,
    )

    assert calculate_confidence(components) == pytest.approx(0.216)


def test_confidence_rejects_invalid_component() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        ConfidenceComponents(
            probability_margin=1.2,
            distribution_similarity=0.8,
            calibration_quality=0.9,
            perturbation_stability=0.75,
        )
