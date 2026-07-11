"""Render human language only from validated forecast and decision fields."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ung_forecast.decision.expected_value import AdviceState
from ung_forecast.schemas import DataFreshness, HorizonForecast, ModelStatus


class UserIntent(StrEnum):
    SELLING = "SELLING"
    BUYING = "BUYING"


@dataclass(frozen=True, slots=True)
class Explanation:
    headline: str
    body: str
    evidence_codes: tuple[str, ...]


def _decision_state(forecast: HorizonForecast, intent: UserIntent) -> AdviceState:
    if intent is UserIntent.SELLING:
        lower = forecast.sell_ev_lower_confidence_bound
        upper = 2.0 * forecast.sell_expected_value - lower
    else:
        lower = forecast.buy_ev_lower_confidence_bound
        upper = 2.0 * forecast.buy_expected_value - lower
    if lower > 0:
        return AdviceState.POSITIVE
    if upper < 0:
        return AdviceState.NEGATIVE
    return AdviceState.INDETERMINATE


def render_explanation(forecast: HorizonForecast, intent: UserIntent) -> Explanation:
    if forecast.data_freshness is not DataFreshness.CURRENT:
        return Explanation(
            headline="FORECAST UNAVAILABLE",
            body="The latest market data is stale or invalid, so no directional guidance is issued.",
            evidence_codes=("data_not_current",),
        )
    if forecast.model_status is not ModelStatus.VALIDATED:
        return Explanation(
            headline="RESEARCH STATUS ONLY",
            body="This forecast horizon has not passed the required out-of-sample approval tests.",
            evidence_codes=("model_not_validated",),
        )

    state = _decision_state(forecast, intent)
    if intent is UserIntent.SELLING:
        if state is AdviceState.POSITIVE:
            return Explanation(
                headline="SELLING NOW LOOKS REASONABLE",
                body="The conservative expected value of selling is positive after accounting for missed-upside risk and modeled costs.",
                evidence_codes=("sell_ev_lcb_positive",),
            )
        if state is AdviceState.NEGATIVE:
            return Explanation(
                headline="DO NOT SELL YET",
                body="The modeled cost of missing further upside is greater than the expected benefit of buying back lower.",
                evidence_codes=("sell_ev_upper_bound_negative",),
            )
        return Explanation(
            headline="NO CLEAR ADVANTAGE — WAIT",
            body="The expected value of selling is not distinguishable from zero at the configured confidence level.",
            evidence_codes=("sell_ev_interval_crosses_zero",),
        )

    if state is AdviceState.POSITIVE:
        return Explanation(
            headline="BUYING NOW LOOKS REASONABLE",
            body="The conservative expected value of buying is positive after accounting for further-downside risk and modeled costs.",
            evidence_codes=("buy_ev_lcb_positive",),
        )
    if state is AdviceState.NEGATIVE:
        return Explanation(
            headline="DO NOT BUY YET",
            body="The modeled risk of further decline is greater than the expected benefit of an immediate entry.",
            evidence_codes=("buy_ev_upper_bound_negative",),
        )
    return Explanation(
        headline="NO CLEAR ADVANTAGE — WAIT",
        body="The expected value of buying is not distinguishable from zero at the configured confidence level.",
        evidence_codes=("buy_ev_interval_crosses_zero",),
    )
