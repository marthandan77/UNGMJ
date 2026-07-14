from __future__ import annotations

from datetime import UTC, datetime

from ung_forecast.explanations import UserIntent, render_explanation
from ung_forecast.horizons import HorizonKey
from ung_forecast.schemas import (
    DataFreshness,
    HorizonForecast,
    ModelIdentity,
    ModelStatus,
    ProbabilityForecast,
)


def make_forecast(
    *,
    sell_ev: float = 0.05,
    sell_lcb: float = 0.01,
    buy_ev: float = -0.03,
    buy_lcb: float = -0.06,
    model_status: ModelStatus = ModelStatus.VALIDATED,
    freshness: DataFreshness = DataFreshness.CURRENT,
) -> HorizonForecast:
    return HorizonForecast(
        timestamp=datetime(2026, 7, 11, 15, 35, tzinfo=UTC),
        horizon=HorizonKey.HOURS_4,
        current_price=10.0,
        lower_price_area=9.5,
        upper_price_area=10.5,
        probabilities=ProbabilityForecast(lower_first=0.55, upper_first=0.3, neither=0.15),
        sell_expected_value=sell_ev,
        buy_expected_value=buy_ev,
        sell_ev_lower_confidence_bound=sell_lcb,
        buy_ev_lower_confidence_bound=buy_lcb,
        confidence_score=0.6,
        model_status=model_status,
        data_freshness=freshness,
        identity=ModelIdentity(
            data_version="data-v1",
            feature_version="features-v1",
            model_version="model-v1",
            configuration_hash="12345678abcdef",
        ),
    )


def test_sell_explanation_comes_from_positive_lcb() -> None:
    result = render_explanation(make_forecast(), UserIntent.SELLING)
    assert result.headline == "SELLING NOW LOOKS REASONABLE"
    assert result.evidence_codes == ("sell_ev_lcb_positive",)


def test_unvalidated_model_never_receives_actionable_language() -> None:
    result = render_explanation(
        make_forecast(model_status=ModelStatus.RESEARCH_ONLY),
        UserIntent.SELLING,
    )
    assert result.headline == "RESEARCH STATUS ONLY"
    assert "sell" not in result.body.lower()


def test_stale_data_suspends_guidance() -> None:
    result = render_explanation(
        make_forecast(freshness=DataFreshness.STALE),
        UserIntent.BUYING,
    )
    assert result.headline == "FORECAST UNAVAILABLE"
    assert result.evidence_codes == ("data_not_current",)


def test_indeterminate_interval_returns_wait() -> None:
    result = render_explanation(
        make_forecast(sell_ev=0.0, sell_lcb=-0.02),
        UserIntent.SELLING,
    )
    assert result.headline == "NO CLEAR ADVANTAGE — WAIT"
    assert result.evidence_codes == ("sell_ev_interval_crosses_zero",)
