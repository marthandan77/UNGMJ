from __future__ import annotations

from datetime import UTC, datetime

from ung_forecast.coordinator import summarize_horizons
from ung_forecast.horizons import HorizonKey
from ung_forecast.schemas import (
    DataFreshness,
    HorizonForecast,
    ModelIdentity,
    ModelStatus,
    ProbabilityForecast,
)


def forecast(horizon: HorizonKey, lower: float, upper: float) -> HorizonForecast:
    return HorizonForecast(
        timestamp=datetime(2026, 7, 11, 15, 35, tzinfo=UTC),
        horizon=horizon,
        current_price=10.0,
        lower_price_area=9.5,
        upper_price_area=10.5,
        probabilities=ProbabilityForecast(
            lower_first=lower,
            upper_first=upper,
            neither=1.0 - lower - upper,
        ),
        sell_expected_value=0.0,
        buy_expected_value=0.0,
        sell_ev_lower_confidence_bound=-0.1,
        buy_ev_lower_confidence_bound=-0.1,
        confidence_score=0.5,
        model_status=ModelStatus.VALIDATED,
        data_freshness=DataFreshness.CURRENT,
        identity=ModelIdentity(
            data_version="data-v1",
            feature_version="features-v1",
            model_version=f"model-{horizon.value}",
            configuration_hash="12345678abcdef",
        ),
    )


def test_coordinator_does_not_mutate_probabilities() -> None:
    forecasts = {
        HorizonKey.MINUTES_60: forecast(HorizonKey.MINUTES_60, 0.6, 0.3),
        HorizonKey.DAYS_7: forecast(HorizonKey.DAYS_7, 0.2, 0.7),
    }
    before = {key: value.model_dump() for key, value in forecasts.items()}

    summary = summarize_horizons(forecasts)

    assert {key: value.model_dump() for key, value in forecasts.items()} == before
    assert summary.structure_label == "short_term_weakness_longer_term_strength"
    assert summary.validated_horizon_count == 2


def test_coherence_is_bounded() -> None:
    summary = summarize_horizons(
        {
            HorizonKey.MINUTES_60: forecast(HorizonKey.MINUTES_60, 0.2, 0.7),
            HorizonKey.HOURS_4: forecast(HorizonKey.HOURS_4, 0.3, 0.6),
            HorizonKey.DAY_1: forecast(HorizonKey.DAY_1, 0.4, 0.5),
        }
    )
    assert 0.0 <= summary.coherence_score <= 1.0
    assert summary.structure_label == "broad_upward_term_structure"
