from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ung_forecast.dashboard import build_forecast_card
from ung_forecast.explanations.renderer import UserIntent
from ung_forecast.forecast_service import assemble_forecast_record
from ung_forecast.horizons import HorizonKey
from ung_forecast.schemas import (
    DataFreshness,
    HorizonForecast,
    ModelIdentity,
    ModelStatus,
    ProbabilityForecast,
)


def test_dashboard_card_only_renders_existing_forecast_fields() -> None:
    timestamp = datetime(2026, 7, 10, 14, 30, tzinfo=UTC)
    forecast = HorizonForecast(
        timestamp=timestamp,
        horizon=HorizonKey.HOURS_4,
        current_price=10.0,
        lower_price_area=9.8,
        upper_price_area=10.3,
        probabilities=ProbabilityForecast(lower_first=0.25, upper_first=0.55, neither=0.20),
        sell_expected_value=-0.04,
        buy_expected_value=0.03,
        sell_ev_lower_confidence_bound=-0.06,
        buy_ev_lower_confidence_bound=0.01,
        confidence_score=0.64,
        model_status=ModelStatus.VALIDATED,
        data_freshness=DataFreshness.CURRENT,
        identity=ModelIdentity(
            data_version="data-v1",
            feature_version="features-v1",
            model_version="model-v1",
            configuration_hash="12345678abcdef",
        ),
    )
    record = assemble_forecast_record(
        forecast,
        intent=UserIntent.BUYING,
        expires_at=timestamp + timedelta(hours=4),
    )
    card = build_forecast_card(record)

    assert card.headline == "BUYING NOW LOOKS REASONABLE"
    assert card.upper_probability_pct == 55.0
    assert card.lower_probability_pct == 25.0
    assert card.model_version == "model-v1"
    assert card.model_status == "VALIDATED"
