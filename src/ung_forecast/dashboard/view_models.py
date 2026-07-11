"""Transform forecast records into UI-ready fields without quantitative logic."""

from __future__ import annotations

from dataclasses import dataclass

from ung_forecast.horizons import HORIZON_SPECS
from ung_forecast.persistence.schemas import ForecastRecord


@dataclass(frozen=True, slots=True)
class ForecastCard:
    horizon_label: str
    headline: str
    body: str
    current_price: float
    lower_price_area: float
    upper_price_area: float
    lower_probability_pct: float
    upper_probability_pct: float
    neither_probability_pct: float
    sell_expected_value: float
    buy_expected_value: float
    confidence_pct: float
    model_status: str
    data_freshness: str
    model_version: str


def build_forecast_card(record: ForecastRecord) -> ForecastCard:
    forecast = record.forecast
    probabilities = forecast.probabilities
    return ForecastCard(
        horizon_label=HORIZON_SPECS[forecast.horizon].display_name,
        headline=record.explanation.headline,
        body=record.explanation.body,
        current_price=forecast.current_price,
        lower_price_area=forecast.lower_price_area,
        upper_price_area=forecast.upper_price_area,
        lower_probability_pct=round(100.0 * probabilities.lower_first, 10),
        upper_probability_pct=round(100.0 * probabilities.upper_first, 10),
        neither_probability_pct=round(100.0 * probabilities.neither, 10),
        sell_expected_value=forecast.sell_expected_value,
        buy_expected_value=forecast.buy_expected_value,
        confidence_pct=round(100.0 * forecast.confidence_score, 10),
        model_status=forecast.model_status.value,
        data_freshness=forecast.data_freshness.value,
        model_version=forecast.identity.model_version,
    )
