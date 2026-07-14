"""Assemble audited forecast records from validated quantitative outputs."""

from __future__ import annotations

from datetime import datetime

from ung_forecast.explanations.renderer import UserIntent, render_explanation
from ung_forecast.persistence.schemas import ForecastRecord
from ung_forecast.schemas import HorizonForecast


def assemble_forecast_record(
    forecast: HorizonForecast,
    *,
    intent: UserIntent,
    expires_at: datetime,
) -> ForecastRecord:
    if expires_at.tzinfo is None:
        raise ValueError("expires_at must be timezone-aware")
    explanation = render_explanation(forecast, intent)
    return ForecastRecord(
        created_at=forecast.timestamp,
        expires_at=expires_at,
        forecast=forecast,
        user_intent=intent,
        explanation=explanation,
    )
