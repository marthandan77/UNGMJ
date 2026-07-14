"""Score stored forecasts only after their complete horizon has expired."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from ung_forecast.barriers import BarrierDefinition
from ung_forecast.labels import LabelStatus, label_future_path
from ung_forecast.persistence.schemas import (
    ForecastRecord,
    ForecastRecordStatus,
    ScoredOutcome,
)


def score_forecast_record(
    record: ForecastRecord,
    future_bars: pd.DataFrame,
    *,
    required_bars: int,
    scored_at: datetime,
) -> ForecastRecord:
    if scored_at.tzinfo is None:
        raise ValueError("scored_at must be timezone-aware")
    if scored_at < record.expires_at:
        raise ValueError("Forecast horizon has not expired")
    if record.status is not ForecastRecordStatus.PENDING:
        raise ValueError("Only pending forecasts may be scored")

    forecast = record.forecast
    volatility_proxy = max(
        forecast.current_price - forecast.lower_price_area,
        forecast.upper_price_area - forecast.current_price,
    )
    barriers = BarrierDefinition(
        current_price=forecast.current_price,
        lower_price=forecast.lower_price_area,
        upper_price=forecast.upper_price_area,
        horizon_key=forecast.horizon.value,
        volatility_estimate=volatility_proxy,
        lower_multiplier=1.0,
        upper_multiplier=1.0,
    )
    result = label_future_path(future_bars, barriers=barriers, required_bars=required_bars)
    if result.status is LabelStatus.INSUFFICIENT_PATH:
        raise ValueError("Insufficient completed future bars to score forecast")
    if result.status is LabelStatus.AMBIGUOUS or result.outcome is None:
        return record.model_copy(update={"status": ForecastRecordStatus.INVALID})

    outcome = ScoredOutcome(
        outcome=result.outcome,
        scored_at=scored_at,
        event_timestamp=(
            result.event_timestamp.to_pydatetime() if result.event_timestamp is not None else None
        ),
        time_to_event_bars=result.time_to_event_bars or required_bars,
        maximum_upward_excursion=result.maximum_upward_excursion,
        maximum_downward_excursion=result.maximum_downward_excursion,
    )
    return record.model_copy(
        update={"status": ForecastRecordStatus.SCORED, "outcome": outcome}
    )
