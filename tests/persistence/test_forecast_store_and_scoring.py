from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from ung_forecast.explanations.renderer import Explanation, UserIntent
from ung_forecast.horizons import HorizonKey
from ung_forecast.outcomes import score_forecast_record
from ung_forecast.persistence import ForecastRecord, ForecastRecordStatus, JsonlForecastStore
from ung_forecast.schemas import (
    DataFreshness,
    HorizonForecast,
    ModelIdentity,
    ModelStatus,
    OutcomeClass,
    ProbabilityForecast,
)


def record() -> ForecastRecord:
    created = datetime(2026, 7, 10, 14, 30, tzinfo=UTC)
    forecast = HorizonForecast(
        timestamp=created,
        horizon=HorizonKey.MINUTES_60,
        current_price=10.0,
        lower_price_area=9.9,
        upper_price_area=10.1,
        probabilities=ProbabilityForecast(lower_first=0.5, upper_first=0.3, neither=0.2),
        sell_expected_value=0.02,
        buy_expected_value=-0.01,
        sell_ev_lower_confidence_bound=0.005,
        buy_ev_lower_confidence_bound=-0.03,
        confidence_score=0.6,
        model_status=ModelStatus.VALIDATED,
        data_freshness=DataFreshness.CURRENT,
        identity=ModelIdentity(
            data_version="data-v1",
            feature_version="features-v1",
            model_version="model-v1",
            configuration_hash="12345678abcdef",
        ),
    )
    return ForecastRecord(
        created_at=created,
        expires_at=created + timedelta(minutes=60),
        forecast=forecast,
        user_intent=UserIntent.SELLING,
        explanation=Explanation("SELLING NOW LOOKS REASONABLE", "Evidence-backed.", ("ev",)),
    )


def future_bars() -> pd.DataFrame:
    index = pd.date_range("2026-07-10 14:35", periods=12, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "High": [10.02, 10.04, 10.05, 10.06, 10.07, 10.08, 10.08, 10.09, 10.08, 10.07, 10.06, 10.05],
            "Low": [9.98, 9.97, 9.96, 9.94, 9.89, 9.91, 9.92, 9.93, 9.94, 9.95, 9.96, 9.97],
        },
        index=index,
    )


def test_jsonl_store_is_append_only_and_latest_record_wins(tmp_path) -> None:
    store = JsonlForecastStore(tmp_path / "forecasts.jsonl")
    pending = record()
    store.append(pending)
    scored = score_forecast_record(
        pending,
        future_bars(),
        required_bars=12,
        scored_at=pending.expires_at,
    )
    store.append(scored)

    assert len(store.read_all()) == 2
    latest = store.latest_by_id()[pending.record_id]
    assert latest.status is ForecastRecordStatus.SCORED
    assert latest.outcome is not None
    assert latest.outcome.outcome is OutcomeClass.LOWER_FIRST


def test_scoring_before_expiry_is_rejected() -> None:
    pending = record()
    with pytest.raises(ValueError, match="has not expired"):
        score_forecast_record(
            pending,
            future_bars(),
            required_bars=12,
            scored_at=pending.expires_at - timedelta(seconds=1),
        )


def test_ambiguous_path_is_marked_invalid() -> None:
    pending = record()
    bars = future_bars()
    bars.iloc[0, bars.columns.get_loc("High")] = 10.11
    bars.iloc[0, bars.columns.get_loc("Low")] = 9.89
    result = score_forecast_record(
        pending,
        bars,
        required_bars=12,
        scored_at=pending.expires_at,
    )
    assert result.status is ForecastRecordStatus.INVALID
    assert result.outcome is None
