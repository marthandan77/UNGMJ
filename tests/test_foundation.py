from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from ung_forecast.configuration import AppConfig, DataConfig
from ung_forecast.horizons import HORIZON_SPECS, HorizonKey
from ung_forecast.schemas import (
    DataFreshness,
    HorizonForecast,
    ModelIdentity,
    ModelStatus,
    ProbabilityForecast,
)


def test_all_locked_horizons_are_defined() -> None:
    assert set(HORIZON_SPECS) == {
        HorizonKey.MINUTES_60,
        HorizonKey.HOURS_4,
        HorizonKey.DAY_1,
        HorizonKey.DAYS_2,
        HorizonKey.DAYS_7,
    }


def test_intraday_horizon_definitions_are_exact() -> None:
    sixty = HORIZON_SPECS[HorizonKey.MINUTES_60]
    four_hours = HORIZON_SPECS[HorizonKey.HOURS_4]

    assert sixty.source_interval == "5m"
    assert sixty.bars_ahead == 12
    assert sixty.complete_sessions_ahead is None

    assert four_hours.source_interval == "15m"
    assert four_hours.bars_ahead == 16
    assert four_hours.complete_sessions_ahead is None


def test_session_horizon_definitions_are_exact() -> None:
    assert HORIZON_SPECS[HorizonKey.DAY_1].complete_sessions_ahead == 1
    assert HORIZON_SPECS[HorizonKey.DAYS_2].complete_sessions_ahead == 2
    assert HORIZON_SPECS[HorizonKey.DAYS_7].complete_sessions_ahead == 7


def test_probability_contract_accepts_valid_distribution() -> None:
    result = ProbabilityForecast(lower_first=0.3, upper_first=0.5, neither=0.2)
    assert result.lower_first + result.upper_first + result.neither == pytest.approx(1.0)


def test_probability_contract_rejects_invalid_distribution() -> None:
    with pytest.raises(ValidationError):
        ProbabilityForecast(lower_first=0.4, upper_first=0.4, neither=0.4)


def test_horizon_forecast_is_immutable_and_ordered() -> None:
    forecast = HorizonForecast(
        timestamp=datetime(2026, 7, 11, 15, 35, tzinfo=UTC),
        horizon=HorizonKey.MINUTES_60,
        current_price=10.8,
        lower_price_area=10.6,
        upper_price_area=11.0,
        probabilities=ProbabilityForecast(lower_first=0.4, upper_first=0.45, neither=0.15),
        sell_expected_value=-0.02,
        buy_expected_value=0.01,
        sell_ev_lower_confidence_bound=-0.04,
        buy_ev_lower_confidence_bound=-0.01,
        confidence_score=0.5,
        model_status=ModelStatus.RESEARCH_ONLY,
        data_freshness=DataFreshness.CURRENT,
        identity=ModelIdentity(
            data_version="data-v1",
            feature_version="features-v1",
            model_version="model-v1",
            configuration_hash="12345678abcdef",
        ),
    )

    assert forecast.model_status is ModelStatus.RESEARCH_ONLY
    with pytest.raises(ValidationError):
        forecast.current_price = 11.0


def test_configuration_hash_is_stable() -> None:
    first = AppConfig()
    second = AppConfig()
    assert first.configuration_hash == second.configuration_hash
    assert len(first.configuration_hash) == 64
    assert first.quantitative_configuration_hash == second.quantitative_configuration_hash


def test_quantitative_hash_ignores_provider_and_cache_path() -> None:
    research = AppConfig(
        data=DataConfig(provider="yfinance", cache_directory=Path("research/cache"))
    )
    runtime = AppConfig(
        data=DataConfig(provider="schwab", cache_directory=Path("data/cache"))
    )
    assert research.configuration_hash != runtime.configuration_hash
    assert research.quantitative_configuration_hash == runtime.quantitative_configuration_hash


def test_quantitative_hash_changes_when_model_input_policy_changes() -> None:
    unadjusted = AppConfig(data=DataConfig(adjusted_prices=False))
    adjusted = AppConfig(data=DataConfig(adjusted_prices=True))
    assert unadjusted.quantitative_configuration_hash != adjusted.quantitative_configuration_hash
