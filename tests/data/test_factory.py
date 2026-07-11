from __future__ import annotations

import pytest

from ung_forecast.configuration import AppConfig, DataConfig
from ung_forecast.data.factory import build_market_data_provider
from ung_forecast.data.provider import YFinanceProvider
from ung_forecast.data.schwab import SchwabMarketDataProvider


def test_factory_builds_yfinance_without_secrets() -> None:
    config = AppConfig(data=DataConfig(provider="yfinance"))
    provider = build_market_data_provider(config)
    assert isinstance(provider, YFinanceProvider)


def test_factory_requires_schwab_secrets() -> None:
    config = AppConfig(data=DataConfig(provider="schwab"))
    with pytest.raises(ValueError, match="\[schwab\]"):
        build_market_data_provider(config, secrets={})


def test_factory_builds_schwab_with_access_token() -> None:
    config = AppConfig(data=DataConfig(provider="schwab"))
    provider = build_market_data_provider(
        config,
        secrets={
            "schwab": {
                "client_id": "client",
                "client_secret": "secret",
                "access_token": "token",
            }
        },
    )
    assert isinstance(provider, SchwabMarketDataProvider)
