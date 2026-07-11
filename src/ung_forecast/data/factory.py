"""Construct the configured market-data provider without exposing secrets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ung_forecast.configuration import AppConfig

from .provider import MarketDataProvider, YFinanceProvider
from .schwab import SchwabCredentials, SchwabMarketDataProvider, SchwabTokenProvider


def _optional_string(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_market_data_provider(
    config: AppConfig,
    *,
    secrets: Mapping[str, Any] | None = None,
) -> MarketDataProvider:
    provider_name = config.data.provider.lower()
    if provider_name == "yfinance":
        return YFinanceProvider(
            timezone=config.data.timezone,
            adjusted_prices=config.data.adjusted_prices,
        )
    if provider_name != "schwab":
        raise ValueError(f"Unsupported market-data provider: {config.data.provider}")
    if secrets is None or "schwab" not in secrets:
        raise ValueError("Streamlit secrets must contain a [schwab] section")

    section = secrets["schwab"]
    if not isinstance(section, Mapping):
        raise ValueError("The [schwab] secrets section must be a mapping")
    credentials = SchwabCredentials(
        client_id=str(section.get("client_id", "")).strip(),
        client_secret=str(section.get("client_secret", "")).strip(),
        refresh_token=_optional_string(section, "refresh_token"),
        access_token=_optional_string(section, "access_token"),
        token_url=str(
            section.get("token_url", "https://api.schwabapi.com/v1/oauth/token")
        ).strip(),
    )
    token_provider = SchwabTokenProvider(credentials)
    base_url = str(
        section.get("market_data_base_url", "https://api.schwabapi.com/marketdata/v1")
    ).strip()
    return SchwabMarketDataProvider(
        token_provider,
        base_url=base_url,
        timezone=config.data.timezone,
    )
