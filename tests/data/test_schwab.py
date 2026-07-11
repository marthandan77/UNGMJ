from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ung_forecast.data.schwab import (
    SchwabCredentials,
    SchwabMarketDataProvider,
    SchwabTokenProvider,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeHttpClient:
    def __init__(self) -> None:
        self.get_calls: list[dict[str, Any]] = []
        self.post_calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any],
        timeout: float,
    ) -> FakeResponse:
        self.get_calls.append(
            {"url": url, "headers": headers, "params": params, "timeout": timeout}
        )
        return FakeResponse(
            {
                "candles": [
                    {
                        "open": 10.00,
                        "high": 10.08,
                        "low": 9.98,
                        "close": 10.05,
                        "volume": 1000,
                        "datetime": 1783693800000,
                    },
                    {
                        "open": 10.05,
                        "high": 10.12,
                        "low": 10.02,
                        "close": 10.10,
                        "volume": 1200,
                        "datetime": 1783694100000,
                    },
                ]
            }
        )

    def post(
        self,
        url: str,
        *,
        data: dict[str, str],
        auth: tuple[str, str],
        timeout: float,
    ) -> FakeResponse:
        self.post_calls.append(
            {"url": url, "data": data, "auth": auth, "timeout": timeout}
        )
        return FakeResponse({"access_token": "refreshed-token", "expires_in": 1800})


def test_schwab_token_provider_uses_static_access_token() -> None:
    client = FakeHttpClient()
    provider = SchwabTokenProvider(
        SchwabCredentials("client", "secret", access_token="static-token"),
        http_client=client,
    )
    assert provider.get_access_token(now=datetime(2026, 7, 10, tzinfo=UTC)) == "static-token"
    assert client.post_calls == []


def test_schwab_token_provider_refreshes_without_exposing_secret() -> None:
    client = FakeHttpClient()
    provider = SchwabTokenProvider(
        SchwabCredentials("client", "secret", refresh_token="refresh-token"),
        http_client=client,
    )
    token = provider.get_access_token(now=datetime(2026, 7, 10, tzinfo=UTC))
    assert token == "refreshed-token"
    assert client.post_calls[0]["auth"] == ("client", "secret")
    assert client.post_calls[0]["data"]["grant_type"] == "refresh_token"


def test_schwab_market_data_adapter_normalizes_candles() -> None:
    client = FakeHttpClient()
    token_provider = SchwabTokenProvider(
        SchwabCredentials("client", "secret", access_token="token"),
        http_client=client,
    )
    provider = SchwabMarketDataProvider(token_provider, http_client=client)
    bundle = provider.download(
        symbol="UNG",
        interval="5m",
        as_of=datetime(2026, 7, 10, 14, 45, tzinfo=UTC),
    )

    assert bundle.provenance.provider == "schwab"
    assert bundle.provenance.symbol == "UNG"
    assert list(bundle.frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert str(bundle.frame.index.tz) == "America/New_York"
    assert client.get_calls[0]["params"]["frequency"] == 5
    assert client.get_calls[0]["headers"]["Authorization"] == "Bearer token"


def test_schwab_rejects_unsupported_interval() -> None:
    token_provider = SchwabTokenProvider(
        SchwabCredentials("client", "secret", access_token="token")
    )
    provider = SchwabMarketDataProvider(token_provider)
    with pytest.raises(ValueError, match="not supported"):
        provider.download(
            symbol="UNG",
            interval="1h",
            as_of=datetime(2026, 7, 10, 14, 45, tzinfo=UTC),
        )
