from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ung_forecast.data.twelvedata import (
    TwelveDataCredentials,
    TwelveDataMarketDataProvider,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeHttpClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        timeout: float,
    ) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(self.payload)


def valid_payload() -> dict[str, Any]:
    return {
        "meta": {
            "symbol": "UNG",
            "interval": "5min",
            "timezone": "America/New_York",
        },
        "values": [
            {
                "datetime": "2026-07-10 10:40:00",
                "open": "10.05",
                "high": "10.12",
                "low": "10.02",
                "close": "10.10",
                "volume": "1200",
            },
            {
                "datetime": "2026-07-10 10:35:00",
                "open": "10.00",
                "high": "10.08",
                "low": "9.98",
                "close": "10.05",
                "volume": "1000",
            },
        ],
        "status": "ok",
    }


def test_twelve_data_normalizes_descending_string_candles() -> None:
    client = FakeHttpClient(valid_payload())
    provider = TwelveDataMarketDataProvider(
        TwelveDataCredentials("secret-key"),
        http_client=client,
    )

    bundle = provider.download(
        symbol="UNG",
        interval="5m",
        as_of=datetime(2026, 7, 10, 14, 50, tzinfo=UTC),
    )

    assert bundle.provenance.provider == "twelvedata"
    assert bundle.provenance.symbol == "UNG"
    assert bundle.frame.index.is_monotonic_increasing
    assert str(bundle.frame.index.tz) == "America/New_York"
    assert list(bundle.frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert bundle.frame.iloc[-1]["Close"] == pytest.approx(10.10)
    assert client.calls[0]["params"]["interval"] == "5min"
    assert client.calls[0]["params"]["apikey"] == "secret-key"
    assert client.calls[0]["params"]["timezone"] == "America/New_York"


def test_twelve_data_surfaces_provider_error_payload() -> None:
    client = FakeHttpClient(
        {
            "status": "error",
            "code": 429,
            "message": "API credits exhausted",
        }
    )
    provider = TwelveDataMarketDataProvider(
        TwelveDataCredentials("secret-key"),
        http_client=client,
    )

    with pytest.raises(RuntimeError, match="429.*credits exhausted"):
        provider.download(
            symbol="UNG",
            interval="5m",
            as_of=datetime(2026, 7, 10, 14, 50, tzinfo=UTC),
        )


def test_twelve_data_rejects_missing_volume() -> None:
    payload = valid_payload()
    for row in payload["values"]:
        row.pop("volume")
    provider = TwelveDataMarketDataProvider(
        TwelveDataCredentials("secret-key"),
        http_client=FakeHttpClient(payload),
    )

    with pytest.raises(ValueError, match="missing fields.*volume"):
        provider.download(
            symbol="UNG",
            interval="5m",
            as_of=datetime(2026, 7, 10, 14, 50, tzinfo=UTC),
        )


def test_twelve_data_rejects_unsupported_interval() -> None:
    provider = TwelveDataMarketDataProvider(
        TwelveDataCredentials("secret-key"),
        http_client=FakeHttpClient(valid_payload()),
    )

    with pytest.raises(ValueError, match="not supported"):
        provider.download(
            symbol="UNG",
            interval="1h",
            as_of=datetime(2026, 7, 10, 14, 50, tzinfo=UTC),
        )


def test_twelve_data_requires_api_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        TwelveDataCredentials("   ")
