from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd

from ung_forecast.data.provider import YFinanceProvider


def make_yfinance_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [10.0, 10.1, 10.2],
            "High": [10.2, 10.3, 10.4],
            "Low": [9.9, 10.0, 10.1],
            "Close": [10.1, 10.2, 10.3],
            "Adj Close": [10.1, 10.2, 10.3],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.DatetimeIndex(
            [
                "2026-07-10 14:30:00+00:00",
                "2026-07-10 14:35:00+00:00",
                "2026-07-10 14:40:00+00:00",
            ]
        ),
    )


def test_yfinance_provider_returns_validated_bundle() -> None:
    provider = YFinanceProvider(adjusted_prices=False)
    with patch("ung_forecast.data.provider.yf.download", return_value=make_yfinance_frame()):
        bundle = provider.download(
            symbol="UNG",
            interval="5m",
            period="5d",
            as_of=datetime(2026, 7, 10, 14, 46, tzinfo=UTC),
        )

    assert bundle.provenance.provider == "yfinance"
    assert bundle.provenance.symbol == "UNG"
    assert bundle.provenance.adjusted_prices is False
    assert str(bundle.frame.index.tz) == "America/New_York"
    assert len(bundle.frame) == 3


def test_yfinance_provider_preserves_adjustment_policy() -> None:
    provider = YFinanceProvider(adjusted_prices=True)
    with patch("ung_forecast.data.provider.yf.download", return_value=make_yfinance_frame()) as call:
        bundle = provider.download(
            symbol="UNG",
            interval="5m",
            period="5d",
            as_of=datetime(2026, 7, 10, 14, 46, tzinfo=UTC),
        )

    assert call.call_args.kwargs["auto_adjust"] is True
    assert bundle.provenance.adjusted_prices is True
