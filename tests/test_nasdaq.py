from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from ungmj import nasdaq


def test_nasdaq_fetches_full_completed_ohlcv_history(monkeypatch) -> None:
    today = datetime.now(timezone.utc).date()
    sessions = pd.bdate_range(end=pd.Timestamp(today) - pd.Timedelta(days=1), periods=600)
    rows = [
        {
            "date": session.strftime("%m/%d/%Y"),
            "open": "$10.00",
            "high": "$11.00",
            "low": "$9.00",
            "close": "$10.50",
            "volume": "1,000",
        }
        for session in sessions
    ]
    payload = {"data": {"tradesTable": {"rows": rows}}}
    requested: list[str] = []

    def fake_get(url: str, **kwargs: object) -> str:
        requested.append(url)
        return json.dumps(payload)

    monkeypatch.setattr(nasdaq, "_get", fake_get)
    result = nasdaq.fetch_nasdaq_ung()

    assert len(result.frame) == 600
    assert list(result.frame.columns) == ["open", "high", "low", "close", "volume"]
    assert result.frame.index[-1].date() < today
    query = parse_qs(urlsplit(requested[0]).query)
    assert query["fromdate"] == ["2007-04-18"]
    assert query["limit"] == ["5000"]
    assert result.snapshot.provider == "nasdaq"
