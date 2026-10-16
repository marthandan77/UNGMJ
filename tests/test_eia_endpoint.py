from __future__ import annotations

import json

import pandas as pd
from urllib.parse import parse_qs, urlsplit

from ungmj import feeds


def test_eia_api_uses_spot_dataset_and_parses_daily_history(monkeypatch) -> None:
    sessions = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=5100)
    all_rows = [
        {"period": session.date().isoformat(), "value": str(2.5 + i / 1000)}
        for i, session in enumerate(sessions)
    ]
    requested: list[str] = []

    def fake_get(url: str, **kwargs: object) -> str:
        requested.append(url)
        offset = int(parse_qs(urlsplit(url).query)["offset"][0])
        return json.dumps({"response": {"data": all_rows[offset:offset + 5000]}})

    monkeypatch.setenv("EIA_API_KEY", "test-key")
    monkeypatch.setattr(feeds, "_get", fake_get)
    frame, snapshot, _ = feeds.fetch_eia_henry_hub_spot()

    assert requested
    assert "/natural-gas/pri/fut/data/" in requested[0]
    assert "facets%5Bseries%5D%5B%5D=RNGWHHD" in requested[0] or "facets[series][]=RNGWHHD" in requested[0]
    assert len(requested) == 2
    assert [parse_qs(urlsplit(url).query)["offset"][0] for url in requested] == ["0", "5000"]
    assert len(frame) == 5100
    assert frame.index.is_monotonic_increasing
    assert frame["henry_hub_spot"].gt(0).all()
    assert snapshot.provider == "eia_api"
