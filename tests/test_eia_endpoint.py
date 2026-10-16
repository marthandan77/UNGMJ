from __future__ import annotations

import json

import pandas as pd

from ungmj import feeds


def test_eia_api_uses_spot_dataset_and_parses_daily_history(monkeypatch) -> None:
    sessions = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=1100)
    payload = {
        "response": {
            "data": [
                {"period": session.date().isoformat(), "value": str(2.5 + i / 1000)}
                for i, session in enumerate(sessions)
            ]
        }
    }
    requested: list[str] = []

    def fake_get(url: str, **kwargs: object) -> str:
        requested.append(url)
        return json.dumps(payload)

    monkeypatch.setenv("EIA_API_KEY", "test-key")
    monkeypatch.setattr(feeds, "_get", fake_get)
    frame, snapshot, _ = feeds.fetch_eia_henry_hub_spot()

    assert requested
    assert "/natural-gas/pri/spt/data/" in requested[0]
    assert "/natural-gas/pri/fut/data/" not in requested[0]
    assert len(frame) == 1100
    assert frame.index.is_monotonic_increasing
    assert frame["henry_hub_spot"].gt(0).all()
    assert snapshot.provider == "eia_api"
