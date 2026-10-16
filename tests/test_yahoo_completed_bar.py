from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ungmj.contracts import SourceSnapshot
from ungmj.feeds import FetchedFrame, latest_session_agreement


def source(provider: str, index: pd.DatetimeIndex, closes: np.ndarray) -> FetchedFrame:
    frame = pd.DataFrame({"close": closes}, index=index)
    snapshot = SourceSnapshot(
        provider=provider,
        instrument="UNG",
        retrieved_at=datetime.now(timezone.utc),
        published_through=index[-1].date(),
        content_sha256="b" * 64,
        source_url="https://example.com/data",
    )
    return FetchedFrame(frame, snapshot, "fixture")


def test_independent_fresh_sources_need_long_history_and_agree() -> None:
    index = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=600)
    closes = np.linspace(10.0, 30.0, len(index))
    first = source("provider-a", index, closes)
    second = source("provider-b", index, closes * 1.001)
    ok, note = latest_session_agreement(first, second)
    assert ok
    assert "Confirmed session" in note


def test_same_provider_cannot_claim_independent_confirmation() -> None:
    index = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=600)
    closes = np.linspace(10.0, 30.0, len(index))
    ok, note = latest_session_agreement(source("same", index, closes), source("same", index, closes))
    assert not ok
    assert "not independent" in note


def test_stale_latest_session_is_blocked() -> None:
    index = pd.bdate_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=10), periods=600)
    closes = np.linspace(10.0, 30.0, len(index))
    ok, note = latest_session_agreement(source("provider-a", index, closes), source("provider-b", index, closes))
    assert not ok
    assert "stale" in note


def test_yahoo_daily_feed_excludes_current_utc_day(monkeypatch) -> None:
    import json
    from datetime import datetime, timedelta, timezone

    from ungmj import feeds

    today = datetime.now(timezone.utc).date()
    dates = [today - timedelta(days=2), today - timedelta(days=1), today]
    payload = {
        "chart": {
            "result": [{
                "timestamp": [int(pd.Timestamp(d, tz="UTC").timestamp()) for d in dates],
                "indicators": {
                    "quote": [{
                        "open": [10.0, 10.1, 10.2],
                        "high": [10.5, 10.6, 10.7],
                        "low": [9.5, 9.6, 9.7],
                        "close": [10.2, 10.3, 10.4],
                        "volume": [1000, 1100, 1200],
                    }]
                },
            }]
        }
    }
    monkeypatch.setattr(feeds, "_get", lambda *args, **kwargs: json.dumps(payload))
    result = feeds.fetch_yahoo_ung()
    assert result.frame.index[-1].date() < today
    assert result.snapshot.published_through == result.frame.index[-1].date()
