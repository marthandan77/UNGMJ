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
