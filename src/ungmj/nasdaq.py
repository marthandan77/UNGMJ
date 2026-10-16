"""Nasdaq public historical close adapter used only for independent source confirmation."""
from __future__ import annotations

import json
import pandas as pd

from .contracts import DataValidationError
from .feeds import FetchedFrame, _get, _snapshot


def fetch_nasdaq_ung() -> FetchedFrame:
    """Fetch Nasdaq's historical closes; primary Yahoo remains the OHLCV modeling feed."""
    url = "https://api.nasdaq.com/api/quote/UNG/historical?assetclass=etf&limit=5000"
    raw = _get(url, headers={
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
    })
    try:
        payload = json.loads(raw)
        rows = payload["data"]["tradesTable"]["rows"]
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError("empty history")
        aliases = {
            "".join(ch for ch in str(column).lower() if ch.isalnum()): column
            for column in frame.columns
        }
        date_col = aliases.get("date")
        close_col = aliases.get("close") or aliases.get("closelast") or aliases.get("lastsaleprice")
        if date_col is None or close_col is None:
            raise ValueError("date/close fields not present")
        selected = frame[[date_col, close_col]].copy()
        selected.columns = ["session", "close"]
        selected["session"] = pd.to_datetime(selected["session"], errors="raise").dt.normalize()
        selected["close"] = pd.to_numeric(
            selected["close"].astype(str).str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False).str.strip(), errors="raise"
        )
        selected = selected.set_index("session").sort_index()
        if selected.index.has_duplicates or selected["close"].isna().any() or (selected["close"] <= 0).any():
            raise ValueError("duplicate, missing, or non-positive close values")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise DataValidationError(f"Nasdaq history response schema/coverage invalid: {exc}") from exc
    if len(selected) < 500:
        raise DataValidationError(f"Nasdaq history unexpectedly short: {len(selected)} rows")
    return FetchedFrame(
        selected, _snapshot("nasdaq", "UNG", url, raw, selected.index[-1].date()), raw
    )
