"""Point-in-time features from Alpaca's UNG one-minute SIP bars."""
from __future__ import annotations

from datetime import datetime, time
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .contracts import DataValidationError


ET = ZoneInfo("America/New_York")
OUTSIDE_RTH_COLUMNS = (
    "ung_after_hours_return", "ung_after_hours_range_pct", "ung_after_hours_volume_ratio",
    "ung_premarket_mid_return", "ung_premarket_mid_range_pct", "ung_premarket_mid_volume_ratio",
    "ung_premarket_1h_return", "ung_premarket_1h_range_pct", "ung_premarket_1h_volume_ratio",
)


def build_current_outside_rth_snapshot(
    intraday: pd.DataFrame, daily_ung: pd.DataFrame, *, now: datetime | None = None,
) -> dict[str, Any]:
    """Build the 6:45 or 8:30 a.m. ET snapshot from the delayed data fetch.

    The output is indexed to the previous completed UNG RTH session because the
    normal forecast targets the current session as horizon one. Alpaca supplies
    consolidated SIP OHLCV bars, not quote spreads or order-book depth here.
    """
    now_et = (now or datetime.now(ET)).astimezone(ET)
    if time(6, 55) <= now_et.time() <= time(7, 15):
        snapshot_slot, cutoff_time = "mid", time(6, 45)
    elif time(8, 40) <= now_et.time() <= time(9, 0):
        snapshot_slot, cutoff_time = "one_hour", time(8, 30)
    else:
        raise DataValidationError("outside-RTH data is fetched at 07:00 or 08:45 ET")
    if intraday.empty or not isinstance(intraday.index, pd.DatetimeIndex) or intraday.index.tz is None:
        raise DataValidationError("Alpaca intraday frame must have timezone-aware timestamps")

    local_index = intraday.index.tz_convert(ET)
    local = intraday.copy()
    local.index = local_index
    target_date = now_et.date()
    cutoff = pd.Timestamp(datetime.combine(target_date, cutoff_time, tzinfo=ET))
    current_pre = local.loc[
        (local.index.date == target_date)
        & (local.index.time >= time(4, 0))
        & (local.index < cutoff)
    ]
    if current_pre.empty:
        raise DataValidationError(f"no UNG premarket bars available by {cutoff_time.strftime('%H:%M')} ET for {target_date}")
    last_bar_end = current_pre.index[-1] + pd.Timedelta(minutes=5)
    if cutoff - last_bar_end > pd.Timedelta(minutes=15):
        raise DataValidationError("latest UNG premarket bar is more than 15 minutes stale")

    daily = daily_ung.copy()
    if daily.index.tz is not None:
        daily.index = daily.index.tz_convert(ET).tz_localize(None)
    daily.index = daily.index.normalize()
    daily = daily.sort_index()
    prior_sessions = daily.index[daily.index.date < target_date]
    if len(prior_sessions) == 0:
        raise DataValidationError("no completed UNG RTH close precedes the current premarket")
    origin = prior_sessions[-1]
    reference_close = float(daily.loc[origin, "close"])
    if reference_close <= 0:
        raise DataValidationError("previous RTH close is invalid")

    def session_bars(session_date, start: time, end: time) -> pd.DataFrame:
        return local.loc[
            (local.index.date == session_date)
            & (local.index.time >= start)
            & (local.index.time < end)
        ]

    previous_post = session_bars(origin.date(), time(16, 0), time(20, 0))
    previous_pre_sessions: list[float] = []
    prior_dates = sorted({stamp.date() for stamp in local.index if stamp.date() < target_date})
    for session_date in prior_dates[-5:]:
        bars = session_bars(session_date, time(4, 0), cutoff_time)
        if not bars.empty:
            previous_pre_sessions.append(float(bars["volume"].sum()))
    post_volumes = []
    for session_date in [day for day in prior_dates if day < origin.date()][-5:]:
        bars = session_bars(session_date, time(16, 0), time(20, 0))
        if not bars.empty:
            post_volumes.append(float(bars["volume"].sum()))

    current_pre_volume = float(current_pre["volume"].sum())
    prior_pre_volume = median(previous_pre_sessions) if previous_pre_sessions else float("nan")
    current_post_volume = float(previous_post["volume"].sum()) if not previous_post.empty else float("nan")
    prior_post = [value for value in post_volumes if value > 0]
    prior_post_volume = median(prior_post) if prior_post else float("nan")
    last_pre_price = float(current_pre["close"].iloc[-1])
    slot_prefix = "mid" if snapshot_slot == "mid" else "1h"
    values = {
        "origin_session": origin.date().isoformat(),
        "target_session": target_date.isoformat(),
        "snapshot_slot": snapshot_slot,
        "available_at": min(last_bar_end, cutoff).isoformat(),
        "source": "alpaca_sip_1m_extended",
        f"ung_premarket_{slot_prefix}_last_price": last_pre_price,
        "ung_after_hours_last_price": float(previous_post["close"].iloc[-1]) if not previous_post.empty else None,
        f"ung_premarket_{slot_prefix}_return": last_pre_price / reference_close - 1.0,
        f"ung_premarket_{slot_prefix}_range_pct": (float(current_pre["high"].max()) - float(current_pre["low"].min())) / reference_close,
        f"ung_premarket_{slot_prefix}_volume_ratio": current_pre_volume / prior_pre_volume if prior_pre_volume > 0 else None,
        "ung_after_hours_return": float(previous_post["close"].iloc[-1]) / reference_close - 1.0 if not previous_post.empty else None,
        "ung_after_hours_range_pct": (float(previous_post["high"].max()) - float(previous_post["low"].min())) / reference_close if not previous_post.empty else None,
        "ung_after_hours_volume_ratio": current_post_volume / prior_post_volume if not previous_post.empty and prior_post_volume > 0 else None,
    }
    return values


def outside_rth_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert persisted snapshots into numeric features indexed by origin session."""
    if not records:
        return pd.DataFrame(columns=list(OUTSIDE_RTH_COLUMNS), index=pd.DatetimeIndex([], name="session"))
    merged: dict[pd.Timestamp, dict[str, Any]] = {}
    for record in sorted(records, key=lambda row: str(row.get("available_at", ""))):
        if record.get("source") != "alpaca_sip_1m_extended":
            continue
        origin = pd.Timestamp(record["origin_session"]).normalize()
        row = merged.setdefault(origin, {})
        for column in OUTSIDE_RTH_COLUMNS:
            if record.get(column) is not None:
                row[column] = record[column]
    frame = pd.DataFrame.from_dict(merged, orient="index")
    frame.index = pd.DatetimeIndex(frame.index, name="session")
    for column in OUTSIDE_RTH_COLUMNS:
        if column not in frame:
            frame[column] = float("nan")
    return frame[list(OUTSIDE_RTH_COLUMNS)].apply(pd.to_numeric, errors="coerce").sort_index()
