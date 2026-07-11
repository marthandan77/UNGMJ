"""Canonical forecast-horizon definitions.

These definitions are immutable contracts shared by training, validation,
forecasting, and the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class HorizonKey(StrEnum):
    MINUTES_60 = "60m"
    HOURS_4 = "4h"
    DAY_1 = "1d"
    DAYS_2 = "2d"
    DAYS_7 = "7d"


@dataclass(frozen=True, slots=True)
class HorizonSpec:
    key: HorizonKey
    source_interval: str
    bars_ahead: int | None
    complete_sessions_ahead: int | None
    display_name: str

    def __post_init__(self) -> None:
        uses_bars = self.bars_ahead is not None
        uses_sessions = self.complete_sessions_ahead is not None
        if uses_bars == uses_sessions:
            raise ValueError("A horizon must use exactly one window definition")
        if self.bars_ahead is not None and self.bars_ahead <= 0:
            raise ValueError("bars_ahead must be positive")
        if self.complete_sessions_ahead is not None and self.complete_sessions_ahead <= 0:
            raise ValueError("complete_sessions_ahead must be positive")


HORIZON_SPECS: dict[HorizonKey, HorizonSpec] = {
    HorizonKey.MINUTES_60: HorizonSpec(
        key=HorizonKey.MINUTES_60,
        source_interval="5m",
        bars_ahead=12,
        complete_sessions_ahead=None,
        display_name="Next 60 minutes",
    ),
    HorizonKey.HOURS_4: HorizonSpec(
        key=HorizonKey.HOURS_4,
        source_interval="15m",
        bars_ahead=16,
        complete_sessions_ahead=None,
        display_name="Next 4 hours",
    ),
    HorizonKey.DAY_1: HorizonSpec(
        key=HorizonKey.DAY_1,
        source_interval="1h",
        bars_ahead=None,
        complete_sessions_ahead=1,
        display_name="Next complete trading day",
    ),
    HorizonKey.DAYS_2: HorizonSpec(
        key=HorizonKey.DAYS_2,
        source_interval="1h",
        bars_ahead=None,
        complete_sessions_ahead=2,
        display_name="Next 2 trading days",
    ),
    HorizonKey.DAYS_7: HorizonSpec(
        key=HorizonKey.DAYS_7,
        source_interval="1d",
        bars_ahead=None,
        complete_sessions_ahead=7,
        display_name="Next 7 trading days",
    ),
}
