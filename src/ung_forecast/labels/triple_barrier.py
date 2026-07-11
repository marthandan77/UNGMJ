"""Triple-barrier path labeling with explicit ambiguity handling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from ung_forecast.barriers import BarrierDefinition
from ung_forecast.schemas import OutcomeClass


class LabelStatus(StrEnum):
    VALID = "VALID"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_PATH = "INSUFFICIENT_PATH"


@dataclass(frozen=True, slots=True)
class LabelResult:
    outcome: OutcomeClass | None
    status: LabelStatus
    event_timestamp: pd.Timestamp | None
    time_to_event_bars: int | None
    maximum_favourable_excursion: float
    maximum_adverse_excursion: float


def label_future_path(
    future_bars: pd.DataFrame,
    *,
    barriers: BarrierDefinition,
    required_bars: int,
) -> LabelResult:
    """Label which barrier is reached first within a fixed future path.

    If a single OHLC bar touches both barriers, the within-bar ordering is
    unknowable from bar data. The sample is marked AMBIGUOUS and must be
    excluded from supervised training.
    """

    if required_bars <= 0:
        raise ValueError("required_bars must be positive")
    required = {"High", "Low"}
    missing = required.difference(future_bars.columns)
    if missing:
        raise ValueError(f"Missing future path columns: {sorted(missing)}")

    observed = future_bars.iloc[:required_bars]
    if len(observed) < required_bars:
        return LabelResult(
            outcome=None,
            status=LabelStatus.INSUFFICIENT_PATH,
            event_timestamp=None,
            time_to_event_bars=None,
            maximum_favourable_excursion=float("nan"),
            maximum_adverse_excursion=float("nan"),
        )

    maximum_upside = float(observed["High"].max() - barriers.current_price)
    maximum_downside = float(barriers.current_price - observed["Low"].min())

    for offset, (timestamp, row) in enumerate(observed.iterrows(), start=1):
        hit_upper = float(row["High"]) >= barriers.upper_price
        hit_lower = float(row["Low"]) <= barriers.lower_price
        if hit_upper and hit_lower:
            return LabelResult(
                outcome=None,
                status=LabelStatus.AMBIGUOUS,
                event_timestamp=pd.Timestamp(timestamp),
                time_to_event_bars=offset,
                maximum_favourable_excursion=maximum_downside,
                maximum_adverse_excursion=maximum_upside,
            )
        if hit_lower:
            return LabelResult(
                outcome=OutcomeClass.LOWER_FIRST,
                status=LabelStatus.VALID,
                event_timestamp=pd.Timestamp(timestamp),
                time_to_event_bars=offset,
                maximum_favourable_excursion=maximum_downside,
                maximum_adverse_excursion=maximum_upside,
            )
        if hit_upper:
            return LabelResult(
                outcome=OutcomeClass.UPPER_FIRST,
                status=LabelStatus.VALID,
                event_timestamp=pd.Timestamp(timestamp),
                time_to_event_bars=offset,
                maximum_favourable_excursion=maximum_upside,
                maximum_adverse_excursion=maximum_downside,
            )

    return LabelResult(
        outcome=OutcomeClass.NEITHER,
        status=LabelStatus.VALID,
        event_timestamp=None,
        time_to_event_bars=required_bars,
        maximum_favourable_excursion=maximum_upside,
        maximum_adverse_excursion=maximum_downside,
    )
