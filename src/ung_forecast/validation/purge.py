"""Remove training observations whose label windows overlap evaluation periods."""

from __future__ import annotations

import pandas as pd


def purge_overlapping_training_rows(
    training_index: pd.DatetimeIndex,
    label_end_time: pd.Series,
    *,
    evaluation_start: pd.Timestamp,
    embargo_end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    if not training_index.is_monotonic_increasing or training_index.has_duplicates:
        raise ValueError("training_index must be unique and increasing")
    aligned = label_end_time.reindex(training_index)
    if aligned.isna().any():
        raise ValueError("Missing label-end timestamps for training rows")
    keep = aligned < evaluation_start
    if embargo_end is not None:
        keep &= training_index > embargo_end
    return training_index[keep.to_numpy()]
