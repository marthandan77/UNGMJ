from __future__ import annotations

import pandas as pd

from ung_forecast.validation.purge import purge_overlapping_training_rows


def test_rows_with_labels_reaching_evaluation_are_purged() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    label_end = pd.Series(
        [index[1], index[2], index[4], index[4], index[5], index[5]],
        index=index,
    )

    kept = purge_overlapping_training_rows(
        index[:4],
        label_end,
        evaluation_start=index[4],
    )

    assert list(kept) == [index[0], index[1]]


def test_missing_label_end_is_rejected() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    label_end = pd.Series([index[1]], index=[index[0]])

    try:
        purge_overlapping_training_rows(index, label_end, evaluation_start=index[-1])
    except ValueError as exc:
        assert "Missing label-end" in str(exc)
    else:
        raise AssertionError("Expected missing label-end failure")
