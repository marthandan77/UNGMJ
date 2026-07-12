from __future__ import annotations

import pandas as pd
import pytest

from ung_forecast.training import barrier_selection
from ung_forecast.training.barrier_selection import (
    BarrierCandidate,
    BarrierSelectionConfig,
    select_barriers_fold_only,
)
from ung_forecast.training.dataset import TrainingDataset


def _dataset(
    index: pd.DatetimeIndex,
    targets: list[str],
    *,
    label_end: pd.DatetimeIndex | None = None,
) -> TrainingDataset:
    features = pd.DataFrame({"x": range(len(index))}, index=index, dtype=float)
    target = pd.Series(targets, index=index, name="target")
    ends = label_end if label_end is not None else index + pd.Timedelta(minutes=1)
    label_end_time = pd.Series(ends, index=index, name="label_end_time")
    metadata = pd.DataFrame(index=index)
    return TrainingDataset(features, target, label_end_time, metadata)


def test_selects_lowest_validation_brier_without_test_index(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2025-01-01", periods=12, freq="5min", tz="UTC")
    training_index = index[:6]
    validation_index = index[6:9]
    candidate_a = BarrierCandidate(0.8, 0.8)
    candidate_b = BarrierCandidate(1.2, 1.2)

    datasets = {
        candidate_a: _dataset(
            index[:9],
            [
                "LOWER_FIRST",
                "UPPER_FIRST",
                "NEITHER",
                "LOWER_FIRST",
                "UPPER_FIRST",
                "NEITHER",
                "LOWER_FIRST",
                "UPPER_FIRST",
                "NEITHER",
            ],
        ),
        candidate_b: _dataset(
            index[:9],
            [
                "LOWER_FIRST",
                "LOWER_FIRST",
                "LOWER_FIRST",
                "UPPER_FIRST",
                "NEITHER",
                "NEITHER",
                "UPPER_FIRST",
                "UPPER_FIRST",
                "NEITHER",
            ],
        ),
    }

    def fake_dataset(*args: object, candidate: BarrierCandidate, **kwargs: object) -> TrainingDataset:
        del args, kwargs
        return datasets[candidate]

    monkeypatch.setattr(barrier_selection, "_candidate_dataset", fake_dataset)
    result = select_barriers_fold_only(
        pd.DataFrame(index=index),
        pd.DataFrame(index=index),
        pd.Series(1.0, index=index),
        training_index=training_index,
        validation_index=validation_index,
        validation_end=validation_index[-1] + pd.Timedelta(minutes=5),
        required_bars=12,
        horizon_key="60m",
        config=BarrierSelectionConfig(
            candidates=(candidate_a, candidate_b),
            minimum_training_rows=6,
            minimum_validation_rows=3,
        ),
    )
    assert result.selected == candidate_a
    assert all(item.validation_rows == 3 for item in result.candidates)


def test_purges_training_labels_crossing_validation_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
    training_index = index[:6]
    validation_index = index[6:9]
    candidate = BarrierCandidate(1.0, 1.0)
    ends = pd.DatetimeIndex(
        [
            index[0] + pd.Timedelta(minutes=1),
            index[1] + pd.Timedelta(minutes=1),
            index[2] + pd.Timedelta(minutes=1),
            index[3] + pd.Timedelta(minutes=1),
            validation_index[0],
            validation_index[0] + pd.Timedelta(minutes=5),
            index[6] + pd.Timedelta(minutes=1),
            index[7] + pd.Timedelta(minutes=1),
            index[8] + pd.Timedelta(minutes=1),
        ]
    )
    dataset = _dataset(
        index[:9],
        [
            "LOWER_FIRST",
            "UPPER_FIRST",
            "NEITHER",
            "LOWER_FIRST",
            "UPPER_FIRST",
            "NEITHER",
            "LOWER_FIRST",
            "UPPER_FIRST",
            "NEITHER",
        ],
        label_end=ends,
    )
    monkeypatch.setattr(barrier_selection, "_candidate_dataset", lambda *args, **kwargs: dataset)

    result = select_barriers_fold_only(
        pd.DataFrame(index=index),
        pd.DataFrame(index=index),
        pd.Series(1.0, index=index),
        training_index=training_index,
        validation_index=validation_index,
        validation_end=index[9],
        required_bars=12,
        horizon_key="60m",
        config=BarrierSelectionConfig(
            candidates=(candidate,),
            minimum_training_rows=4,
            minimum_validation_rows=3,
            require_all_classes=False,
        ),
    )
    assert result.candidates[0].training_rows == 4


def test_excludes_validation_labels_crossing_untouched_test_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
    training_index = index[:6]
    validation_index = index[6:9]
    candidate = BarrierCandidate(1.0, 1.0)
    ends = pd.DatetimeIndex(
        [
            *(index[:8] + pd.Timedelta(minutes=1)),
            index[9] + pd.Timedelta(minutes=5),
        ]
    )
    dataset = _dataset(
        index[:9],
        [
            "LOWER_FIRST",
            "UPPER_FIRST",
            "NEITHER",
            "LOWER_FIRST",
            "UPPER_FIRST",
            "NEITHER",
            "LOWER_FIRST",
            "UPPER_FIRST",
            "NEITHER",
        ],
        label_end=ends,
    )

    monkeypatch.setattr(
        barrier_selection,
        "_candidate_dataset",
        lambda *args, **kwargs: dataset,
    )
    result = select_barriers_fold_only(
        pd.DataFrame(index=index),
        pd.DataFrame(index=index),
        pd.Series(1.0, index=index),
        training_index=training_index,
        validation_index=validation_index,
        validation_end=index[8],
        required_bars=12,
        horizon_key="60m",
        config=BarrierSelectionConfig(
            candidates=(candidate,),
            minimum_training_rows=6,
            minimum_validation_rows=2,
            require_all_classes=False,
        ),
    )
    assert result.candidates[0].validation_rows == 2


def test_rejects_candidate_without_required_class_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2025-01-01", periods=9, freq="5min", tz="UTC")
    candidate = BarrierCandidate(1.0, 1.0)
    dataset = _dataset(
        index,
        ["LOWER_FIRST"] * 3 + ["UPPER_FIRST"] * 3 + ["LOWER_FIRST"] * 3,
    )
    monkeypatch.setattr(barrier_selection, "_candidate_dataset", lambda *args, **kwargs: dataset)
    with pytest.raises(ValueError, match="No barrier candidate"):
        select_barriers_fold_only(
            pd.DataFrame(index=index),
            pd.DataFrame(index=index),
            pd.Series(1.0, index=index),
            training_index=index[:6],
            validation_index=index[6:],
            validation_end=index[-1] + pd.Timedelta(minutes=5),
            required_bars=12,
            horizon_key="60m",
            config=BarrierSelectionConfig(
                candidates=(candidate,),
                minimum_training_rows=6,
                minimum_validation_rows=3,
            ),
        )


def test_candidate_grid_must_be_unique() -> None:
    candidate = BarrierCandidate(1.0, 1.0)
    with pytest.raises(ValueError, match="unique"):
        BarrierSelectionConfig(candidates=(candidate, candidate))
