from __future__ import annotations

import pandas as pd
import pytest

from ung_forecast.training import runner_60m
from ung_forecast.training.barrier_selection import (
    BarrierCandidate,
    BarrierCandidateResult,
    BarrierSelectionConfig,
    BarrierSelectionResult,
)
from ung_forecast.training.dataset import TrainingDataset
from ung_forecast.training.runner_60m import (
    SixtyMinuteRunnerConfig,
    build_sixty_minute_run_plan,
)


def _market_data(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2025-01-02 14:30", periods=rows, freq="5min", tz="UTC")
    close = pd.Series(range(rows), index=index, dtype=float) + 20.0
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": 1000.0,
        },
        index=index,
    )


def _dataset(index: pd.DatetimeIndex) -> TrainingDataset:
    targets = ["LOWER_FIRST", "UPPER_FIRST", "NEITHER"]
    target = pd.Series(
        [targets[position % 3] for position in range(len(index))],
        index=index,
        name="target",
    )
    return TrainingDataset(
        features=pd.DataFrame({"x": range(len(index))}, index=index, dtype=float),
        target=target,
        label_end_time=pd.Series(
            index + pd.Timedelta(minutes=5), index=index, name="label_end_time"
        ),
        metadata=pd.DataFrame(index=index),
    )


def test_runner_uses_first_test_timestamp_as_strict_selection_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _market_data()
    features = pd.DataFrame({"x": 1.0}, index=market.index)
    volatility = pd.Series(0.02, index=market.index)
    candidate = BarrierCandidate(1.0, 1.0)
    dataset = _dataset(market.index[:-12])
    captured_boundaries: list[pd.Timestamp] = []

    def fake_select(*args: object, **kwargs: object) -> BarrierSelectionResult:
        del args
        captured_boundaries.append(pd.Timestamp(kwargs["validation_end"]))
        result = BarrierCandidateResult(
            candidate=candidate,
            accepted=True,
            validation_brier=0.5,
            training_rows=30,
            validation_rows=10,
            training_classes=("LOWER_FIRST", "NEITHER", "UPPER_FIRST"),
            validation_classes=("LOWER_FIRST", "NEITHER", "UPPER_FIRST"),
            rejection_reason=None,
        )
        return BarrierSelectionResult(selected=candidate, candidates=(result,))

    monkeypatch.setattr(runner_60m, "select_barriers_fold_only", fake_select)
    monkeypatch.setattr(runner_60m, "build_training_dataset", lambda *args, **kwargs: dataset)

    plan = build_sixty_minute_run_plan(
        market,
        features,
        volatility,
        config=SixtyMinuteRunnerConfig(
            minimum_train_size=30,
            validation_size=12,
            test_size=12,
            purge_size=2,
            embargo_size=2,
            barrier_selection=BarrierSelectionConfig(
                candidates=(candidate,),
                minimum_training_rows=10,
                minimum_validation_rows=3,
                require_all_classes=False,
            ),
        ),
    )

    assert plan.folds
    assert len(captured_boundaries) == len(plan.folds)
    for fold, boundary in zip(plan.folds, captured_boundaries, strict=True):
        assert boundary == fold.test_index[0]
        assert fold.validation_index.max() < fold.test_index.min()
        validation_label_ends = pd.to_datetime(
            fold.dataset.label_end_time.loc[fold.validation_index]
        )
        assert (validation_label_ends < fold.test_index[0]).all()


def test_runner_config_rejects_invalid_fold_sizes() -> None:
    candidate = BarrierCandidate(1.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        SixtyMinuteRunnerConfig(
            minimum_train_size=0,
            validation_size=10,
            test_size=10,
            purge_size=0,
            embargo_size=0,
            barrier_selection=BarrierSelectionConfig(candidates=(candidate,)),
        )
