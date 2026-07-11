"""Generate expanding-window train/validation/test folds with purge and embargo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    train: range
    validation: range
    test: range

    def __post_init__(self) -> None:
        if not self.train or not self.validation or not self.test:
            raise ValueError("Fold segments cannot be empty")
        if self.train.stop > self.validation.start:
            raise ValueError("Training overlaps validation")
        if self.validation.stop > self.test.start:
            raise ValueError("Validation overlaps test")


def generate_walk_forward_folds(
    *,
    sample_count: int,
    minimum_train_size: int,
    validation_size: int,
    test_size: int,
    purge_size: int,
    embargo_size: int,
    step_size: int | None = None,
) -> list[WalkForwardFold]:
    if min(
        sample_count,
        minimum_train_size,
        validation_size,
        test_size,
    ) <= 0:
        raise ValueError("sample and segment sizes must be positive")
    if purge_size < 0 or embargo_size < 0:
        raise ValueError("purge and embargo sizes cannot be negative")

    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    folds: list[WalkForwardFold] = []
    validation_start = minimum_train_size + purge_size
    while True:
        validation_stop = validation_start + validation_size
        test_start = validation_stop + embargo_size
        test_stop = test_start + test_size
        if test_stop > sample_count:
            break

        train_stop = validation_start - purge_size
        folds.append(
            WalkForwardFold(
                train=range(0, train_stop),
                validation=range(validation_start, validation_stop),
                test=range(test_start, test_stop),
            )
        )
        validation_start += step

    if not folds:
        raise ValueError("Insufficient samples for one complete walk-forward fold")
    return folds
