from __future__ import annotations

import pytest

from ung_forecast.validation import generate_walk_forward_folds


def test_walk_forward_folds_are_ordered_and_separated() -> None:
    folds = generate_walk_forward_folds(
        sample_count=300,
        minimum_train_size=100,
        validation_size=30,
        test_size=20,
        purge_size=12,
        embargo_size=12,
        step_size=20,
    )

    assert len(folds) > 1
    for fold in folds:
        assert fold.train.stop + 12 == fold.validation.start
        assert fold.validation.stop + 12 == fold.test.start
        assert set(fold.train).isdisjoint(fold.validation)
        assert set(fold.validation).isdisjoint(fold.test)
        assert set(fold.train).isdisjoint(fold.test)


def test_training_window_expands_across_folds() -> None:
    folds = generate_walk_forward_folds(
        sample_count=260,
        minimum_train_size=100,
        validation_size=20,
        test_size=20,
        purge_size=10,
        embargo_size=10,
        step_size=20,
    )

    stops = [fold.train.stop for fold in folds]
    assert stops == sorted(stops)
    assert len(set(stops)) == len(stops)


def test_insufficient_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="Insufficient samples"):
        generate_walk_forward_folds(
            sample_count=100,
            minimum_train_size=80,
            validation_size=20,
            test_size=20,
            purge_size=10,
            embargo_size=10,
        )
