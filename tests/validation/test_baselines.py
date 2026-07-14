from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ung_forecast.validation.baselines import (
    RecencyWeightedBaselineConfig,
    constant_probability_frame,
    recency_weighted_class_probabilities,
)
from ung_forecast.validation.metrics import CLASS_ORDER


def test_recency_weighting_gives_more_mass_to_recent_class() -> None:
    target = pd.Series(
        ["LOWER_FIRST"] * 50 + ["UPPER_FIRST"] * 10 + ["NEITHER"] * 10,
        dtype="object",
    )
    probabilities = recency_weighted_class_probabilities(
        target,
        config=RecencyWeightedBaselineConfig(half_life=5.0),
    )

    assert probabilities.index.tolist() == CLASS_ORDER
    assert probabilities["NEITHER"] > probabilities["LOWER_FIRST"]
    assert float(probabilities.sum()) == pytest.approx(1.0)


def test_recency_baseline_uses_only_supplied_training_rows() -> None:
    earlier = pd.Series(["LOWER_FIRST", "UPPER_FIRST", "NEITHER"] * 20)
    with_future = pd.concat([earlier, pd.Series(["UPPER_FIRST"] * 100)], ignore_index=True)

    earlier_probabilities = recency_weighted_class_probabilities(earlier)
    repeated = constant_probability_frame(earlier_probabilities, pd.RangeIndex(5))

    assert np.allclose(repeated.iloc[0].to_numpy(), repeated.iloc[-1].to_numpy())
    assert not np.allclose(
        earlier_probabilities.to_numpy(),
        recency_weighted_class_probabilities(with_future).to_numpy(),
    )


def test_recency_baseline_rejects_unknown_class() -> None:
    with pytest.raises(ValueError, match="Unknown target classes"):
        recency_weighted_class_probabilities(pd.Series(["LOWER_FIRST", "UNKNOWN"]))


def test_constant_probability_frame_requires_unit_sum() -> None:
    probabilities = pd.Series(
        {"LOWER_FIRST": 0.5, "UPPER_FIRST": 0.5, "NEITHER": 0.5}
    )
    with pytest.raises(ValueError, match="sum to one"):
        constant_probability_frame(probabilities, pd.RangeIndex(2))
