from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ung_forecast.horizons import HorizonKey
from ung_forecast.models import (
    ElasticNetMultinomialModel,
    HorizonModelRegistry,
    ModelRecord,
)
from ung_forecast.schemas import OutcomeClass


def training_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        rng.normal(size=(180, 4)),
        columns=["a", "b", "c", "d"],
    )
    labels = pd.Series(
        [
            OutcomeClass.LOWER_FIRST.value,
            OutcomeClass.UPPER_FIRST.value,
            OutcomeClass.NEITHER.value,
        ]
        * 60
    )
    return features, labels


def test_elastic_net_probabilities_sum_to_one() -> None:
    features, labels = training_data()
    model = ElasticNetMultinomialModel()
    model.fit(features, labels)

    forecasts = model.predict_probabilities(features.iloc[:5])

    assert len(forecasts) == 5
    for forecast in forecasts:
        assert forecast.lower_first + forecast.upper_first + forecast.neither == pytest.approx(1.0)


def test_model_rejects_feature_schema_change() -> None:
    features, labels = training_data()
    model = ElasticNetMultinomialModel()
    model.fit(features, labels)

    with pytest.raises(ValueError, match="Feature schema"):
        model.predict_probabilities(features[["b", "a", "c", "d"]].iloc[:1])


def test_registry_keeps_horizons_independent() -> None:
    registry = HorizonModelRegistry()
    sixty = ModelRecord(HorizonKey.MINUTES_60, "60m-v1", "features-v1", True, True)
    four_hour = ModelRecord(HorizonKey.HOURS_4, "4h-v1", "features-v1", True, True)
    registry.register_champion(sixty)
    registry.register_champion(four_hour)

    assert registry.resolve(HorizonKey.MINUTES_60, comparison_available=True) == sixty
    assert registry.resolve(HorizonKey.HOURS_4, comparison_available=True) == four_hour
    assert registry.champion_horizons() == {
        HorizonKey.MINUTES_60,
        HorizonKey.HOURS_4,
    }


def test_registry_uses_separately_trained_no_comparison_fallback() -> None:
    registry = HorizonModelRegistry()
    fallback = ModelRecord(HorizonKey.MINUTES_60, "60m-fallback-v1", "features-v1", False, True)
    registry.register_fallback(fallback)

    assert registry.resolve(HorizonKey.MINUTES_60, comparison_available=False) == fallback


def test_fallback_cannot_require_comparison_data() -> None:
    registry = HorizonModelRegistry()
    with pytest.raises(ValueError, match="must not require"):
        registry.register_fallback(
            ModelRecord(HorizonKey.MINUTES_60, "bad", "features-v1", True, True)
        )
