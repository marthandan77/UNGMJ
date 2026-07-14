from __future__ import annotations

import pandas as pd
import pytest

from ung_forecast.models.plain_logistic import PlainMultinomialLogisticModel


def training_data() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "momentum": [-2.0, -1.5, -1.0, 0.0, 0.1, -0.1, 1.0, 1.5, 2.0],
            "volatility": [1.0, 1.1, 0.9, 0.4, 0.5, 0.45, 0.9, 1.1, 1.0],
        }
    )
    target = pd.Series(
        [
            "LOWER_FIRST",
            "LOWER_FIRST",
            "LOWER_FIRST",
            "NEITHER",
            "NEITHER",
            "NEITHER",
            "UPPER_FIRST",
            "UPPER_FIRST",
            "UPPER_FIRST",
        ]
    )
    return features, target


def test_plain_logistic_outputs_valid_probabilities() -> None:
    features, target = training_data()
    model = PlainMultinomialLogisticModel()
    model.fit(features, target)

    forecasts = model.predict_probabilities(features.iloc[[0, 4, 8]])

    assert len(forecasts) == 3
    for forecast in forecasts:
        assert forecast.lower_first + forecast.upper_first + forecast.neither == pytest.approx(1.0)


def test_plain_logistic_enforces_feature_order() -> None:
    features, target = training_data()
    model = PlainMultinomialLogisticModel()
    model.fit(features, target)

    with pytest.raises(ValueError, match="Feature schema"):
        model.predict_probabilities(features[["volatility", "momentum"]])


def test_plain_logistic_requires_all_three_classes() -> None:
    features, target = training_data()
    with pytest.raises(ValueError, match="all outcomes"):
        PlainMultinomialLogisticModel().fit(features.iloc[:6], target.iloc[:6])
