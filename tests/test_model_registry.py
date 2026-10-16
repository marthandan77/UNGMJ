"""Smoke tests for independent model registry forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ungmj.model_registry import MODEL_REGISTRY, fit_model_forecast


@pytest.mark.parametrize("spec", MODEL_REGISTRY, ids=lambda spec: spec.key)
def test_registry_models_fit_independently(spec):
    rng = np.random.default_rng(20261016)
    index = pd.date_range("2020-01-01", periods=140, freq="B")
    features = pd.DataFrame(
        rng.normal(size=(140, 5)),
        index=index,
        columns=["return_1", "return_5", "realized_vol_5", "range_expansion", "gas_return_1"],
    )
    target_return = 0.002 * features["return_1"] - 0.001 * features["realized_vol_5"]
    targets = pd.DataFrame(
        {
            "forward_return": target_return,
            "forward_high_return": target_return + 0.02,
            "forward_low_return": target_return - 0.02,
        },
        index=index,
    )
    forecast = fit_model_forecast(spec.key, features, targets, 20.0, min_train_rows=100)
    assert forecast.model_name == spec.key
    assert 0.0 <= forecast.probability_up <= 1.0
    assert np.isfinite(forecast.expected_return)
    assert forecast.expected_close > 0
    assert forecast.predicted_low <= forecast.predicted_high
