from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ungmj.contracts import DataValidationError, ForecastVintage
from ungmj.features import build_features, build_forward_targets
from ungmj.ledger import ForecastLedger
from ungmj.model import evaluate_predictions, fit_and_forecast, walk_forward_predictions


def bars(n: int = 360) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = 20 * np.exp(np.cumsum(0.001 + 0.01 * np.sin(np.arange(n) / 7)))
    return pd.DataFrame({
        "open": close * 0.998,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": 1_000_000 + (np.arange(n) % 17) * 1000,
    }, index=idx)


def test_features_are_past_only_and_targets_are_separate() -> None:
    frame = bars()
    features = build_features(frame)
    targets = build_forward_targets(frame, 3)
    assert "forward_return" not in features
    assert targets["forward_return"].iloc[-3:].isna().all()
    changed_future = frame.copy()
    changed_future.iloc[-1, changed_future.columns.get_loc("close")] *= 1.5
    changed_features = build_features(changed_future)
    pd.testing.assert_series_equal(features.iloc[-2], changed_features.iloc[-2])


def test_henry_hub_features_are_lagged() -> None:
    frame = bars(100)
    gas = pd.DataFrame({"henry_hub_spot": np.linspace(2, 5, 100)}, index=frame.index)
    features = build_features(frame, gas)
    assert pd.isna(features["henry_hub_spot"].iloc[0])
    assert features["henry_hub_spot"].iloc[1] == pytest.approx(2.0)


def test_forward_target_uses_next_horizon_sessions() -> None:
    frame = bars(20)
    targets = build_forward_targets(frame, 2)
    i = 5
    expected = frame["close"].iloc[i + 2] / frame["close"].iloc[i] - 1
    assert targets["forward_return"].iloc[i] == pytest.approx(expected)
    assert targets["forward_high_return"].iloc[i] == pytest.approx(
        frame["high"].iloc[i + 1:i + 3].max() / frame["close"].iloc[i] - 1
    )


def test_ridge_forecast_outputs_valid_prices() -> None:
    frame = bars()
    features = build_features(frame)
    target = build_forward_targets(frame, 1)
    eligible = target.loc[target.index <= frame.index[-2]]
    result = fit_and_forecast(features, eligible, float(frame["close"].iloc[-1]), min_train_rows=100)
    assert 0 <= result.probability_up <= 1
    assert result.predicted_low <= result.predicted_high
    assert result.expected_close > 0


def test_walk_forward_predictions_are_chronological() -> None:
    frame = bars()
    features = build_features(frame)
    targets = build_forward_targets(frame, 1)
    pred = walk_forward_predictions(features, targets, min_train_rows=150, step=5)
    assert len(pred) > 0
    assert pred.index.is_monotonic_increasing
    assert set(pred.columns) >= {"predicted_return", "actual_return", "absolute_error"}


def test_evaluation_empty_is_explicit() -> None:
    assert evaluate_predictions(pd.DataFrame())["n"] == 0


def vintage(fid: str, as_of: datetime, horizon: int = 1) -> ForecastVintage:
    return ForecastVintage(
        forecast_id=fid, instrument="UNG", as_of=as_of, horizon_sessions=horizon,
        reference_close=20, probability_up=0.6, expected_return=0.01, expected_close=20.2,
        predicted_low=19, predicted_high=22, lower_return_quantile=-0.1,
        upper_return_quantile=0.1, model_name="ridge", model_version="0.1.0",
        input_fingerprint="a" * 64,
    )


def test_ledger_rejects_duplicate_forecast_id(tmp_path) -> None:
    ledger = ForecastLedger(tmp_path)
    f = vintage("one", datetime(2026, 1, 1, tzinfo=timezone.utc))
    ledger.append_forecast(f)
    with pytest.raises(DataValidationError, match="already exists"):
        ledger.append_forecast(f)


def test_ledger_scores_matured_vintage_once(tmp_path) -> None:
    ledger = ForecastLedger(tmp_path)
    as_of = datetime(2026, 1, 5, 22, tzinfo=timezone.utc)
    ledger.append_forecast(vintage("one", as_of, horizon=1))
    frame = pd.DataFrame({"close": [20.0, 21.0]}, index=pd.to_datetime(["2026-01-05", "2026-01-06"]))
    now = datetime(2026, 1, 7, 12, tzinfo=timezone.utc)
    assert ledger.score_matured(frame, now=now) == 1
    assert ledger.score_matured(frame, now=now) == 0
