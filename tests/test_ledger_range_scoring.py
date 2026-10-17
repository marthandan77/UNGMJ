"""Tests for realized high/low range scoring."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ungmj.contracts import ForecastVintage
from ungmj.ledger import ForecastLedger


def test_scores_realized_five_session_low_high_range(tmp_path):
    index = pd.date_range("2026-01-05", periods=7, freq="B")
    bars = pd.DataFrame(
        {
            "open": [20.0, 20.1, 20.2, 19.8, 20.4, 20.0, 20.3],
            "high": [20.2, 20.8, 21.0, 20.1, 21.2, 20.7, 20.9],
            "low": [19.8, 19.9, 19.7, 19.2, 19.8, 19.6, 20.0],
            "close": [20.0, 20.4, 20.0, 19.9, 20.5, 20.3, 20.6],
            "volume": [1000] * 7,
        },
        index=index,
    )
    origin = index[0].date()
    forecast = ForecastVintage(
        forecast_id="test-five-session-range",
        instrument="UNG",
        as_of=datetime(2026, 1, 5, 22, 0, tzinfo=timezone.utc),
        origin_session=origin,
        horizon_sessions=5,
        reference_close=20.0,
        probability_up=0.55,
        expected_return=0.01,
        expected_close=20.2,
        predicted_low=19.5,
        predicted_high=21.0,
        lower_return_quantile=-0.03,
        upper_return_quantile=0.04,
        calibration_sample_count=0,
        calibration_method="direct_low_high_models_uncalibrated",
        model_name="ridge",
        model_version="0.2.0",
        input_fingerprint="a" * 64,
        status="PROVISIONAL",
    )
    ledger = ForecastLedger(tmp_path)
    ledger.append_forecast(forecast)

    scored = ledger.score_matured(
        bars,
        now=datetime(2026, 1, 13, 12, 0, tzinfo=timezone.utc),
    )
    assert scored == 1
    outcome = pd.read_json(tmp_path / "outcomes.jsonl", lines=True).iloc[0]
    assert outcome["actual_low"] == 19.2
    assert outcome["actual_high"] == 21.2
    assert outcome["range_covered"] == False
    assert outcome["low_range_error"] < 0
    assert outcome["high_range_error"] > 0
    assert abs(float(outcome["low_price_absolute_error"]) - 0.3) < 1e-9
    assert abs(float(outcome["high_price_absolute_error"]) - 0.2) < 1e-9
