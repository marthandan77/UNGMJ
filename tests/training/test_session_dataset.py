from __future__ import annotations

import pandas as pd

from ung_forecast.training.session_dataset import build_session_training_dataset


def daily_data() -> pd.DataFrame:
    index = pd.bdate_range("2026-01-05", periods=12, tz="America/New_York")
    close = pd.Series([10.0, 10.1, 10.2, 10.0, 9.8, 9.9, 10.2, 10.3, 10.1, 10.0, 10.4, 10.5], index=index)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.15,
            "Low": close - 0.15,
            "Close": close,
            "Volume": 1000,
        },
        index=index,
    )


def test_session_dataset_uses_future_daily_sessions() -> None:
    daily = daily_data()
    feature_index = daily.index[:-2] + pd.Timedelta(hours=16)
    features = pd.DataFrame({"feature": range(len(feature_index))}, index=feature_index)
    volatility = pd.Series(0.1, index=feature_index)

    dataset = build_session_training_dataset(
        daily,
        features,
        volatility,
        sessions_ahead=2,
        lower_multiplier=1.0,
        upper_multiplier=1.0,
        horizon_key="2d",
    )

    assert not dataset.features.empty
    for timestamp, label_end in dataset.label_end_time.items():
        session_position = daily.index.get_loc(pd.Timestamp(timestamp).normalize())
        expected_end = daily.index[session_position + 2]
        assert label_end == expected_end


def test_session_dataset_rejects_unsupported_session_count() -> None:
    daily = daily_data()
    features = pd.DataFrame({"feature": [1.0]}, index=[daily.index[0]])
    volatility = pd.Series([0.1], index=features.index)

    try:
        build_session_training_dataset(
            daily,
            features,
            volatility,
            sessions_ahead=3,
            lower_multiplier=1.0,
            upper_multiplier=1.0,
            horizon_key="3d",
        )
    except ValueError as exc:
        assert "1, 2, or 7" in str(exc)
    else:
        raise AssertionError("Expected unsupported session-count failure")
