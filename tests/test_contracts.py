from datetime import date, datetime, timezone

import pandas as pd
import pytest

from ungmj.contracts import DataValidationError, ForecastVintage, SourceSnapshot, validate_ohlcv

HASH = "a" * 64


def bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 10.5],
            "high": [11.0, 11.5],
            "low": [9.5, 10.0],
            "close": [10.5, 11.0],
            "volume": [1000, 1200],
        },
        index=pd.to_datetime(["2026-10-14", "2026-10-15"]),
    )


def test_validate_ohlcv_returns_sorted_copy() -> None:
    frame = bars().iloc[::-1]
    result = validate_ohlcv(frame, expected_latest_session=date(2026, 10, 15))
    assert result.index.is_monotonic_increasing
    assert result.iloc[-1]["close"] == 11.0
    assert frame.index[0] == pd.Timestamp("2026-10-15")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda f: f.drop(columns=["volume"]), "missing required columns"),
        (lambda f: f.assign(close=[10.5, float("nan")]), "NaN or infinity"),
        (lambda f: f.assign(open=[0.0, 10.5]), "must all be positive"),
        (lambda f: f.assign(volume=[1000, -1]), "volume must be non-negative"),
        (lambda f: f.assign(high=[9.0, 11.5]), "high must be >="),
        (lambda f: f.assign(low=[10.2, 10.0]), "low must be <="),
    ],
)
def test_validate_ohlcv_rejects_invalid_values(mutation, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        validate_ohlcv(mutation(bars()))


def test_validate_ohlcv_rejects_duplicate_sessions() -> None:
    frame = bars()
    frame.index = pd.to_datetime(["2026-10-15", "2026-10-15"])
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_ohlcv(frame)


def test_validate_ohlcv_rejects_wrong_latest_session() -> None:
    with pytest.raises(DataValidationError, match="does not match expected"):
        validate_ohlcv(bars(), expected_latest_session=date(2026, 10, 16))


def test_source_snapshot_requires_timezone_and_sha256() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        SourceSnapshot("provider", "UNG", datetime(2026, 10, 16), None, HASH, "https://example.com")
    with pytest.raises(ValueError, match="hexadecimal"):
        SourceSnapshot(
            "provider", "UNG", datetime(2026, 10, 16, tzinfo=timezone.utc),
            None, "z" * 64, "https://example.com"
        )


def test_forecast_vintage_rejects_invalid_probability_and_range() -> None:
    common = dict(
        forecast_id="f-001", instrument="UNG",
        as_of=datetime(2026, 10, 16, tzinfo=timezone.utc), origin_session=date(2026, 10, 15), horizon_sessions=1,
        reference_close=10.0, probability_up=0.6, expected_return=0.01,
        expected_close=10.1, predicted_low=9.0, predicted_high=11.0,
        lower_return_quantile=-0.1, upper_return_quantile=0.1,
        model_name="ridge", model_version="0.1.0", input_fingerprint=HASH,
    )
    ForecastVintage(**common)
    with pytest.raises(ValueError, match="probability_up"):
        ForecastVintage(**{**common, "probability_up": 1.2})
    with pytest.raises(ValueError, match="predicted_low"):
        ForecastVintage(**{**common, "predicted_low": 12.0})
