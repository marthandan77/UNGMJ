"""Canonical feature formulas shared by research and forecasting.

All rolling calculations are right-aligned and use only observations available
at or before each output timestamp.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def log_return(close: pd.Series, periods: int) -> pd.Series:
    if periods <= 0:
        raise ValueError("periods must be positive")
    if (close <= 0).any():
        raise ValueError("close prices must be positive")
    return np.log(close / close.shift(periods)).rename(f"log_return_{periods}")


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        raise ValueError("window must exceed one")
    one_period_returns = log_return(close, 1)
    values = one_period_returns.pow(2).rolling(window=window, min_periods=window).sum()
    return np.sqrt(values).rename(f"realized_volatility_{window}")


def session_vwap(frame: pd.DataFrame) -> pd.Series:
    required = {"High", "Low", "Close", "Volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for VWAP: {sorted(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("VWAP requires a timezone-aware DatetimeIndex")

    typical_price = (frame["High"] + frame["Low"] + frame["Close"]) / 3.0
    session_key = pd.Series(frame.index.date, index=frame.index)
    cumulative_value = (typical_price * frame["Volume"]).groupby(session_key).cumsum()
    cumulative_volume = frame["Volume"].groupby(session_key).cumsum()
    vwap = cumulative_value / cumulative_volume.replace(0, np.nan)
    return vwap.rename("session_vwap")


def standardized_vwap_deviation(frame: pd.DataFrame, window: int) -> pd.Series:
    if window <= 1:
        raise ValueError("window must exceed one")
    deviation = frame["Close"] - session_vwap(frame)
    rolling_mean = deviation.rolling(window=window, min_periods=window).mean()
    rolling_std = deviation.rolling(window=window, min_periods=window).std(ddof=0)
    zscore = (deviation - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore.rename(f"vwap_z_{window}")


def variance_ratio(close: pd.Series, q: int, window: int) -> pd.Series:
    if q <= 1:
        raise ValueError("q must exceed one")
    if window <= q:
        raise ValueError("window must exceed q")

    one_period = log_return(close, 1)
    q_period = log_return(close, q)
    one_variance = one_period.rolling(window=window, min_periods=window).var(ddof=0)
    q_variance = q_period.rolling(window=window, min_periods=window).var(ddof=0)
    ratio = q_variance / (q * one_variance.replace(0, np.nan))
    return ratio.rename(f"variance_ratio_q{q}_w{window}")


def time_adjusted_relative_volume(volume: pd.Series, lookback_sessions: int) -> pd.Series:
    if lookback_sessions <= 0:
        raise ValueError("lookback_sessions must be positive")
    if not isinstance(volume.index, pd.DatetimeIndex) or volume.index.tz is None:
        raise ValueError("Relative volume requires a timezone-aware DatetimeIndex")

    slot = pd.Series(volume.index.strftime("%H:%M"), index=volume.index)
    output = pd.Series(index=volume.index, dtype=float, name="relative_volume")
    for _, positions in slot.groupby(slot).groups.items():
        slot_volume = volume.loc[positions]
        historical_median = slot_volume.shift(1).rolling(
            window=lookback_sessions,
            min_periods=lookback_sessions,
        ).median()
        output.loc[positions] = slot_volume / historical_median.replace(0, np.nan)
    return output


def time_of_day_encoding(index: pd.DatetimeIndex) -> pd.DataFrame:
    if index.tz is None:
        raise ValueError("Time encoding requires timezone-aware timestamps")
    minutes = index.hour * 60 + index.minute
    angle = 2.0 * math.pi * minutes / (24.0 * 60.0)
    return pd.DataFrame(
        {
            "time_sin": np.sin(angle),
            "time_cos": np.cos(angle),
        },
        index=index,
    )


def rolling_residual_zscore(
    primary_returns: pd.Series,
    comparison_returns: pd.Series,
    *,
    regression_window: int,
    zscore_window: int,
    ridge_alpha: float = 1.0,
) -> pd.Series:
    """Estimate an out-of-sample rolling residual and standardize it.

    At timestamp t, the regression is fitted only on observations strictly
    before t. The current comparison return is then used to predict the current
    primary return.
    """

    if regression_window <= 1 or zscore_window <= 1:
        raise ValueError("rolling windows must exceed one")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha cannot be negative")

    aligned = pd.concat(
        [primary_returns.rename("primary"), comparison_returns.rename("comparison")],
        axis=1,
        join="inner",
    )
    residual = pd.Series(index=aligned.index, dtype=float, name="ung_ng_residual")

    for position in range(regression_window, len(aligned)):
        history = aligned.iloc[position - regression_window : position].dropna()
        current = aligned.iloc[position]
        if len(history) < regression_window or current.isna().any():
            continue
        model = Ridge(alpha=ridge_alpha)
        model.fit(history[["comparison"]].to_numpy(), history["primary"].to_numpy())
        prediction = float(model.predict(np.array([[current["comparison"]]]))[0])
        residual.iloc[position] = float(current["primary"] - prediction)

    rolling_mean = residual.shift(1).rolling(
        window=zscore_window,
        min_periods=zscore_window,
    ).mean()
    rolling_std = residual.shift(1).rolling(
        window=zscore_window,
        min_periods=zscore_window,
    ).std(ddof=0)
    return ((residual - rolling_mean) / rolling_std.replace(0, np.nan)).rename(
        "ung_ng_residual_z"
    )
