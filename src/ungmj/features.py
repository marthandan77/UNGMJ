"""Point-in-time feature engineering and separate forward-label construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import DataValidationError, validate_ohlcv


def build_features(ohlcv: pd.DataFrame, henry_hub: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build features known at each UNG close; no centered windows or future fills."""
    bars = validate_ohlcv(ohlcv)
    close, high, low, volume = bars["close"], bars["high"], bars["low"], bars["volume"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    features = pd.DataFrame(index=bars.index)
    log_return = np.log(close).diff()
    features["return_1"] = log_return
    for window in (2, 3, 5, 10, 20, 60):
        features[f"momentum_{window}"] = np.log(close / close.shift(window))
    features["overnight_gap"] = np.log(bars["open"] / previous_close)
    features["true_range_pct"] = true_range / previous_close
    features["realized_vol_5"] = log_return.rolling(5, min_periods=5).std()
    features["realized_vol_20"] = log_return.rolling(20, min_periods=20).std()
    features["range_expansion"] = (high - low) / true_range.rolling(20, min_periods=10).median()
    features["close_location"] = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    log_volume = np.log1p(volume)
    features["volume_surprise_20"] = (
        log_volume - log_volume.rolling(20, min_periods=10).mean()
    ) / log_volume.rolling(20, min_periods=10).std().replace(0, np.nan)
    for window in (10, 20, 60):
        mean = close.rolling(window, min_periods=window).mean()
        features[f"trend_distance_{window}"] = close / mean - 1.0
    if henry_hub is not None:
        if "henry_hub_spot" not in henry_hub:
            raise DataValidationError("Henry Hub frame requires henry_hub_spot")
        gas = henry_hub[["henry_hub_spot"]].copy()
        gas.index = pd.to_datetime(gas.index).normalize()
        if gas.index.has_duplicates:
            raise DataValidationError("Henry Hub observations contain duplicate sessions")
        gas["henry_hub_spot"] = pd.to_numeric(gas["henry_hub_spot"], errors="raise")
        if gas["henry_hub_spot"].isna().any() or (gas["henry_hub_spot"] <= 0).any():
            raise DataValidationError("Henry Hub spot observations must be positive and complete")
        # Spot series may be released after the UNG close; use a one-session lag conservatively.
        gas["henry_hub_return_1"] = np.log(gas["henry_hub_spot"]).diff()
        features = features.join(gas[["henry_hub_spot", "henry_hub_return_1"]].shift(1), how="left")
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def build_forward_targets(ohlcv: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build future return and future high/low labels for training/scoring only."""
    if horizon not in (1, 2, 3, 5):
        raise ValueError("horizon must be one of 1, 2, 3, 5 sessions")
    bars = validate_ohlcv(ohlcv)
    close, high, low = bars["close"], bars["high"], bars["low"]
    future_close = close.shift(-horizon)
    future_high = pd.concat([high.shift(-step) for step in range(1, horizon + 1)], axis=1).max(axis=1)
    future_low = pd.concat([low.shift(-step) for step in range(1, horizon + 1)], axis=1).min(axis=1)
    result = pd.DataFrame(index=bars.index)
    result["forward_return"] = future_close / close - 1.0
    result["forward_high_return"] = future_high / close - 1.0
    result["forward_low_return"] = future_low / close - 1.0
    return result
