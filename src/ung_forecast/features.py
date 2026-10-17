"""Point-in-time features and rolling five-session targets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(
    ung: pd.DataFrame,
    henry_hub: pd.Series | pd.DataFrame | None = None,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build features using data available on each UNG session only."""
    df = ung.copy().sort_index()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    out = pd.DataFrame(index=df.index)
    for window in (1, 3, 5, 10, 20):
        out[f"return_{window}"] = close.pct_change(window)
    for window in (5, 10, 20):
        out[f"range_{window}"] = (high.rolling(window).max() - low.rolling(window).min()) / close
        out[f"realized_vol_{window}"] = close.pct_change().rolling(window).std() * np.sqrt(252)
    for window in (5, 10, 20, 50):
        ema = close.ewm(span=window, adjust=False).mean()
        out[f"price_vs_ema_{window}"] = close / ema - 1.0
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + gain / loss))
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    out["atr_14_pct"] = tr.rolling(14).mean() / close
    out["volume_ratio_20"] = volume / volume.rolling(20).mean()
    out["volume_zscore_20"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    if henry_hub is not None:
        gas = henry_hub.squeeze().rename("henry_hub")
        gas = gas.reindex(out.index).ffill()
        out["henry_hub"] = gas
        out["henry_hub_return_1"] = gas.pct_change()
        out["ung_gas_spread_1"] = out["return_1"] - out["henry_hub_return_1"]
    if context is not None:
        if "available_at" not in context.columns:
            raise ValueError("context must contain available_at")
        available = context.sort_values("available_at").copy()
        left = pd.DataFrame({"session": out.index}).sort_values("session")
        available["available_at"] = pd.to_datetime(available["available_at"], utc=True).dt.tz_convert(None)
        right = available.reset_index(drop=False)
        merged = pd.merge_asof(left, right, left_on="session", right_on="available_at", direction="backward")
        merged.index = out.index
        for column in ("storage_bcf", "managed_money_net_pct_oi", "producer_merchant_net_pct_oi"):
            if column in merged:
                out[column] = merged[column].to_numpy()
    return out.replace([np.inf, -np.inf], np.nan)


def build_forward_targets(ung: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create future targets; rows near the end remain NaN until outcomes mature."""
    if horizon not in (1, 2, 3, 4, 5):
        raise ValueError("horizon must be between 1 and 5")
    close = ung["close"]
    future_close = close.shift(-horizon)
    future_low = ung["low"].shift(-1).rolling(horizon, min_periods=horizon).min().shift(-(horizon - 1))
    future_high = ung["high"].shift(-1).rolling(horizon, min_periods=horizon).max().shift(-(horizon - 1))
    return pd.DataFrame({
        "forward_return": future_close / close - 1,
        "forward_low_return": future_low / close - 1,
        "forward_high_return": future_high / close - 1,
    }, index=ung.index)
