"""Point-in-time features and rolling five-session targets."""
from __future__ import annotations

import numpy as np
import pandas as pd


FORECAST_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "momentum": ("return_1", "return_3", "return_5", "return_10", "return_20", "rsi_14"),
    "volatility_range": ("range_5", "range_10", "range_20", "realized_vol_5", "realized_vol_10", "realized_vol_20", "atr_14_pct"),
    "trend": ("price_vs_ema_5", "price_vs_ema_10", "price_vs_ema_20", "price_vs_ema_50"),
    "volume": ("volume_ratio_20", "volume_zscore_20"),
    "gas_context": ("henry_hub_return_1", "ung_gas_spread_1", "storage_bcf", "managed_money_net_pct_oi", "producer_merchant_net_pct_oi"),
    "futures_curve": ("ng_front_return_1", "ng_front_return_5", "ng_front_realized_vol_5", "ng_front_volume_ratio", "ng_front_open_interest_change", "ng_curve_slope_1_2", "ng_curve_slope_1_6", "ng_calendar_spread_1_2"),
    "weather": ("hdd_forecast", "hdd_forecast_change_1d", "hdd_forecast_change_3d", "cdd_forecast", "weather_ensemble_spread"),
    "fundamentals": ("eia_storage_surprise_bcf", "eia_storage_change_bcf", "managed_money_net_pct_oi", "producer_merchant_net_pct_oi"),
    "outside_rth_mid": ("ung_after_hours_return", "ung_after_hours_range_pct", "ung_after_hours_volume_ratio", "ung_premarket_mid_return", "ung_premarket_mid_range_pct", "ung_premarket_mid_volume_ratio"),
    "outside_rth_1h": ("ung_after_hours_return", "ung_after_hours_range_pct", "ung_after_hours_volume_ratio", "ung_premarket_1h_return", "ung_premarket_1h_range_pct", "ung_premarket_1h_volume_ratio"),
}


def forecast_expert_columns(features: pd.DataFrame, *, min_observations: int = 100) -> dict[str, list[str]]:
    """Return the full-information expert and available source-group features."""
    groups = {"hist_gradient_boosting": list(features.columns)}
    for name, columns in FORECAST_FEATURE_GROUPS.items():
        available = [column for column in columns if column in features.columns and features[column].notna().any()]
        observation_count = int(features[available].notna().any(axis=1).sum()) if available else 0
        latest_has_group_data = bool(available and features.iloc[-1][available].notna().any())
        if available and observation_count >= min_observations and latest_has_group_data:
            groups[f"hgb_expert_{name}"] = available
    return groups


def build_features(
    ung: pd.DataFrame,
    henry_hub: pd.Series | pd.DataFrame | None = None,
    context: pd.DataFrame | None = None,
    outside_rth: pd.DataFrame | None = None,
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
    if outside_rth is not None and not outside_rth.empty:
        external = outside_rth.copy()
        if not isinstance(external.index, pd.DatetimeIndex):
            raise ValueError("outside_rth feature index must be a DatetimeIndex")
        if external.index.tz is not None:
            external.index = external.index.tz_convert(None)
        external.index = external.index.normalize()
        if external.index.has_duplicates:
            raise ValueError("outside_rth features contain duplicate origin sessions")
        out = out.join(external, how="left")
    if context is not None:
        if "available_at" not in context.columns:
            raise ValueError("context must contain available_at")
        for column in ("storage_bcf", "managed_money_net_pct_oi", "producer_merchant_net_pct_oi"):
            if column not in context:
                continue
            # Align each feed independently. A shared as-of join can replace an
            # EIA value with a CFTC row whose EIA field is null (and vice versa).
            available = context.sort_index()[["available_at", column]].dropna(
                subset=["available_at", column]
            ).copy()
            available["available_at"] = pd.to_datetime(
                available["available_at"], utc=True
            ).dt.tz_convert(None)
            available = available.sort_values("available_at", kind="stable").drop_duplicates(
                "available_at", keep="last"
            )
            left = pd.DataFrame({"session": pd.to_datetime(out.index)}).sort_values("session")
            merged = pd.merge_asof(
                left, available, left_on="session", right_on="available_at", direction="backward"
            )
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
