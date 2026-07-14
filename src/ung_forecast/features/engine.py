"""Horizon-aware feature construction using one shared formula library."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ung_forecast.horizons import HorizonKey

from .formulas import (
    log_return,
    realized_volatility,
    rolling_residual_zscore,
    standardized_vwap_deviation,
    time_adjusted_relative_volume,
    time_of_day_encoding,
    variance_ratio,
)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    return_windows: tuple[int, ...]
    volatility_window: int
    vwap_window: int | None
    variance_ratio_q: int
    variance_ratio_window: int
    relative_volume_sessions: int
    residual_regression_window: int
    residual_zscore_window: int


FEATURE_SPECS: dict[HorizonKey, FeatureSpec] = {
    HorizonKey.MINUTES_60: FeatureSpec((3, 6), 6, 12, 3, 24, 10, 36, 24),
    HorizonKey.HOURS_4: FeatureSpec((4, 8), 16, 16, 4, 32, 10, 40, 24),
    HorizonKey.DAY_1: FeatureSpec((1, 3), 8, 8, 3, 24, 10, 30, 20),
    HorizonKey.DAYS_2: FeatureSpec((1, 2, 5), 10, None, 3, 20, 10, 30, 20),
    HorizonKey.DAYS_7: FeatureSpec((5, 10, 20), 20, None, 5, 40, 20, 40, 30),
}


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    frame: pd.DataFrame
    max_source_timestamp: pd.Series
    feature_version: str
    includes_comparison: bool

    def assert_no_future_sources(self) -> None:
        aligned = self.max_source_timestamp.reindex(self.frame.index)
        if aligned.isna().any():
            raise ValueError("Missing source-timestamp audit values")
        output_times = pd.Series(self.frame.index, index=self.frame.index)
        if (aligned > output_times).any():
            raise ValueError("Feature frame contains future source timestamps")


def build_feature_frame(
    primary: pd.DataFrame,
    *,
    horizon: HorizonKey,
    comparison: pd.DataFrame | None = None,
    feature_version: str = "features-v1",
) -> FeatureBuildResult:
    if not isinstance(primary.index, pd.DatetimeIndex) or primary.index.tz is None:
        raise ValueError("Primary data must use a timezone-aware DatetimeIndex")
    if not primary.index.is_monotonic_increasing or primary.index.has_duplicates:
        raise ValueError("Primary timestamps must be unique and increasing")

    spec = FEATURE_SPECS[horizon]
    features = pd.DataFrame(index=primary.index)
    for window in spec.return_windows:
        features[f"return_{window}"] = log_return(primary["Close"], window)
    features["realized_volatility"] = realized_volatility(
        primary["Close"], spec.volatility_window
    )
    if spec.vwap_window is not None:
        features["vwap_z"] = standardized_vwap_deviation(primary, spec.vwap_window)
    features["variance_ratio"] = variance_ratio(
        primary["Close"],
        q=spec.variance_ratio_q,
        window=spec.variance_ratio_window,
    )
    features["relative_volume"] = time_adjusted_relative_volume(
        primary["Volume"], spec.relative_volume_sessions
    )
    features = features.join(time_of_day_encoding(primary.index))

    includes_comparison = comparison is not None
    if comparison is not None:
        if not primary.index.equals(comparison.index):
            raise ValueError("Primary and comparison timestamps must already be synchronized")
        primary_returns = log_return(primary["Close"], 1)
        comparison_returns = log_return(comparison["Close"], 1)
        features["ung_ng_residual_z"] = rolling_residual_zscore(
            primary_returns,
            comparison_returns,
            regression_window=spec.residual_regression_window,
            zscore_window=spec.residual_zscore_window,
        )

    source_audit = pd.Series(primary.index, index=primary.index, name="max_source_timestamp")
    result = FeatureBuildResult(
        frame=features,
        max_source_timestamp=source_audit,
        feature_version=feature_version,
        includes_comparison=includes_comparison,
    )
    result.assert_no_future_sources()
    return result
