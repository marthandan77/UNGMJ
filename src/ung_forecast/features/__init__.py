"""Leakage-safe shared feature calculations."""

from .engine import FeatureBuildResult, build_feature_frame
from .formulas import (
    log_return,
    realized_volatility,
    rolling_residual_zscore,
    session_vwap,
    standardized_vwap_deviation,
    time_adjusted_relative_volume,
    time_of_day_encoding,
    variance_ratio,
)

__all__ = [
    "FeatureBuildResult",
    "build_feature_frame",
    "log_return",
    "realized_volatility",
    "rolling_residual_zscore",
    "session_vwap",
    "standardized_vwap_deviation",
    "time_adjusted_relative_volume",
    "time_of_day_encoding",
    "variance_ratio",
]
