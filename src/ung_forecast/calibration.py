"""Horizon-specific online adaptive calibration for joint daily ranges."""
from __future__ import annotations

import math
from typing import Any


def _adaptive_miscoverage(rows: list[dict[str, Any]], target: float, learning_rate: float) -> float:
    """Replay the ACI level updates from matured outcomes in time order."""
    alpha = target
    for row in rows:
        covered = bool(row.get("range_contains_day", False))
        error = 0.0 if covered else 1.0
        # A miss lowers alpha, which raises the residual quantile and widens
        # the next interval. Sustained coverage does the reverse.
        alpha += learning_rate * (target - error)
        alpha = min(0.50, max(0.01, alpha))
    return alpha


def calibrate_range(
    forecast: dict[str, Any], outcomes: list[dict[str, Any]], *,
    minimum: int = 20, target_coverage: float = 0.90,
    learning_rate: float = 0.02, window: int = 60,
) -> dict[str, Any]:
    """Adjust joint low/high range margins with online, horizon-specific ACI.

    Each horizon is calibrated independently from matured outcomes for the
    same forecast. Errors are replayed chronologically so a miss on a five-day
    forecast affects the five-day calibration only after that forecast matures.
    """
    rows = [
        row for row in outcomes
        if row.get("model_name") == forecast.get("model_name")
        and int(row.get("horizon", -1)) == int(forecast.get("horizon", -2))
        and "low_error" in row and "high_error" in row
    ]
    rows.sort(key=lambda row: (str(row.get("target_session", "")), str(row.get("scored_at", ""))))
    unique_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("target_session", row.get("scored_at", len(unique_rows))))
        unique_rows[key] = row
    rows = list(unique_rows.values())
    result = dict(forecast)
    result["calibration_sample_count"] = len(rows)
    result["calibration_target_coverage"] = target_coverage
    if len(rows) < minimum:
        result["calibration_status"] = "WARMING_UP"
        result["calibration_miscoverage"] = 1.0 - target_coverage
        return result

    alpha = _adaptive_miscoverage(rows, 1.0 - target_coverage, learning_rate)
    recent = rows[-window:]
    joint_misses = sorted(max(
        max(float(row["low_error"]), 0.0),
        max(-float(row["high_error"]), 0.0),
    ) for row in recent)
    quantile_index = min(len(recent) - 1, max(0, math.ceil((len(recent) + 1) * (1.0 - alpha)) - 1))
    lower_expansion = upper_expansion = joint_misses[quantile_index]
    result["predicted_low"] = max(0.01, float(result["predicted_low"]) - lower_expansion)
    result["predicted_high"] = max(
        result["predicted_low"], float(result["predicted_high"]) + upper_expansion
    )
    result["calibration_status"] = "ADAPTIVE_CONFORMAL"
    result["calibration_miscoverage"] = alpha
    result["calibration_lower_expansion"] = lower_expansion
    result["calibration_upper_expansion"] = upper_expansion
    result["calibration_recent_coverage"] = sum(
        bool(row.get("range_contains_day", False)) for row in recent
    ) / len(recent)
    return result
