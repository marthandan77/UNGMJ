"""Empirical, out-of-sample calibration for individual forecast ranges."""
from __future__ import annotations

import math
from typing import Any


def calibrate_range(
    forecast: dict[str, Any], outcomes: list[dict[str, Any]], *, minimum: int = 30,
) -> dict[str, Any]:
    """Adjust bounds only when enough same-model/same-horizon outcomes exist."""
    rows = [
        row for row in outcomes
        if row.get("model_name") == forecast.get("model_name")
        and int(row.get("horizon", -1)) == int(forecast.get("horizon", -2))
    ]
    rows = rows[-60:]
    result = dict(forecast)
    result["calibration_sample_count"] = len(rows)
    if len(rows) < minimum:
        result["calibration_status"] = "UNCALIBRATED"
        return result
    low_errors = sorted(float(row["low_error"]) for row in rows)
    high_errors = sorted(float(row["high_error"]) for row in rows)
    # Errors are forecast - actual. If the forecast low was too high, low_error
    # is positive and the next lower bound must move down. If the forecast high
    # was too low, high_error is negative and the next upper bound must move up.
    low_error_p90 = low_errors[min(len(low_errors) - 1, math.ceil(0.90 * (len(low_errors) - 1)))]
    high_error_p10 = high_errors[max(0, int(0.10 * (len(high_errors) - 1)))]
    lower_expansion = max(low_error_p90, 0.0)
    upper_expansion = max(-high_error_p10, 0.0)
    result["predicted_low"] = max(0.01, float(result["predicted_low"]) - lower_expansion)
    result["predicted_high"] = max(
        result["predicted_low"], float(result["predicted_high"]) + upper_expansion
    )
    result["calibration_status"] = "CALIBRATED"
    return result
