"""Canonical barrier construction shared by labeling and forecasting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BarrierDefinition:
    current_price: float
    lower_price: float
    upper_price: float
    horizon_key: str
    volatility_estimate: float
    lower_multiplier: float
    upper_multiplier: float

    def __post_init__(self) -> None:
        if self.current_price <= 0:
            raise ValueError("current_price must be positive")
        if self.volatility_estimate <= 0:
            raise ValueError("volatility_estimate must be positive")
        if self.lower_multiplier <= 0 or self.upper_multiplier <= 0:
            raise ValueError("barrier multipliers must be positive")
        if not self.lower_price < self.current_price < self.upper_price:
            raise ValueError("Barrier prices must satisfy lower < current < upper")


def create_barriers(
    *,
    current_price: float,
    volatility_estimate: float,
    lower_multiplier: float,
    upper_multiplier: float,
    horizon_key: str,
    minimum_lower_distance: float = 0.0,
    minimum_upper_distance: float = 0.0,
) -> BarrierDefinition:
    """Construct symmetric or asymmetric volatility-scaled price barriers.

    Distances are in price units. Economic floors can be supplied by the
    decision layer without changing the underlying volatility formula.
    """

    if minimum_lower_distance < 0 or minimum_upper_distance < 0:
        raise ValueError("minimum distances cannot be negative")

    lower_distance = max(lower_multiplier * volatility_estimate, minimum_lower_distance)
    upper_distance = max(upper_multiplier * volatility_estimate, minimum_upper_distance)
    lower_price = current_price - lower_distance
    upper_price = current_price + upper_distance
    if lower_price <= 0:
        raise ValueError("Lower barrier must remain positive")

    return BarrierDefinition(
        current_price=current_price,
        lower_price=lower_price,
        upper_price=upper_price,
        horizon_key=horizon_key,
        volatility_estimate=volatility_estimate,
        lower_multiplier=lower_multiplier,
        upper_multiplier=upper_multiplier,
    )
