"""Versioned model registry with separate champion and fallback records per horizon."""

from __future__ import annotations

from dataclasses import dataclass

from ung_forecast.horizons import HorizonKey


@dataclass(frozen=True, slots=True)
class ModelRecord:
    horizon: HorizonKey
    model_version: str
    feature_version: str
    includes_comparison: bool
    approved: bool


class HorizonModelRegistry:
    def __init__(self) -> None:
        self._champions: dict[HorizonKey, ModelRecord] = {}
        self._fallbacks: dict[HorizonKey, ModelRecord] = {}

    def register_champion(self, record: ModelRecord) -> None:
        if not record.approved:
            raise ValueError("Champion model must be approved")
        self._champions[record.horizon] = record

    def register_fallback(self, record: ModelRecord) -> None:
        if record.includes_comparison:
            raise ValueError("Fallback model must not require comparison data")
        self._fallbacks[record.horizon] = record

    def resolve(self, horizon: HorizonKey, *, comparison_available: bool) -> ModelRecord:
        if comparison_available:
            champion = self._champions.get(horizon)
            if champion is not None:
                return champion
        fallback = self._fallbacks.get(horizon)
        if fallback is None:
            raise LookupError(f"No suitable model registered for {horizon}")
        return fallback

    def champion_horizons(self) -> frozenset[HorizonKey]:
        return frozenset(self._champions)
