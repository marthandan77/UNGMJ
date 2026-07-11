"""Typed contracts exchanged between quantitative engines and presentation layers."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from math import isclose

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .horizons import HorizonKey


class OutcomeClass(StrEnum):
    LOWER_FIRST = "LOWER_FIRST"
    UPPER_FIRST = "UPPER_FIRST"
    NEITHER = "NEITHER"


class ModelStatus(StrEnum):
    VALIDATED = "VALIDATED"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    UNAVAILABLE = "UNAVAILABLE"


class DataFreshness(StrEnum):
    CURRENT = "CURRENT"
    STALE = "STALE"
    INVALID = "INVALID"


class ProbabilityForecast(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lower_first: float = Field(ge=0.0, le=1.0)
    upper_first: float = Field(ge=0.0, le=1.0)
    neither: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> "ProbabilityForecast":
        total = self.lower_first + self.upper_first + self.neither
        if not isclose(total, 1.0, abs_tol=1e-8):
            raise ValueError(f"Outcome probabilities must sum to one; received {total}")
        return self


class ModelIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    configuration_hash: str = Field(min_length=8)


class HorizonForecast(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    horizon: HorizonKey
    current_price: float = Field(gt=0.0)
    lower_price_area: float = Field(gt=0.0)
    upper_price_area: float = Field(gt=0.0)
    probabilities: ProbabilityForecast
    sell_expected_value: float
    buy_expected_value: float
    sell_ev_lower_confidence_bound: float
    buy_ev_lower_confidence_bound: float
    confidence_score: float = Field(ge=0.0, le=1.0)
    model_status: ModelStatus
    data_freshness: DataFreshness
    identity: ModelIdentity

    @model_validator(mode="after")
    def price_areas_are_ordered(self) -> "HorizonForecast":
        if not self.lower_price_area < self.current_price < self.upper_price_area:
            raise ValueError("Price areas must satisfy lower < current < upper")
        return self
