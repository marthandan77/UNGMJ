"""Immutable records for forecasts and verified outcomes."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ung_forecast.explanations.renderer import Explanation, UserIntent
from ung_forecast.schemas import HorizonForecast, OutcomeClass


class ForecastRecordStatus(StrEnum):
    PENDING = "PENDING"
    SCORED = "SCORED"
    INVALID = "INVALID"


class ScoredOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: OutcomeClass
    scored_at: datetime
    event_timestamp: datetime | None
    time_to_event_bars: int
    maximum_upward_excursion: float
    maximum_downward_excursion: float


class ForecastRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(default_factory=lambda: uuid4().hex, min_length=8)
    created_at: datetime
    expires_at: datetime
    forecast: HorizonForecast
    user_intent: UserIntent
    explanation: Explanation
    status: ForecastRecordStatus = ForecastRecordStatus.PENDING
    outcome: ScoredOutcome | None = None

    def model_post_init(self, __context: object) -> None:
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        if self.status is ForecastRecordStatus.SCORED and self.outcome is None:
            raise ValueError("Scored records require an outcome")
        if self.status is ForecastRecordStatus.PENDING and self.outcome is not None:
            raise ValueError("Pending records cannot already contain an outcome")
