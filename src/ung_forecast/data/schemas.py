"""Typed market-data contracts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class DataProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    provider: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    interval: str = Field(min_length=1)
    downloaded_at: datetime
    source_timezone: str = Field(min_length=1)
    normalized_timezone: str = Field(min_length=1)
    adjusted_prices: bool
    cache_path: Path | None = None


class MarketDataBundle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    frame: pd.DataFrame
    provenance: DataProvenance

    def model_post_init(self, __context: object) -> None:
        if self.frame.empty:
            raise ValueError("Market data frame cannot be empty")
