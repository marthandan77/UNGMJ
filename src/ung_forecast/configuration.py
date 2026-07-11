"""Configuration loading with deterministic hashing for forecast reproducibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    primary_symbol: str = "UNG"
    comparison_symbol: str = "NG=F"
    timezone: str = "America/New_York"
    provider: str = "yfinance"
    adjusted_prices: bool = False
    cache_directory: Path = Path("data/cache")


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    application_name: str = "UNG Forecast Machine"
    research_mode: bool = True
    data: DataConfig = Field(default_factory=DataConfig)

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    return AppConfig.model_validate(raw)
