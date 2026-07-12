"""Immutable model artifact manifest with validation metadata and checksums."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ung_forecast.horizons import HorizonKey


class StatisticalMetricsSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    brier_score: float = Field(ge=0.0)
    log_loss: float = Field(ge=0.0)
    calibration_error: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(gt=0)
    baseline_brier_score: float = Field(ge=0.0)
    baseline_log_loss: float = Field(ge=0.0)


class ArtifactFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def path_is_relative_and_safe(self) -> "ArtifactFile":
        if self.relative_path.startswith(("/", "\\")):
            raise ValueError("Artifact paths must be relative")
        parts = self.relative_path.replace("\\", "/").split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("Artifact paths must not contain empty, dot, or parent segments")
        return self


class HorizonArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default="1", pattern=r"^1$")
    horizon: HorizonKey
    created_at: datetime
    model_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    configuration_hash: str = Field(min_length=8)
    feature_names: tuple[str, ...] = Field(min_length=1)
    includes_comparison: bool = False
    statistical_approved: bool
    approval_reasons: tuple[str, ...] = ()
    metrics: StatisticalMetricsSnapshot
    model_file: ArtifactFile
    calibrator_file: ArtifactFile

    @model_validator(mode="after")
    def approval_state_is_consistent(self) -> "HorizonArtifactManifest":
        if self.statistical_approved and self.approval_reasons:
            raise ValueError("Approved artifacts cannot contain approval failure reasons")
        if not self.statistical_approved and not self.approval_reasons:
            raise ValueError("Research-only artifacts require at least one approval reason")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique and ordered")
        return self
