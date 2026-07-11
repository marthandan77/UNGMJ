"""Persist fitted model bundles with cryptographic metadata checksums."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from ung_forecast.horizons import HorizonKey


@dataclass(frozen=True, slots=True)
class ModelArtifactMetadata:
    horizon: HorizonKey
    model_version: str
    feature_version: str
    data_version: str
    configuration_hash: str
    created_at: datetime
    approved: bool
    includes_comparison: bool
    validation_metrics: dict[str, float | int]
    artifact_sha256: str = ""


class ModelArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, metadata: ModelArtifactMetadata) -> Path:
        return self.root / metadata.horizon.value / metadata.model_version

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def write(self, bundle: Any, metadata: ModelArtifactMetadata) -> ModelArtifactMetadata:
        directory = self._directory(metadata)
        directory.mkdir(parents=True, exist_ok=False)
        artifact_path = directory / "model.joblib"
        metadata_path = directory / "metadata.json"
        joblib.dump(bundle, artifact_path)
        checksum = self._sha256(artifact_path)
        finalized = ModelArtifactMetadata(**(asdict(metadata) | {"artifact_sha256": checksum}))
        metadata_path.write_text(
            json.dumps(asdict(finalized), default=str, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return finalized

    def load(self, horizon: HorizonKey, model_version: str) -> tuple[Any, ModelArtifactMetadata]:
        directory = self.root / horizon.value / model_version
        artifact_path = directory / "model.joblib"
        metadata_path = directory / "metadata.json"
        if not artifact_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {horizon.value}/{model_version}")
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = ModelArtifactMetadata(
            horizon=HorizonKey(raw["horizon"]),
            model_version=str(raw["model_version"]),
            feature_version=str(raw["feature_version"]),
            data_version=str(raw["data_version"]),
            configuration_hash=str(raw["configuration_hash"]),
            created_at=datetime.fromisoformat(raw["created_at"]),
            approved=bool(raw["approved"]),
            includes_comparison=bool(raw["includes_comparison"]),
            validation_metrics=dict(raw["validation_metrics"]),
            artifact_sha256=str(raw["artifact_sha256"]),
        )
        actual_checksum = self._sha256(artifact_path)
        if actual_checksum != metadata.artifact_sha256:
            raise ValueError("Model artifact checksum mismatch")
        return joblib.load(artifact_path), metadata
