"""Versioned, integrity-checked model artifact contracts."""

from .loader import (
    ArtifactLoadResult,
    ArtifactState,
    discover_horizon_artifacts,
    load_manifest,
)
from .manifest import ArtifactFile, HorizonArtifactManifest, StatisticalMetricsSnapshot
from .materialize import materialize_repository_artifacts
from .runtime import LoadedHorizonArtifacts, load_validated_artifacts

__all__ = [
    "ArtifactFile",
    "ArtifactLoadResult",
    "ArtifactState",
    "HorizonArtifactManifest",
    "LoadedHorizonArtifacts",
    "StatisticalMetricsSnapshot",
    "discover_horizon_artifacts",
    "load_manifest",
    "load_validated_artifacts",
    "materialize_repository_artifacts",
]
