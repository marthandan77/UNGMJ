"""Load checksum-verified model objects for forecast-time use.

Joblib files must originate from this trusted repository or approved build output.
Checksums detect corruption; they do not make untrusted pickle content safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import joblib

from ung_forecast.models.calibration import MulticlassProbabilityCalibrator
from ung_forecast.models.elastic_net import ElasticNetMultinomialModel

from .loader import ArtifactLoadResult, ArtifactState, resolve_checked
from .manifest import HorizonArtifactManifest


@dataclass(frozen=True, slots=True)
class LoadedHorizonArtifacts:
    manifest: HorizonArtifactManifest
    model: ElasticNetMultinomialModel
    calibrator: MulticlassProbabilityCalibrator


def load_validated_artifacts(result: ArtifactLoadResult) -> LoadedHorizonArtifacts:
    if result.state is not ArtifactState.VALIDATED:
        raise ValueError("Only validated artifacts may be loaded for inference")
    if result.manifest is None or result.artifact_directory is None:
        raise ValueError("Validated artifact result is incomplete")

    model_path = resolve_checked(result.artifact_directory, result.manifest.model_file)
    calibrator_path = resolve_checked(result.artifact_directory, result.manifest.calibrator_file)
    model_object = joblib.load(model_path)
    calibrator_object = joblib.load(calibrator_path)
    if not isinstance(model_object, ElasticNetMultinomialModel):
        raise TypeError("Artifact model has an unexpected runtime type")
    if not isinstance(calibrator_object, MulticlassProbabilityCalibrator):
        raise TypeError("Artifact calibrator has an unexpected runtime type")
    if model_object.feature_names != result.manifest.feature_names:
        raise ValueError("Fitted model feature schema does not match manifest")
    return LoadedHorizonArtifacts(
        manifest=result.manifest,
        model=cast(ElasticNetMultinomialModel, model_object),
        calibrator=cast(MulticlassProbabilityCalibrator, calibrator_object),
    )
