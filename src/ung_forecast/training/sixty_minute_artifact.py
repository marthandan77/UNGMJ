"""Final fitting and runtime-compatible artifact production for the 60-minute horizon."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from ung_forecast.artifacts.loader import ArtifactState, discover_horizon_artifacts
from ung_forecast.artifacts.manifest import (
    ArtifactFile,
    HorizonArtifactManifest,
    StatisticalMetricsSnapshot,
)
from ung_forecast.artifacts.runtime import load_validated_artifacts
from ung_forecast.horizons import HorizonKey
from ung_forecast.models.calibration import CalibrationConfig, MulticlassProbabilityCalibrator
from ung_forecast.models.elastic_net import ElasticNetConfig, ElasticNetMultinomialModel
from ung_forecast.training.dataset import build_training_dataset
from ung_forecast.training.runner_60m import HORIZON_KEY, REQUIRED_5M_BARS, SixtyMinuteRunPlan
from ung_forecast.training.sixty_minute_evaluation import _probability_frame
from ung_forecast.training.sixty_minute_report import SixtyMinuteResearchReport
from ung_forecast.validation.metrics import CLASS_ORDER
from ung_forecast.validation.purge import purge_overlapping_training_rows


@dataclass(frozen=True, slots=True)
class SixtyMinuteFinalFitConfig:
    configuration_hash: str
    calibration_rows: int = 250
    calibration_method: str = "platt"
    model_version: str = "60m-research-v1"
    feature_version: str = "features-v1"
    data_version: str = "runtime"
    elastic_net: ElasticNetConfig | None = None

    def __post_init__(self) -> None:
        if len(self.configuration_hash) < 8:
            raise ValueError("configuration_hash must contain at least eight characters")
        if self.calibration_rows <= 0:
            raise ValueError("calibration_rows must be positive")
        if not self.model_version or not self.feature_version or not self.data_version:
            raise ValueError("Artifact version fields cannot be empty")


@dataclass(frozen=True, slots=True)
class SixtyMinuteArtifactResult:
    manifest: HorizonArtifactManifest
    artifact_directory: Path
    version_directory: Path
    training_rows: int
    calibration_rows: int
    selected_lower_multiplier: float
    selected_upper_multiplier: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_barrier(plan: SixtyMinuteRunPlan) -> tuple[float, float]:
    if not plan.folds:
        raise ValueError("Cannot select a final barrier from an empty plan")
    counts: dict[tuple[float, float], int] = {}
    for fold in plan.folds:
        key = (
            fold.selected_barrier.lower_multiplier,
            fold.selected_barrier.upper_multiplier,
        )
        counts[key] = counts.get(key, 0) + 1
    return min(counts, key=lambda key: (-counts[key], key[0], key[1]))


def _manifest(
    *,
    config: SixtyMinuteFinalFitConfig,
    report: SixtyMinuteResearchReport,
    feature_names: tuple[str, ...],
    model_file: ArtifactFile,
    calibrator_file: ArtifactFile,
) -> HorizonArtifactManifest:
    metrics = report.aggregate_metrics
    reasons = report.approval_reasons
    if not report.statistically_approved and not reasons:
        reasons = ("statistical_approval_not_granted",)
    return HorizonArtifactManifest(
        horizon=HorizonKey.MINUTES_60,
        created_at=datetime.now(timezone.utc),
        model_version=config.model_version,
        feature_version=config.feature_version,
        data_version=config.data_version,
        configuration_hash=config.configuration_hash,
        feature_names=feature_names,
        includes_comparison=False,
        statistical_approved=report.statistically_approved,
        approval_reasons=reasons,
        metrics=StatisticalMetricsSnapshot(
            brier_score=metrics.elastic_net.brier_score,
            log_loss=metrics.elastic_net.log_loss,
            calibration_error=metrics.elastic_net.calibration_error,
            sample_count=metrics.elastic_net.sample_count,
            baseline_brier_score=metrics.unconditional.brier_score,
            baseline_log_loss=metrics.unconditional.log_loss,
        ),
        model_file=model_file,
        calibrator_file=calibrator_file,
    )


def fit_and_write_sixty_minute_artifact(
    market_data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    volatility: pd.Series,
    *,
    plan: SixtyMinuteRunPlan,
    report: SixtyMinuteResearchReport,
    artifact_root: str | Path,
    config: SixtyMinuteFinalFitConfig,
) -> SixtyMinuteArtifactResult:
    """Fit with a held-out calibration tail and write the runtime manifest contract."""

    lower_multiplier, upper_multiplier = _selected_barrier(plan)
    dataset = build_training_dataset(
        market_data,
        feature_frame,
        volatility,
        required_bars=REQUIRED_5M_BARS,
        lower_multiplier=lower_multiplier,
        upper_multiplier=upper_multiplier,
        horizon_key=HORIZON_KEY,
    )
    if len(dataset.features) <= config.calibration_rows:
        raise ValueError("Insufficient rows for final training and calibration")

    calibration_index = pd.DatetimeIndex(dataset.features.index[-config.calibration_rows :])
    candidate_training_index = pd.DatetimeIndex(dataset.features.index[: -config.calibration_rows])
    training_index = purge_overlapping_training_rows(
        candidate_training_index,
        dataset.label_end_time,
        evaluation_start=calibration_index[0],
    )
    if training_index.empty:
        raise ValueError("Purging removed all final training rows")

    training_target = dataset.target.loc[training_index]
    calibration_target = dataset.target.loc[calibration_index]
    expected_classes = set(CLASS_ORDER)
    if set(training_target.astype(str).unique()) != expected_classes:
        raise ValueError("Final training block does not contain all classes")
    if set(calibration_target.astype(str).unique()) != expected_classes:
        raise ValueError("Final calibration block does not contain all classes")

    model = ElasticNetMultinomialModel(config.elastic_net)
    model.fit(dataset.features.loc[training_index], training_target)
    calibration_raw = _probability_frame(
        model.predict_probabilities(dataset.features.loc[calibration_index]),
        calibration_index,
    )
    calibrator = MulticlassProbabilityCalibrator(CalibrationConfig(method=config.calibration_method))
    calibrator.fit(calibration_raw, calibration_target)

    artifact_directory = Path(artifact_root) / HorizonKey.MINUTES_60.value
    version_directory = artifact_directory / config.model_version
    version_directory.mkdir(parents=True, exist_ok=False)
    model_path = version_directory / "model.joblib"
    calibrator_path = version_directory / "calibrator.joblib"
    joblib.dump(model, model_path)
    joblib.dump(calibrator, calibrator_path)

    model_file = ArtifactFile(
        relative_path=f"{config.model_version}/model.joblib",
        sha256=_sha256(model_path),
    )
    calibrator_file = ArtifactFile(
        relative_path=f"{config.model_version}/calibrator.joblib",
        sha256=_sha256(calibrator_path),
    )
    manifest = _manifest(
        config=config,
        report=report,
        feature_names=tuple(dataset.features.columns),
        model_file=model_file,
        calibrator_file=calibrator_file,
    )
    manifest_path = artifact_directory / "manifest.json"
    temporary_manifest = artifact_directory / "manifest.json.tmp"
    temporary_manifest.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)

    discovered = discover_horizon_artifacts(
        artifact_root,
        expected_configuration_hash=config.configuration_hash,
        comparison_available=False,
    )[HorizonKey.MINUTES_60]
    expected_state = (
        ArtifactState.VALIDATED if report.statistically_approved else ArtifactState.RESEARCH_ONLY
    )
    if discovered.state is not expected_state:
        raise ValueError("Written artifact did not rediscover with the expected state")
    if report.statistically_approved:
        loaded = load_validated_artifacts(discovered)
        probe = dataset.features.loc[calibration_index[-min(5, len(calibration_index)) :]]
        original_raw = _probability_frame(model.predict_probabilities(probe), probe.index)
        loaded_raw = _probability_frame(loaded.model.predict_probabilities(probe), probe.index)
        original_probabilities = calibrator.transform(original_raw)
        loaded_probabilities = loaded.calibrator.transform(loaded_raw)
        if not original_probabilities.equals(loaded_probabilities):
            raise ValueError("Artifact reload inference parity check failed")

    return SixtyMinuteArtifactResult(
        manifest=manifest,
        artifact_directory=artifact_directory,
        version_directory=version_directory,
        training_rows=len(training_index),
        calibration_rows=len(calibration_index),
        selected_lower_multiplier=lower_multiplier,
        selected_upper_multiplier=upper_multiplier,
    )
