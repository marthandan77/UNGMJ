"""Final research-only fitting and artifact production for the 60-minute horizon."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ung_forecast.horizons import HorizonKey
from ung_forecast.models.artifacts import ModelArtifactMetadata, ModelArtifactStore
from ung_forecast.models.calibration import CalibrationConfig, MulticlassProbabilityCalibrator
from ung_forecast.models.elastic_net import ElasticNetConfig, ElasticNetMultinomialModel
from ung_forecast.schemas import ProbabilityForecast
from ung_forecast.training.dataset import build_training_dataset
from ung_forecast.training.runner_60m import HORIZON_KEY, REQUIRED_5M_BARS, SixtyMinuteRunPlan
from ung_forecast.training.sixty_minute_evaluation import _probability_frame
from ung_forecast.training.sixty_minute_report import SixtyMinuteResearchReport
from ung_forecast.validation.metrics import CLASS_ORDER
from ung_forecast.validation.purge import purge_overlapping_training_rows


@dataclass(frozen=True, slots=True)
class SixtyMinuteFinalFitConfig:
    calibration_rows: int = 250
    calibration_method: str = "platt"
    model_version: str = "60m-research-v1"
    feature_version: str = "features-v1"
    data_version: str = "runtime"
    elastic_net: ElasticNetConfig | None = None

    def __post_init__(self) -> None:
        if self.calibration_rows <= 0:
            raise ValueError("calibration_rows must be positive")
        if not self.model_version or not self.feature_version or not self.data_version:
            raise ValueError("Artifact version fields cannot be empty")


@dataclass(slots=True)
class SixtyMinuteModelBundle:
    model: ElasticNetMultinomialModel
    calibrator: MulticlassProbabilityCalibrator
    lower_multiplier: float
    upper_multiplier: float
    feature_names: tuple[str, ...]
    horizon_key: str = HORIZON_KEY
    required_bars: int = REQUIRED_5M_BARS

    def predict_probabilities(self, features: pd.DataFrame) -> list[ProbabilityForecast]:
        if tuple(features.columns) != self.feature_names:
            raise ValueError("Feature schema does not match artifact")
        raw = _probability_frame(self.model.predict_probabilities(features), features.index)
        calibrated = self.calibrator.transform(raw)
        return [
            ProbabilityForecast(
                lower_first=float(row["LOWER_FIRST"]),
                upper_first=float(row["UPPER_FIRST"]),
                neither=float(row["NEITHER"]),
            )
            for _, row in calibrated.iterrows()
        ]


@dataclass(frozen=True, slots=True)
class SixtyMinuteArtifactResult:
    bundle: SixtyMinuteModelBundle
    metadata: ModelArtifactMetadata
    artifact_directory: Path
    training_rows: int
    calibration_rows: int


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


def _configuration_hash(config: SixtyMinuteFinalFitConfig, barrier: tuple[float, float]) -> str:
    payload = {
        "config": asdict(config),
        "barrier": barrier,
        "horizon": HORIZON_KEY,
        "required_bars": REQUIRED_5M_BARS,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validation_metrics(report: SixtyMinuteResearchReport) -> dict[str, float | int]:
    metrics = report.aggregate_metrics
    return {
        "fold_count": report.fold_count,
        "statistically_approved": int(report.statistically_approved),
        "elastic_net_brier": metrics.elastic_net.brier_score,
        "elastic_net_log_loss": metrics.elastic_net.log_loss,
        "elastic_net_calibration_error": metrics.elastic_net.calibration_error,
        "unconditional_brier": metrics.unconditional.brier_score,
        "test_sample_count": metrics.elastic_net.sample_count,
    }


def fit_and_write_sixty_minute_artifact(
    market_data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    volatility: pd.Series,
    *,
    plan: SixtyMinuteRunPlan,
    report: SixtyMinuteResearchReport,
    artifact_root: str | Path,
    config: SixtyMinuteFinalFitConfig | None = None,
) -> SixtyMinuteArtifactResult:
    """Fit a research bundle using a held-out calibration tail and persist it."""

    effective = config or SixtyMinuteFinalFitConfig()
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
    if len(dataset.features) <= effective.calibration_rows:
        raise ValueError("Insufficient rows for final training and calibration")

    calibration_index = pd.DatetimeIndex(dataset.features.index[-effective.calibration_rows :])
    candidate_training_index = pd.DatetimeIndex(
        dataset.features.index[: -effective.calibration_rows]
    )
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

    model = ElasticNetMultinomialModel(effective.elastic_net)
    model.fit(dataset.features.loc[training_index], training_target)
    calibration_raw = _probability_frame(
        model.predict_probabilities(dataset.features.loc[calibration_index]),
        calibration_index,
    )
    calibrator = MulticlassProbabilityCalibrator(
        CalibrationConfig(method=effective.calibration_method)
    )
    calibrator.fit(calibration_raw, calibration_target)

    feature_names = tuple(dataset.features.columns)
    bundle = SixtyMinuteModelBundle(
        model=model,
        calibrator=calibrator,
        lower_multiplier=lower_multiplier,
        upper_multiplier=upper_multiplier,
        feature_names=feature_names,
    )
    metadata = ModelArtifactMetadata(
        horizon=HorizonKey.MINUTES_60,
        model_version=effective.model_version,
        feature_version=effective.feature_version,
        data_version=effective.data_version,
        configuration_hash=_configuration_hash(effective, (lower_multiplier, upper_multiplier)),
        created_at=datetime.now(timezone.utc),
        approved=False,
        includes_comparison=True,
        validation_metrics=_validation_metrics(report),
    )
    store = ModelArtifactStore(artifact_root)
    finalized = store.write(bundle, metadata)
    directory = Path(artifact_root) / HorizonKey.MINUTES_60.value / effective.model_version
    reloaded, loaded_metadata = store.load(HorizonKey.MINUTES_60, effective.model_version)
    if not isinstance(reloaded, SixtyMinuteModelBundle):
        raise TypeError("Reloaded artifact has an unexpected bundle type")
    if loaded_metadata != finalized:
        raise ValueError("Reloaded metadata does not match written metadata")

    probe = dataset.features.loc[calibration_index[-min(5, len(calibration_index)) :]]
    original_probabilities = bundle.predict_probabilities(probe)
    reloaded_probabilities = reloaded.predict_probabilities(probe)
    if original_probabilities != reloaded_probabilities:
        raise ValueError("Artifact reload inference parity check failed")

    return SixtyMinuteArtifactResult(
        bundle=bundle,
        metadata=finalized,
        artifact_directory=directory,
        training_rows=len(training_index),
        calibration_rows=len(calibration_index),
    )
