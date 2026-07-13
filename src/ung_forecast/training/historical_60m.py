"""Historical-data preparation and execution for the 60-minute research pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

from ung_forecast.configuration import load_config
from ung_forecast.data.provider import YFinanceProvider
from ung_forecast.data.validator import REQUIRED_COLUMNS, validate_ohlcv
from ung_forecast.features.engine import build_feature_frame
from ung_forecast.features.formulas import price_scaled_volatility
from ung_forecast.horizons import HorizonKey
from ung_forecast.training.barrier_selection import BarrierCandidate, BarrierSelectionConfig
from ung_forecast.training.pipeline_60m import (
    SixtyMinutePipelineConfig,
    SixtyMinutePipelineResult,
    run_sixty_minute_research_pipeline,
)
from ung_forecast.training.runner_60m import SixtyMinuteRunnerConfig
from ung_forecast.training.sixty_minute_artifact import SixtyMinuteFinalFitConfig
from ung_forecast.training.sixty_minute_evaluation import SixtyMinuteEvaluationConfig
from ung_forecast.validation.approval import StatisticalApprovalCriteria

HistoricalSource = Literal["yfinance", "csv", "parquet"]
DEFAULT_BARRIER_CANDIDATES = tuple(
    BarrierCandidate(lower, upper)
    for lower in (0.75, 1.0, 1.25, 1.5)
    for upper in (0.75, 1.0, 1.25, 1.5)
)


@dataclass(frozen=True, slots=True)
class HistoricalRunConfig:
    source: HistoricalSource = "yfinance"
    input_path: Path | None = None
    symbol: str = "UNG"
    period: str = "60d"
    timezone: str = "America/New_York"
    artifact_root: Path = Path("artifacts/models")
    summary_path: Path = Path("artifacts/reports/60m-latest.json")
    minimum_rows: int = 1500
    minimum_train_size: int = 1000
    validation_size: int = 300
    test_size: int = 300
    purge_size: int = 12
    embargo_size: int = 12
    calibration_rows: int = 250

    def __post_init__(self) -> None:
        if self.source in {"csv", "parquet"} and self.input_path is None:
            raise ValueError("input_path is required for CSV or Parquet sources")
        numeric = (
            self.minimum_rows,
            self.minimum_train_size,
            self.validation_size,
            self.test_size,
            self.purge_size,
            self.embargo_size,
            self.calibration_rows,
        )
        if any(value <= 0 for value in numeric):
            raise ValueError("Historical training row and fold sizes must be positive")


@dataclass(frozen=True, slots=True)
class HistoricalRunSummary:
    symbol: str
    source: str
    rows: int
    first_timestamp: str
    last_timestamp: str
    feature_rows: int
    fold_count: int
    statistically_approved: bool
    approval_reasons: tuple[str, ...]
    aggregate_best_brier_model: str
    elastic_net_brier: float
    elastic_net_log_loss: float
    elastic_net_calibration_error: float
    unconditional_brier: float
    test_sample_count: int
    selected_barriers: tuple[tuple[float, float], ...]
    artifact_directory: str
    manifest_path: str

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    missing = [name for name in REQUIRED_COLUMNS if name.lower() not in lookup]
    if missing:
        raise ValueError(f"Historical input is missing required OHLCV columns: {missing}")
    selected = frame[[lookup[name.lower()] for name in REQUIRED_COLUMNS]].copy()
    selected.columns = list(REQUIRED_COLUMNS)
    return selected


def _load_file(path: Path, *, source: HistoricalSource, timezone: str) -> pd.DataFrame:
    if source == "csv":
        raw = pd.read_csv(path)
    elif source == "parquet":
        raw = pd.read_parquet(path)
    else:
        raise ValueError("File loading supports only CSV and Parquet")

    if not isinstance(raw.index, pd.DatetimeIndex):
        timestamp_column = next(
            (
                column
                for column in raw.columns
                if str(column).strip().lower() in {"timestamp", "datetime", "date", "time"}
            ),
            None,
        )
        if timestamp_column is None:
            raise ValueError("Historical file requires a timestamp, datetime, date, or time column")
        timestamps = pd.to_datetime(raw.pop(timestamp_column), errors="raise", utc=True)
        raw.index = pd.DatetimeIndex(timestamps)

    frame = _canonical_columns(raw)
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        frame.index = index.tz_localize(timezone)
    latest = pd.Timestamp(frame.index[-1])
    as_of = latest.to_pydatetime() + timedelta(minutes=5, seconds=1)
    return validate_ohlcv(frame, interval="5m", as_of=as_of, timezone=timezone)


def load_historical_ung_data(config: HistoricalRunConfig) -> pd.DataFrame:
    if config.source == "yfinance":
        provider = YFinanceProvider(timezone=config.timezone, adjusted_prices=False)
        frame = provider.download_historical(
            symbol=config.symbol,
            interval="5m",
            period=config.period,
        ).frame
    else:
        assert config.input_path is not None
        frame = _load_file(config.input_path, source=config.source, timezone=config.timezone)
    if len(frame) < config.minimum_rows:
        raise ValueError(
            f"Historical dataset has {len(frame)} rows; minimum required is {config.minimum_rows}"
        )
    return frame


def _pipeline_config(config: HistoricalRunConfig) -> SixtyMinutePipelineConfig:
    application = load_config()
    return SixtyMinutePipelineConfig(
        runner=SixtyMinuteRunnerConfig(
            minimum_train_size=config.minimum_train_size,
            validation_size=config.validation_size,
            test_size=config.test_size,
            purge_size=config.purge_size,
            embargo_size=config.embargo_size,
            barrier_selection=BarrierSelectionConfig(
                candidates=DEFAULT_BARRIER_CANDIDATES,
                minimum_training_rows=max(300, config.minimum_train_size // 3),
                minimum_validation_rows=max(75, config.validation_size // 3),
                require_all_classes=True,
            ),
        ),
        evaluation=SixtyMinuteEvaluationConfig(),
        approval=StatisticalApprovalCriteria(),
        final_fit=SixtyMinuteFinalFitConfig(
            configuration_hash=application.quantitative_configuration_hash,
            calibration_rows=config.calibration_rows,
            model_version="60m-empirical-v1",
            feature_version="features-v1",
            data_version=f"{config.source}-{config.period}",
        ),
    )


def execute_historical_sixty_minute_run(
    config: HistoricalRunConfig,
) -> tuple[SixtyMinutePipelineResult, HistoricalRunSummary]:
    market_data = load_historical_ung_data(config)
    feature_result = build_feature_frame(market_data, horizon=HorizonKey.MINUTES_60)
    feature_result.assert_no_future_sources()
    volatility = price_scaled_volatility(
        market_data["Close"],
        feature_result.frame["realized_volatility"],
    )
    result = run_sixty_minute_research_pipeline(
        market_data,
        feature_result.frame,
        volatility,
        artifact_root=config.artifact_root,
        config=_pipeline_config(config),
    )
    aggregate = result.report.aggregate_metrics
    summary = HistoricalRunSummary(
        symbol=config.symbol,
        source=config.source,
        rows=len(market_data),
        first_timestamp=str(market_data.index[0]),
        last_timestamp=str(market_data.index[-1]),
        feature_rows=int(feature_result.frame.dropna().shape[0]),
        fold_count=result.report.fold_count,
        statistically_approved=result.report.statistically_approved,
        approval_reasons=result.report.approval_reasons,
        aggregate_best_brier_model=result.report.aggregate_best_brier_model,
        elastic_net_brier=aggregate.elastic_net.brier_score,
        elastic_net_log_loss=aggregate.elastic_net.log_loss,
        elastic_net_calibration_error=aggregate.elastic_net.calibration_error,
        unconditional_brier=aggregate.unconditional.brier_score,
        test_sample_count=aggregate.elastic_net.sample_count,
        selected_barriers=tuple(
            (
                fold.selected_barrier.lower_multiplier,
                fold.selected_barrier.upper_multiplier,
            )
            for fold in result.plan.folds
        ),
        artifact_directory=str(result.artifact.artifact_directory),
        manifest_path=str(result.artifact.artifact_directory.parent / "manifest.json"),
    )
    summary.write_json(config.summary_path)
    return result, summary
