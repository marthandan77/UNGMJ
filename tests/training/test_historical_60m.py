from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from ung_forecast.training import historical_60m
from ung_forecast.training.historical_60m import (
    FoldDiagnosticSummary,
    HistoricalRunConfig,
    HistoricalRunSummary,
    execute_historical_sixty_minute_run,
    load_historical_ung_data,
)


def _frame(rows: int = 80) -> pd.DataFrame:
    index = pd.date_range("2025-01-02 09:30", periods=rows, freq="5min", tz="UTC")
    close = pd.Series(range(100, 100 + rows), index=index, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1000.0},
        index=index,
    )


def _diagnostic() -> FoldDiagnosticSummary:
    return FoldDiagnosticSummary(
        fold_number=1,
        lower_multiplier=1.0,
        upper_multiplier=1.25,
        training_class_counts={"LOWER_FIRST": 10, "UPPER_FIRST": 10, "NEITHER": 10},
        validation_class_counts={"LOWER_FIRST": 3, "UPPER_FIRST": 3, "NEITHER": 4},
        test_class_counts={"LOWER_FIRST": 3, "UPPER_FIRST": 3, "NEITHER": 4},
        calibration_fit_rows=5,
        calibration_selection_rows=5,
        plain_probability_mode="calibrated",
        elastic_probability_mode="raw",
        unconditional_brier=0.50,
        recency_weighted_brier=0.48,
        plain_logistic_raw_brier=0.45,
        plain_logistic_selected_brier=0.44,
        plain_logistic_calibrated_brier=0.44,
        elastic_net_raw_brier=0.42,
        elastic_net_selected_brier=0.42,
        elastic_net_calibrated_brier=0.43,
        plain_selection_raw_brier=0.46,
        plain_selection_calibrated_brier=0.43,
        elastic_selection_raw_brier=0.41,
        elastic_selection_calibrated_brier=0.44,
        plain_calibration_brier_delta=-0.01,
        elastic_calibration_brier_delta=0.01,
        best_brier_model="elastic_net_raw",
    )


def test_csv_loader_normalizes_columns_timezone_and_ignores_live_staleness(tmp_path: Path) -> None:
    frame = _frame(20).reset_index(names="timestamp")
    frame.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    path = tmp_path / "ung.csv"
    frame.to_csv(path, index=False)
    loaded = load_historical_ung_data(
        HistoricalRunConfig(source="csv", input_path=path, minimum_rows=20)
    )
    assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.tz is not None
    assert len(loaded) == 20


def test_rejects_historical_dataset_below_minimum_rows(tmp_path: Path) -> None:
    path = tmp_path / "short.csv"
    _frame(10).reset_index(names="timestamp").to_csv(path, index=False)
    with pytest.raises(ValueError, match="minimum required"):
        load_historical_ung_data(
            HistoricalRunConfig(source="csv", input_path=path, minimum_rows=11)
        )


def test_summary_writes_calibration_selection_diagnostics(tmp_path: Path) -> None:
    summary = HistoricalRunSummary(
        symbol="UNG",
        source="csv",
        rows=100,
        first_timestamp="a",
        last_timestamp="b",
        feature_rows=90,
        fold_count=1,
        statistically_approved=False,
        approval_reasons=("insufficient_samples",),
        aggregate_best_brier_model="elastic_net_raw",
        elastic_net_brier=0.42,
        elastic_net_log_loss=1.0,
        elastic_net_calibration_error=0.2,
        unconditional_brier=0.6,
        test_sample_count=60,
        selected_barriers=((1.0, 1.25),),
        fold_diagnostics=(_diagnostic(),),
        artifact_directory="artifacts/models/60m/v1",
        manifest_path="artifacts/models/60m/manifest.json",
    )
    path = tmp_path / "summary.json"
    summary.write_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    diagnostic = payload["fold_diagnostics"][0]
    assert diagnostic["elastic_probability_mode"] == "raw"
    assert diagnostic["calibration_fit_rows"] == 5
    assert diagnostic["test_class_counts"]["NEITHER"] == 4


def test_execution_price_scales_volatility_and_serializes_selected_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _frame()
    features = pd.DataFrame({"realized_volatility": 0.02, "signal": 1.0}, index=market.index)
    feature_result = SimpleNamespace(frame=features, assert_no_future_sources=lambda: None)
    metrics = SimpleNamespace(
        unconditional=SimpleNamespace(brier_score=0.50),
        recency_weighted=SimpleNamespace(brier_score=0.48),
        plain_logistic_raw=SimpleNamespace(brier_score=0.45),
        plain_logistic=SimpleNamespace(brier_score=0.44),
        elastic_net_raw=SimpleNamespace(brier_score=0.42),
        elastic_net=SimpleNamespace(
            brier_score=0.42, log_loss=0.8, calibration_error=0.1, sample_count=20
        ),
    )
    counts = SimpleNamespace(lower_first=10, upper_first=10, neither=10)
    evaluated_fold = SimpleNamespace(
        fold_number=1,
        selected_lower_multiplier=1.0,
        selected_upper_multiplier=1.25,
        training_class_counts=counts,
        validation_class_counts=counts,
        test_class_counts=counts,
        metrics=metrics,
        plain_probability_mode="calibrated",
        elastic_probability_mode="raw",
        calibration_fit_rows=5,
        calibration_selection_rows=5,
        plain_selection_raw_brier=0.46,
        plain_selection_calibrated_brier=0.43,
        elastic_selection_raw_brier=0.41,
        elastic_selection_calibrated_brier=0.44,
        plain_calibrated_test_brier=0.44,
        elastic_calibrated_test_brier=0.43,
        plain_calibration_brier_delta=-0.01,
        elastic_calibration_brier_delta=0.01,
        best_brier_model="elastic_net_raw",
    )
    artifact_directory = tmp_path / "models/60m/v1"
    result = cast(
        Any,
        SimpleNamespace(
            report=SimpleNamespace(
                aggregate_metrics=metrics,
                fold_count=1,
                statistically_approved=False,
                approval_reasons=("research_only",),
                aggregate_best_brier_model="elastic_net_raw",
            ),
            plan=SimpleNamespace(
                folds=(SimpleNamespace(selected_barrier=SimpleNamespace(lower_multiplier=1.0, upper_multiplier=1.25)),)
            ),
            evaluation=SimpleNamespace(folds=(evaluated_fold,)),
            artifact=SimpleNamespace(artifact_directory=artifact_directory),
        ),
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(historical_60m, "load_historical_ung_data", lambda config: market)
    monkeypatch.setattr(historical_60m, "build_feature_frame", lambda *args, **kwargs: feature_result)

    def fake_pipeline(
        market_data: pd.DataFrame,
        feature_frame: pd.DataFrame,
        volatility: pd.Series,
        **kwargs: object,
    ) -> Any:
        captured["volatility"] = volatility
        return result

    monkeypatch.setattr(historical_60m, "run_sixty_minute_research_pipeline", fake_pipeline)
    _, summary = execute_historical_sixty_minute_run(
        HistoricalRunConfig(
            source="yfinance",
            minimum_rows=20,
            minimum_train_size=30,
            validation_size=10,
            test_size=10,
            purge_size=1,
            embargo_size=1,
            calibration_rows=10,
            artifact_root=tmp_path / "models",
            summary_path=tmp_path / "report.json",
        )
    )
    expected = (market["Close"] * features["realized_volatility"]).rename(
        "price_scaled_volatility"
    )
    pd.testing.assert_series_equal(cast(pd.Series, captured["volatility"]), expected)
    assert summary.fold_diagnostics[0].elastic_probability_mode == "raw"
