from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from ung_forecast.training import historical_60m
from ung_forecast.training.historical_60m import (
    HistoricalRunConfig,
    HistoricalRunSummary,
    execute_historical_sixty_minute_run,
    load_historical_ung_data,
)


def _frame(rows: int = 80) -> pd.DataFrame:
    index = pd.date_range("2025-01-02 09:30", periods=rows, freq="5min", tz="UTC")
    close = pd.Series(range(100, 100 + rows), index=index, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1000.0,
        },
        index=index,
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
    frame = _frame(10).reset_index(names="timestamp")
    path = tmp_path / "short.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="minimum required"):
        load_historical_ung_data(
            HistoricalRunConfig(source="csv", input_path=path, minimum_rows=11)
        )


def test_summary_writes_machine_readable_json(tmp_path: Path) -> None:
    summary = HistoricalRunSummary(
        symbol="UNG",
        source="csv",
        rows=100,
        first_timestamp="a",
        last_timestamp="b",
        feature_rows=90,
        fold_count=2,
        statistically_approved=False,
        approval_reasons=("insufficient_samples",),
        aggregate_best_brier_model="unconditional",
        elastic_net_brier=0.7,
        elastic_net_log_loss=1.0,
        elastic_net_calibration_error=0.2,
        unconditional_brier=0.6,
        test_sample_count=60,
        selected_barriers=((1.0, 1.25),),
        artifact_directory="artifacts/models/60m/v1",
        manifest_path="artifacts/models/60m/manifest.json",
    )
    path = tmp_path / "summary.json"
    summary.write_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "UNG"
    assert payload["statistically_approved"] is False
    assert payload["selected_barriers"] == [[1.0, 1.25]]


def test_execution_reuses_feature_volatility_and_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _frame(80)
    feature_frame = pd.DataFrame(
        {"realized_volatility": 0.02, "signal": 1.0},
        index=market.index,
    )
    feature_result = SimpleNamespace(
        frame=feature_frame,
        assert_no_future_sources=lambda: None,
    )
    metrics = SimpleNamespace(
        elastic_net=SimpleNamespace(
            brier_score=0.4,
            log_loss=0.8,
            calibration_error=0.1,
            sample_count=20,
        ),
        unconditional=SimpleNamespace(brier_score=0.5),
    )
    report = SimpleNamespace(
        aggregate_metrics=metrics,
        fold_count=1,
        statistically_approved=False,
        approval_reasons=("research_only",),
        aggregate_best_brier_model="unconditional",
    )
    fold = SimpleNamespace(
        selected_barrier=SimpleNamespace(lower_multiplier=1.0, upper_multiplier=1.25)
    )
    artifact_directory = tmp_path / "models" / "60m" / "v1"
    result = cast(
        Any,
        SimpleNamespace(
            report=report,
            plan=SimpleNamespace(folds=(fold,)),
            artifact=SimpleNamespace(artifact_directory=artifact_directory),
        ),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(historical_60m, "load_historical_ung_data", lambda config: market)
    monkeypatch.setattr(historical_60m, "build_feature_frame", lambda *a, **k: feature_result)

    def fake_pipeline(
        market_data: pd.DataFrame,
        features: pd.DataFrame,
        volatility: pd.Series,
        **kwargs: object,
    ) -> Any:
        captured["market"] = market_data
        captured["features"] = features
        captured["volatility"] = volatility
        return result

    monkeypatch.setattr(historical_60m, "run_sixty_minute_research_pipeline", fake_pipeline)
    summary_path = tmp_path / "report.json"
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
            summary_path=summary_path,
        )
    )

    assert captured["market"] is market
    assert captured["features"] is feature_frame
    assert cast(pd.Series, captured["volatility"]).equals(feature_frame["realized_volatility"])
    assert summary.fold_count == 1
    assert summary.selected_barriers == ((1.0, 1.25),)
    assert summary_path.exists()
