"""Append-only forecast and outcome ledgers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .models import score_forecast

LEDGER_DATA_BUDGET_BYTES = 8 * 1024 * 1024


class ForecastLedger:
    def __init__(self, directory: str | Path = "data/ledger") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.forecasts_path = self.directory / "forecasts.jsonl"
        self.outcomes_path = self.directory / "outcomes.jsonl"
        self.outside_rth_path = self.directory / "outside-rth-features.jsonl"
        self._forecasts_cache: list[dict[str, Any]] | None = None
        self._outcomes_cache: list[dict[str, Any]] | None = None
        self._outside_rth_cache: list[dict[str, Any]] | None = None

    def _read(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")

    def forecasts(self) -> list[dict[str, Any]]:
        if self._forecasts_cache is None:
            self._forecasts_cache = self._read(self.forecasts_path)
        return self._forecasts_cache

    def outcomes(self) -> list[dict[str, Any]]:
        if self._outcomes_cache is None:
            self._outcomes_cache = self._read(self.outcomes_path)
        return self._outcomes_cache

    def outside_rth_features(self) -> list[dict[str, Any]]:
        if self._outside_rth_cache is None:
            self._outside_rth_cache = self._read(self.outside_rth_path)
        return self._outside_rth_cache

    def append_outside_rth_feature(self, record: dict[str, Any]) -> bool:
        """Persist each scheduled premarket snapshot once per target session and slot."""
        existing = {
            (str(row["target_session"]), str(row.get("snapshot_slot", "one_hour")))
            for row in self.outside_rth_features()
            if row.get("source") == "alpaca_sip_1m_extended"
        }
        key = (str(record["target_session"]), str(record.get("snapshot_slot", "one_hour")))
        if key in existing:
            return False
        with self.outside_rth_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
        self.outside_rth_features().append(record)
        return True

    def append_forecast(self, record: dict[str, Any]) -> bool:
        """Append once by forecast_id; return False for an existing vintage."""
        existing = {str(row["forecast_id"]) for row in self.forecasts()}
        if str(record["forecast_id"]) in existing:
            return False
        with self.forecasts_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
        self.forecasts().append(record)
        return True

    def score_matured(self, bars: pd.DataFrame) -> int:
        """Score each forecast once when its target session exists in completed bars."""
        existing = {str(row["forecast_id"]) for row in self.outcomes()}
        rows: list[dict[str, Any]] = []
        for forecast in self.forecasts():
            forecast_id = str(forecast["forecast_id"])
            target = pd.Timestamp(forecast["target_session"]).normalize()
            if forecast_id in existing or target not in bars.index:
                continue
            actual = bars.loc[target]
            result = score_forecast(forecast, float(actual.low), float(actual.close), float(actual.high))
            rows.append({
                "forecast_id": forecast_id,
                "scored_at": datetime.now(timezone.utc).isoformat(),
                "target_session": target.date().isoformat(),
                "model_name": forecast["model_name"],
                "horizon": forecast["horizon"],
                "reference_close": float(forecast.get("reference_close", 1.0)),
                **result,
            })
        if rows:
            with self.outcomes_path.open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
            self.outcomes().extend(rows)
        return len(rows)

    def purge(
        self, *, max_age_days: int = 365, keep_per_model: int = 400,
        max_outside_rth_records: int = 1200,
    ) -> dict[str, int]:
        """Bound tracked learning data; do not create duplicate monthly archives."""
        cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=max_age_days)

        def retain(rows: list[dict[str, Any]], date_keys: tuple[str, ...]) -> list[dict[str, Any]]:
            groups: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                groups.setdefault(str(row.get("model_name", "unknown")), []).append(row)
            kept: list[dict[str, Any]] = []
            for group in groups.values():
                recent = []
                for row in group:
                    value = next((row.get(key) for key in date_keys if row.get(key)), None)
                    try:
                        if value and pd.Timestamp(value).tz_localize(None) >= cutoff:
                            recent.append(row)
                    except (TypeError, ValueError):
                        recent.append(row)
                kept.extend(recent[-keep_per_model:])
            return kept

        forecasts, outcomes = self.forecasts(), self.outcomes()
        original_forecast_count, original_outcome_count = len(forecasts), len(outcomes)
        supported = lambda row: (
            row.get("model_name") in {"hist_gradient_boosting", "dma_consensus"}
            or str(row.get("model_name", "")).startswith("hgb_expert_")
        )
        forecasts = [row for row in forecasts if supported(row)]
        outcomes = [row for row in outcomes if supported(row)]
        kept_forecasts = retain(forecasts, ("issued_at", "target_session"))
        kept_outcomes = retain(outcomes, ("scored_at", "target_session"))
        for path, rows in ((self.forecasts_path, kept_forecasts), (self.outcomes_path, kept_outcomes)):
            self._write_jsonl(path, rows)
        outside_rows = self.outside_rth_features()
        eligible_outside = [
            row for row in outside_rows
            if row.get("source") == "alpaca_sip_1m_extended"
        ][-max_outside_rth_records:]
        self._write_jsonl(self.outside_rth_path, eligible_outside)
        self._forecasts_cache, self._outcomes_cache = kept_forecasts, kept_outcomes
        self._outside_rth_cache = eligible_outside
        ledger_bytes = sum(
            path.stat().st_size for path in self.directory.rglob("*")
            if path.is_file() and path.suffix in {".json", ".jsonl"}
        )
        if ledger_bytes > LEDGER_DATA_BUDGET_BYTES:
            raise RuntimeError(
                f"tracked learning data is {ledger_bytes} bytes, above the 8 MiB ledger budget"
            )
        return {
            "forecasts_removed": original_forecast_count - len(kept_forecasts),
            "outcomes_removed": original_outcome_count - len(kept_outcomes),
            "outside_rth_removed": len(outside_rows) - len(eligible_outside),
            "ledger_bytes": ledger_bytes,
        }
