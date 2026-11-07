"""Append-only forecast and outcome ledgers."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .models import score_forecast


class ForecastLedger:
    def __init__(self, directory: str | Path = "data/ledger") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.forecasts_path = self.directory / "forecasts.jsonl"
        self.outcomes_path = self.directory / "outcomes.jsonl"
        self._forecasts_cache: list[dict[str, Any]] | None = None
        self._outcomes_cache: list[dict[str, Any]] | None = None

    def _read(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def forecasts(self) -> list[dict[str, Any]]:
        if self._forecasts_cache is None:
            self._forecasts_cache = self._read(self.forecasts_path)
        return self._forecasts_cache

    def outcomes(self) -> list[dict[str, Any]]:
        if self._outcomes_cache is None:
            self._outcomes_cache = self._read(self.outcomes_path)
        return self._outcomes_cache

    def append_forecast(self, record: dict[str, Any]) -> bool:
        """Append once by forecast_id; return False for an existing vintage."""
        existing = {str(row["forecast_id"]) for row in self.forecasts()}
        if str(record["forecast_id"]) in existing:
            return False
        with self.forecasts_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
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
                **result,
            })
        if rows:
            with self.outcomes_path.open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            self.outcomes().extend(rows)
        return len(rows)

    def purge(self, *, max_age_days: int = 730, keep_per_model: int = 1000) -> dict[str, int]:
        """Retain recent ledger evidence and rewrite obsolete rows."""
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
        kept_forecasts = retain(forecasts, ("issued_at", "target_session"))
        kept_outcomes = retain(outcomes, ("scored_at", "target_session"))
        archive_dir = self.directory / "archive" / pd.Timestamp.now(tz="UTC").strftime("%Y-%m")
        archive_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.forecasts_path, self.outcomes_path):
            archive_path = archive_dir / path.name
            if path.exists() and not archive_path.exists():
                shutil.copy2(path, archive_path)
        for path, rows in ((self.forecasts_path, kept_forecasts), (self.outcomes_path, kept_outcomes)):
            with path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self._forecasts_cache, self._outcomes_cache = kept_forecasts, kept_outcomes
        return {"forecasts_removed": len(forecasts) - len(kept_forecasts), "outcomes_removed": len(outcomes) - len(kept_outcomes)}
