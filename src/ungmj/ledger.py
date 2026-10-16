"""Append-only forecast vintages and idempotent outcome scoring."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .contracts import ForecastVintage, DataValidationError


class ForecastLedger:
    """JSONL ledger. Existing forecast IDs cannot be overwritten or silently duplicated."""

    def __init__(self, root: str | Path = "data/ledger") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.forecasts_path = self.root / "forecasts.jsonl"
        self.outcomes_path = self.root / "outcomes.jsonl"

    @staticmethod
    def _read_ids(path: Path, field: str) -> set[str]:
        if not path.exists():
            return set()
        ids: set[str] = set()
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    ids.add(json.loads(line)[field])
        return ids

    def append_forecast(self, forecast: ForecastVintage) -> None:
        known = self._read_ids(self.forecasts_path, "forecast_id")
        if forecast.forecast_id in known:
            raise DataValidationError(f"forecast_id already exists: {forecast.forecast_id}")
        payload = asdict(forecast)
        payload["as_of"] = forecast.as_of.isoformat()
        payload["origin_session"] = forecast.origin_session.isoformat()
        self._append(self.forecasts_path, payload)

    def score_matured(self, bars: pd.DataFrame, *, now: datetime | None = None) -> int:
        """Score vintages only after the target horizon has elapsed; each ID scored once."""
        if not self.forecasts_path.exists():
            return 0
        now = now or datetime.now(timezone.utc)
        existing = self._read_ids(self.outcomes_path, "forecast_id")
        outcomes: list[dict[str, object]] = []
        forecasts = [json.loads(line) for line in self.forecasts_path.read_text().splitlines() if line.strip()]
        sessions = pd.DatetimeIndex(pd.to_datetime(bars.index)).normalize()
        for forecast in forecasts:
            fid = forecast["forecast_id"]
            if fid in existing:
                continue
            origin_session = pd.Timestamp(forecast["origin_session"]).normalize()
            eligible = sessions[sessions > origin_session]
            horizon = int(forecast["horizon_sessions"])
            if len(eligible) < horizon:
                continue
            target_session = eligible[horizon - 1]
            # Never score before the actual target session has closed.
            if pd.Timestamp(now.date()) < target_session:
                continue
            reference = float(forecast["reference_close"])
            target_close = float(bars.loc[target_session, "close"])
            realized_return = target_close / reference - 1.0
            outcomes.append({
                "forecast_id": fid,
                "scored_at": now.astimezone(timezone.utc).isoformat(),
                "target_session": target_session.date().isoformat(),
                "realized_close": target_close,
                "realized_return": realized_return,
                "direction_correct": (float(forecast["probability_up"]) >= 0.5) == (realized_return > 0),
                "absolute_return_error": abs(float(forecast["expected_return"]) - realized_return),
            })
        for outcome in outcomes:
            self._append(self.outcomes_path, outcome)
        return len(outcomes)

    @staticmethod
    def _append(path: Path, payload: dict[str, object]) -> None:
        # Single-writer workflow contract; fsync before returning.
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
