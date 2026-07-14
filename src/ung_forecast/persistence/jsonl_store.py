"""Append-only JSONL store for prototype forecast audit records.

The interface is intentionally simple so PostgreSQL can replace this adapter
without changing forecast or scoring logic.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import ForecastRecord


class JsonlForecastStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ForecastRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True))
            handle.write("\n")

    def read_all(self) -> list[ForecastRecord]:
        if not self.path.exists():
            return []
        records: list[ForecastRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    records.append(ForecastRecord.model_validate_json(payload))
                except ValueError as exc:
                    raise ValueError(f"Invalid forecast record at line {line_number}") from exc
        return records

    def latest_by_id(self) -> dict[str, ForecastRecord]:
        latest: dict[str, ForecastRecord] = {}
        for record in self.read_all():
            latest[record.record_id] = record
        return latest
