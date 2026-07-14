"""Parquet persistence for validated market data and provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import DataProvenance, MarketDataBundle


class ParquetCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        return symbol.replace("=", "_").replace("^", "_").replace("/", "_")

    def _paths(self, symbol: str, interval: str) -> tuple[Path, Path]:
        stem = f"{self._safe_symbol(symbol)}_{interval}"
        return self.root / f"{stem}.parquet", self.root / f"{stem}.metadata.json"

    def write(self, bundle: MarketDataBundle) -> MarketDataBundle:
        data_path, metadata_path = self._paths(
            bundle.provenance.symbol,
            bundle.provenance.interval,
        )
        frame = bundle.frame.sort_index().copy()
        frame.to_parquet(data_path)

        provenance = bundle.provenance.model_copy(update={"cache_path": data_path})
        index_timezone = (
            str(frame.index.tz)
            if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None
            else None
        )
        metadata = {
            "provenance": provenance.model_dump(mode="json"),
            "index_timezone": index_timezone,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return MarketDataBundle(frame=frame, provenance=provenance)

    def read(self, symbol: str, interval: str) -> MarketDataBundle | None:
        data_path, metadata_path = self._paths(symbol, interval)
        if not data_path.exists() or not metadata_path.exists():
            return None

        raw_metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        provenance_payload = raw_metadata.get("provenance", raw_metadata)
        provenance = DataProvenance.model_validate(provenance_payload)
        frame = pd.read_parquet(data_path)
        index_timezone = raw_metadata.get("index_timezone")
        if (
            isinstance(index_timezone, str)
            and isinstance(frame.index, pd.DatetimeIndex)
            and frame.index.tz is not None
        ):
            frame.index = frame.index.tz_convert(index_timezone)
        return MarketDataBundle(frame=frame, provenance=provenance)
