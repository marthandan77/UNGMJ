"""Parquet persistence for validated market data and provenance."""

from __future__ import annotations

import json
from pathlib import Path

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
        metadata_path.write_text(
            json.dumps(provenance.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return MarketDataBundle(frame=frame, provenance=provenance)

    def read(self, symbol: str, interval: str) -> MarketDataBundle | None:
        data_path, metadata_path = self._paths(symbol, interval)
        if not data_path.exists() or not metadata_path.exists():
            return None

        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        provenance = DataProvenance.model_validate(raw_metadata)
        frame = pd.read_parquet(data_path)
        return MarketDataBundle(frame=frame, provenance=provenance)
