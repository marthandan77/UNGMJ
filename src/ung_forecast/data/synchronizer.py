"""Strict timestamp alignment for related market-data series.

The synchronizer uses exact timestamp intersection and never forward-fills.
"""

from __future__ import annotations

import pandas as pd

from .schemas import MarketDataBundle
from .validator import DataValidationError


class SynchronizedMarketData:
    def __init__(self, primary: MarketDataBundle, comparison: MarketDataBundle) -> None:
        if primary.provenance.interval != comparison.provenance.interval:
            raise DataValidationError("Intervals must match before synchronization")
        self.primary = primary
        self.comparison = comparison

    def align(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        shared_index = self.primary.frame.index.intersection(self.comparison.frame.index)
        if shared_index.empty:
            raise DataValidationError("No synchronized timestamps are available")

        primary = self.primary.frame.loc[shared_index].copy()
        comparison = self.comparison.frame.loc[shared_index].copy()
        if not primary.index.equals(comparison.index):
            raise DataValidationError("Timestamp alignment failed")
        return primary, comparison
