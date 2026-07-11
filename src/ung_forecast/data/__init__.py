"""Market-data ingestion, validation, normalization, synchronization, and caching."""

from .cache import ParquetCache
from .factory import build_market_data_provider
from .provider import MarketDataProvider, YFinanceProvider
from .schemas import DataProvenance, MarketDataBundle
from .schwab import SchwabCredentials, SchwabMarketDataProvider, SchwabTokenProvider
from .synchronizer import SynchronizedMarketData
from .validator import DataValidationError, validate_ohlcv

__all__ = [
    "DataProvenance",
    "DataValidationError",
    "MarketDataBundle",
    "MarketDataProvider",
    "ParquetCache",
    "SchwabCredentials",
    "SchwabMarketDataProvider",
    "SchwabTokenProvider",
    "SynchronizedMarketData",
    "YFinanceProvider",
    "build_market_data_provider",
    "validate_ohlcv",
]
