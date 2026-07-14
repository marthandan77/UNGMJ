"""Market-data ingestion, validation, normalization, synchronization, and caching."""

from .cache import ParquetCache
from .factory import build_market_data_provider
from .provider import MarketDataProvider, YFinanceProvider
from .runtime_loader import REQUIRED_INTERVALS, RuntimeDataSet, load_runtime_market_data
from .schemas import DataProvenance, MarketDataBundle
from .schwab import SchwabCredentials, SchwabMarketDataProvider, SchwabTokenProvider
from .synchronizer import SynchronizedMarketData
from .twelvedata import TwelveDataCredentials, TwelveDataMarketDataProvider
from .validator import DataValidationError, validate_ohlcv

__all__ = [
    "DataProvenance",
    "DataValidationError",
    "MarketDataBundle",
    "MarketDataProvider",
    "ParquetCache",
    "REQUIRED_INTERVALS",
    "RuntimeDataSet",
    "SchwabCredentials",
    "SchwabMarketDataProvider",
    "SchwabTokenProvider",
    "SynchronizedMarketData",
    "TwelveDataCredentials",
    "TwelveDataMarketDataProvider",
    "YFinanceProvider",
    "build_market_data_provider",
    "load_runtime_market_data",
    "validate_ohlcv",
]
