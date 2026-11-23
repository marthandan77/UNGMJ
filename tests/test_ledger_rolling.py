import pandas as pd

from ung_forecast.ledger import ForecastLedger
from ung_forecast.rolling import generate_rolling_forecasts


def prices(n: int = 220) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(range(100, 100 + n), index=index, dtype=float)
    return pd.DataFrame({"open": close - 1, "high": close + 2, "low": close - 2, "close": close, "volume": 1000}, index=index)


def test_rolling_engine_creates_hgb_experts_and_five_dma_forecasts(tmp_path) -> None:
    ledger = ForecastLedger(tmp_path)
    records = generate_rolling_forecasts(prices(), model_names=("hist_gradient_boosting",), ledger=ledger)
    consensus = [row for row in records if row["model_name"] == "dma_consensus"]
    assert len(consensus) == 5
    assert [row["horizon"] for row in consensus] == [1, 2, 3, 4, 5]
    assert len(records) > 5
    assert len(ledger.forecasts()) == len(records)
    assert len(ledger.forecasts()) == len({row["forecast_id"] for row in ledger.forecasts()})


def test_matured_forecasts_are_scored_once(tmp_path) -> None:
    ledger = ForecastLedger(tmp_path)
    records = generate_rolling_forecasts(prices(), model_names=("hist_gradient_boosting",), ledger=ledger)
    bars = prices().copy()
    target = pd.Timestamp(next(row for row in records if row["horizon"] == 1)["target_session"])
    bars.loc[target] = [320, 325, 315, 322, 1000]
    expected_scored = sum(row["target_session"] == target.date().isoformat() for row in records)
    assert ledger.score_matured(bars) == expected_scored
    assert ledger.score_matured(bars) == 0
    assert len(ledger.outcomes()) == 1
