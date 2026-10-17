import pandas as pd
import pytest

from ung_forecast.context_data import combine_context_sources, make_point_in_time_context, parse_cftc_positioning
from ung_forecast.contracts import DataValidationError


def test_context_is_not_available_before_release() -> None:
    context = pd.DataFrame(
        {"storage_bcf": [1000], "available_at": [pd.Timestamp("2026-01-08 15:30")]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-02")], name="observation_date"),
    )
    sessions = pd.date_range("2026-01-05", periods=5, freq="B")
    result = make_point_in_time_context(context, sessions)
    assert pd.isna(result.loc[pd.Timestamp("2026-01-08"), "storage_bcf"])
    assert result.loc[pd.Timestamp("2026-01-09"), "storage_bcf"] == 1000


def test_context_requires_release_column() -> None:
    with pytest.raises(DataValidationError):
        make_point_in_time_context(pd.DataFrame({"value": [1]}), pd.date_range("2026-01-01", periods=1))


def test_cftc_parser_creates_net_positions_and_release_time() -> None:
    csv = "report_date,open_interest,managed_money_long,managed_money_short,producer_merchant_long,producer_merchant_short\n2026-01-06,100,60,20,30,50\n"
    result = parse_cftc_positioning(csv)
    assert result.iloc[0]["managed_money_net"] == 40
    assert result.iloc[0]["producer_merchant_net"] == -20
    assert result.iloc[0]["available_at"] > pd.Timestamp("2026-01-06")


def test_context_sources_combine_release_columns() -> None:
    first = pd.DataFrame({"a": [1], "available_at": [pd.Timestamp("2026-01-08 15:30")]}, index=[pd.Timestamp("2026-01-02")])
    second = pd.DataFrame({"b": [2], "available_at": [pd.Timestamp("2026-01-09 15:30")]}, index=[pd.Timestamp("2026-01-02")])
    combined = combine_context_sources([first, second])
    assert combined.loc[pd.Timestamp("2026-01-02"), "available_at"] == pd.Timestamp("2026-01-09 15:30")
