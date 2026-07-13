"""Run the complete historical 60-minute UNG research pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ung_forecast.training.historical_60m import (
    HistoricalRunConfig,
    execute_historical_sixty_minute_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("yfinance", "csv", "parquet"), default="yfinance")
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--symbol", default="UNG")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/models"))
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("artifacts/reports/60m-latest.json"),
    )
    parser.add_argument("--minimum-rows", type=int, default=1500)
    parser.add_argument("--minimum-train-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--purge-size", type=int, default=12)
    parser.add_argument("--embargo-size", type=int, default=12)
    parser.add_argument("--calibration-rows", type=int, default=250)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = HistoricalRunConfig(
        source=args.source,
        input_path=args.input_path,
        symbol=args.symbol,
        period=args.period,
        timezone=args.timezone,
        artifact_root=args.artifact_root,
        summary_path=args.summary_path,
        minimum_rows=args.minimum_rows,
        minimum_train_size=args.minimum_train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        purge_size=args.purge_size,
        embargo_size=args.embargo_size,
        calibration_rows=args.calibration_rows,
    )
    _, summary = execute_historical_sixty_minute_run(config)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
