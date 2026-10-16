"""Build a static, presentation-only dashboard from generated ledger files."""
from __future__ import annotations
import json
from pathlib import Path

def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main() -> None:
    root = Path("data/ledger")
    site = Path("site")
    site.mkdir(parents=True, exist_ok=True)
    report_path = root / "latest-run.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
        "status": "BLOCKED", "error": "No pipeline run has produced a report."
    }
    forecasts = read_jsonl(root / "forecasts.jsonl")
    outcomes = read_jsonl(root / "outcomes.jsonl")
    site.joinpath("latest-run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    site.joinpath("forecasts.json").write_text(json.dumps(forecasts[-100:], indent=2) + "\n", encoding="utf-8")
    site.joinpath("outcomes.json").write_text(json.dumps(outcomes[-500:], indent=2) + "\n", encoding="utf-8")
    site.joinpath("dashboard-meta.json").write_text(json.dumps({
        "forecast_vintages_available": len(forecasts),
        "matured_outcomes_available": len(outcomes),
        "presentation_only": True,
    }, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
