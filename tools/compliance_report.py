"""Generate a machine-readable and Markdown build-compliance report."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROW_PATTERN = re.compile(r"^\|\s*(AC-[^|]+)\|[^|]*\|[^|]*\|\s*([A-Z_]+)\s*\|$")


@dataclass(frozen=True, slots=True)
class ComplianceSummary:
    total_requirements: int
    tested: int
    implemented: int
    pending: int
    blocked: int
    unknown: int

    @property
    def complete(self) -> bool:
        return self.total_requirements > 0 and self.pending == 0 and self.blocked == 0 and self.unknown == 0


def parse_traceability(path: str | Path) -> ComplianceSummary:
    counts = {"TESTED": 0, "IMPLEMENTED": 0, "PENDING": 0, "BLOCKED": 0, "UNKNOWN": 0}
    total = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = ROW_PATTERN.match(line)
        if match is None:
            continue
        total += 1
        status = match.group(2)
        counts[status if status in counts else "UNKNOWN"] += 1
    if total == 0:
        raise ValueError("No acceptance requirements found in traceability matrix")
    return ComplianceSummary(
        total_requirements=total,
        tested=counts["TESTED"],
        implemented=counts["IMPLEMENTED"],
        pending=counts["PENDING"],
        blocked=counts["BLOCKED"],
        unknown=counts["UNKNOWN"],
    )


def write_report(summary: ComplianceSummary, output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(summary) | {"complete": summary.complete}
    if output_path.suffix == ".json":
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return
    lines = [
        "# Build Compliance Report",
        "",
        f"- Total mapped requirements: {summary.total_requirements}",
        f"- Tested: {summary.tested}",
        f"- Implemented, awaiting confirmed tests: {summary.implemented}",
        f"- Pending: {summary.pending}",
        f"- Blocked: {summary.blocked}",
        f"- Unknown status: {summary.unknown}",
        f"- Complete: {'YES' if summary.complete else 'NO'}",
        "",
        "A requirement is complete only after its mapped implementation and automated test pass CI.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = parse_traceability("docs/TRACEABILITY_MATRIX.md")
    write_report(result, "reports/build_compliance_report.md")
    write_report(result, "reports/build_compliance_report.json")
