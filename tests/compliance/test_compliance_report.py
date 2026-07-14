from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path("tools/compliance_report.py")
    spec = importlib.util.spec_from_file_location("compliance_report", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load compliance report module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_traceability_parser_counts_statuses(tmp_path) -> None:
    module = load_module()
    matrix = tmp_path / "matrix.md"
    matrix.write_text(
        "\n".join(
            [
                "| Requirement | Implementation | Test | Status |",
                "|---|---|---|---|",
                "| AC-ONE | a.py | test_a.py | TESTED |",
                "| AC-TWO | b.py | test_b.py | IMPLEMENTED |",
                "| AC-THREE | c.py | test_c.py | PENDING |",
            ]
        ),
        encoding="utf-8",
    )
    summary = module.parse_traceability(matrix)
    assert summary.total_requirements == 3
    assert summary.tested == 1
    assert summary.implemented == 1
    assert summary.pending == 1
    assert summary.complete is False


def test_report_marks_complete_only_without_pending_or_blocked(tmp_path) -> None:
    module = load_module()
    summary = module.ComplianceSummary(2, 2, 0, 0, 0, 0)
    output = tmp_path / "report.json"
    module.write_report(summary, output)
    assert '"complete": true' in output.read_text(encoding="utf-8")
