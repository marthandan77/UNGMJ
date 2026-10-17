import json


def test_run_report_schema_is_json_serializable(tmp_path):
    report = {
        "status": "PROVISIONAL", "context": {"eia_storage": "UNAVAILABLE"},
        "forecasts": 10, "scored": 0,
    }
    path = tmp_path / "latest-run.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["forecasts"] == 10
    assert loaded["status"] in {"READY", "PROVISIONAL", "BLOCKED"}
