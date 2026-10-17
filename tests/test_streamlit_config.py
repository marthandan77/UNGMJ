from pathlib import Path


def test_environment_template_exists() -> None:
    template = Path(__file__).parents[1] / ".env.example"
    content = template.read_text(encoding="utf-8")
    assert "EIA_API_KEY" in content
    assert "CFTC_NATURAL_GAS_CSV_URL" in content
