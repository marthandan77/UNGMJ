from ung_forecast.weights import compute_model_weights


def rows(model, error, n=20):
    return [
        {"model_name": model, "horizon": 1, "target_session": f"2026-01-{i + 1:02d}",
         "reference_close": 100.0, "low_error": error, "close_error": error, "high_error": error}
        for i in range(n)
    ]


def test_weights_are_learned_per_horizon_and_bounded() -> None:
    result = compute_model_weights(rows("hist_gradient_boosting", 1.0) + rows("hgb_expert_volume", 2.0))
    assert len(result) == 2
    gradient_boosting = next(row for row in result if row["model_name"] == "hist_gradient_boosting")
    weaker_expert = next(row for row in result if row["model_name"] == "hgb_expert_volume")
    assert gradient_boosting["weight"] > weaker_expert["weight"]
    assert all(0.0 < row["weight"] < 1.0 for row in result)


def test_weights_wait_for_minimum_evidence() -> None:
    result = compute_model_weights(rows("hist_gradient_boosting", 1.0, 2))
    assert len(result) == 1
    assert result[0]["status"] == "WARMING_UP"
