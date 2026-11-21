from ung_forecast.weights import compute_model_weights


def rows(model, error, n=20):
    return [{"model_name": model, "horizon": 1, "close_error": error} for _ in range(n)]


def test_weights_are_learned_per_horizon_and_bounded() -> None:
    result = compute_model_weights(rows("hist_gradient_boosting", 1.0) + rows("random_forest", 2.0))
    assert len(result) == 2
    gradient_boosting = next(row for row in result if row["model_name"] == "hist_gradient_boosting")
    forest = next(row for row in result if row["model_name"] == "random_forest")
    assert gradient_boosting["weight"] > forest["weight"]
    assert all(0.05 <= row["weight"] <= 0.50 for row in result)


def test_weights_wait_for_minimum_evidence() -> None:
    assert compute_model_weights(rows("hist_gradient_boosting", 1.0, 2)) == []
