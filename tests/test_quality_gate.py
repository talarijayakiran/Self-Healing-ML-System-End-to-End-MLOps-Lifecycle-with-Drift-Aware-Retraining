from src.evaluation.quality_gate import (
    DEFAULT_MAX_RMSE,
    evaluate_model_quality,
)


def test_quality_gate_passes_when_rmse_is_within_threshold():
    result = evaluate_model_quality(
        rmse=2.0,
        max_rmse=3.0,
    )

    assert result.passed is True
    assert result.rmse == 2.0
    assert result.max_rmse == 3.0


def test_quality_gate_passes_when_rmse_equals_threshold():
    result = evaluate_model_quality(
        rmse=3.0,
        max_rmse=3.0,
    )

    assert result.passed is True


def test_quality_gate_fails_when_rmse_exceeds_threshold():
    result = evaluate_model_quality(
        rmse=3.5,
        max_rmse=3.0,
    )

    assert result.passed is False
    assert result.rmse == 3.5
    assert result.max_rmse == 3.0


def test_quality_gate_rejects_negative_rmse():
    try:
        evaluate_model_quality(
            rmse=-1.0,
            max_rmse=3.0,
        )
    except ValueError as exc:
        assert str(exc) == "RMSE cannot be negative."
    else:
        raise AssertionError("Expected ValueError")


def test_quality_gate_rejects_invalid_threshold():
    try:
        evaluate_model_quality(
            rmse=2.0,
            max_rmse=0.0,
        )
    except ValueError as exc:
        assert str(exc) == (
            "Maximum RMSE threshold must be greater than 0."
        )
    else:
        raise AssertionError("Expected ValueError")


def test_default_threshold_is_defined():
    assert DEFAULT_MAX_RMSE > 0