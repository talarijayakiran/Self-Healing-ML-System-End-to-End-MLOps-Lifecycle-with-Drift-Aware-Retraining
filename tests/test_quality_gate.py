from src.config import settings as settings_module
from src.config.settings import Settings
from src.evaluation.quality_gate import (
    DEFAULT_MAX_RMSE,
    evaluate_model_quality,
)


# ============================================================
# HELPERS
# ============================================================


def _build_settings(
    *,
    max_rmse: float,
) -> Settings:
    """
    Build a complete runtime Settings object for tests.

    Settings is intentionally frozen, so tests replace the
    complete settings object rather than mutating individual
    fields.
    """

    current = settings_module.settings

    return Settings(
        max_rmse=max_rmse,
        drift_threshold=current.drift_threshold,
        observation_window_size=(
            current.observation_window_size
        ),
        min_observations=current.min_observations,
        prediction_log_path=(
            current.prediction_log_path
        ),
        drift_report_path=(
            current.drift_report_path
        ),
        reference_data_path=(
            current.reference_data_path
        ),
        mlflow_tracking_uri=(
            current.mlflow_tracking_uri
        ),
        mlflow_model_name=(
            current.mlflow_model_name
        ),
        mlflow_production_alias=(
            current.mlflow_production_alias
        ),
    )


# ============================================================
# EXPLICIT THRESHOLD TESTS
# ============================================================


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


# ============================================================
# RUNTIME CONFIGURATION TESTS
# ============================================================


def test_quality_gate_uses_runtime_max_rmse(
    monkeypatch,
):
    runtime_settings = _build_settings(
        max_rmse=2.5,
    )

    monkeypatch.setattr(
        settings_module,
        "settings",
        runtime_settings,
    )

    result = evaluate_model_quality(
        rmse=2.5,
    )

    assert result.passed is True
    assert result.max_rmse == 2.5


def test_quality_gate_rejects_using_runtime_max_rmse(
    monkeypatch,
):
    runtime_settings = _build_settings(
        max_rmse=2.5,
    )

    monkeypatch.setattr(
        settings_module,
        "settings",
        runtime_settings,
    )

    result = evaluate_model_quality(
        rmse=2.6,
    )

    assert result.passed is False
    assert result.max_rmse == 2.5


def test_explicit_threshold_overrides_runtime_configuration(
    monkeypatch,
):
    runtime_settings = _build_settings(
        max_rmse=2.0,
    )

    monkeypatch.setattr(
        settings_module,
        "settings",
        runtime_settings,
    )

    result = evaluate_model_quality(
        rmse=2.5,
        max_rmse=3.0,
    )

    assert result.passed is True
    assert result.max_rmse == 3.0


# ============================================================
# VALIDATION TESTS
# ============================================================


def test_quality_gate_rejects_negative_rmse():
    try:
        evaluate_model_quality(
            rmse=-1.0,
            max_rmse=3.0,
        )
    except ValueError as exc:
        assert str(exc) == (
            "RMSE cannot be negative."
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_quality_gate_rejects_non_finite_rmse():
    try:
        evaluate_model_quality(
            rmse=float("nan"),
            max_rmse=3.0,
        )
    except ValueError as exc:
        assert str(exc) == (
            "RMSE must be finite."
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )


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
        raise AssertionError(
            "Expected ValueError"
        )


def test_quality_gate_rejects_non_finite_threshold():
    try:
        evaluate_model_quality(
            rmse=2.0,
            max_rmse=float("inf"),
        )
    except ValueError as exc:
        assert str(exc) == (
            "Maximum RMSE threshold must be finite."
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )


# ============================================================
# CONFIGURATION CONTRACT
# ============================================================


def test_default_threshold_matches_runtime_configuration():
    assert DEFAULT_MAX_RMSE == (
        settings_module.settings.max_rmse
    )


def test_default_threshold_is_positive():
    assert DEFAULT_MAX_RMSE > 0