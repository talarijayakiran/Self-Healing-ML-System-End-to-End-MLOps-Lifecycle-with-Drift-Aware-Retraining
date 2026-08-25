import pytest

from src.config import settings as settings_module


def test_default_settings_are_loaded(
    monkeypatch,
):
    monkeypatch.delenv(
        "MAX_RMSE",
        raising=False,
    )

    monkeypatch.delenv(
        "DRIFT_THRESHOLD",
        raising=False,
    )

    monkeypatch.delenv(
        "OBSERVATION_WINDOW_SIZE",
        raising=False,
    )

    monkeypatch.delenv(
        "MIN_OBSERVATIONS",
        raising=False,
    )

    settings = settings_module.load_settings()

    assert settings.max_rmse == 3.0
    assert settings.drift_threshold == 0.20
    assert settings.observation_window_size == 50
    assert settings.min_observations == 10


def test_environment_overrides_defaults(
    monkeypatch,
):
    monkeypatch.setenv(
        "MAX_RMSE",
        "2.5",
    )

    monkeypatch.setenv(
        "DRIFT_THRESHOLD",
        "0.15",
    )

    monkeypatch.setenv(
        "OBSERVATION_WINDOW_SIZE",
        "100",
    )

    monkeypatch.setenv(
        "MIN_OBSERVATIONS",
        "20",
    )

    settings = settings_module.load_settings()

    assert settings.max_rmse == 2.5
    assert settings.drift_threshold == 0.15
    assert settings.observation_window_size == 100
    assert settings.min_observations == 20


def test_invalid_rmse_is_rejected(
    monkeypatch,
):
    monkeypatch.setenv(
        "MAX_RMSE",
        "not-a-number",
    )

    with pytest.raises(
        ValueError,
        match="MAX_RMSE must be a valid number",
    ):
        settings_module.load_settings()


def test_negative_rmse_is_rejected(
    monkeypatch,
):
    monkeypatch.setenv(
        "MAX_RMSE",
        "-1",
    )

    with pytest.raises(
        ValueError,
        match="MAX_RMSE must be greater than 0",
    ):
        settings_module.load_settings()


def test_invalid_drift_threshold_is_rejected(
    monkeypatch,
):
    monkeypatch.setenv(
        "DRIFT_THRESHOLD",
        "invalid",
    )

    with pytest.raises(
        ValueError,
        match="DRIFT_THRESHOLD must be a valid number",
    ):
        settings_module.load_settings()


def test_invalid_observation_window_is_rejected(
    monkeypatch,
):
    monkeypatch.setenv(
        "OBSERVATION_WINDOW_SIZE",
        "0",
    )

    with pytest.raises(
        ValueError,
        match="OBSERVATION_WINDOW_SIZE must be greater than 0",
    ):
        settings_module.load_settings()


def test_minimum_observations_cannot_exceed_window(
    monkeypatch,
):
    monkeypatch.setenv(
        "OBSERVATION_WINDOW_SIZE",
        "10",
    )

    monkeypatch.setenv(
        "MIN_OBSERVATIONS",
        "20",
    )

    with pytest.raises(
        ValueError,
        match="MIN_OBSERVATIONS cannot be greater",
    ):
        settings_module.load_settings()


def test_empty_mlflow_model_name_is_rejected(
    monkeypatch,
):
    monkeypatch.setenv(
        "MLFLOW_MODEL_NAME",
        "   ",
    )

    with pytest.raises(
        ValueError,
        match="MLFLOW_MODEL_NAME must not be empty",
    ):
        settings_module.load_settings()