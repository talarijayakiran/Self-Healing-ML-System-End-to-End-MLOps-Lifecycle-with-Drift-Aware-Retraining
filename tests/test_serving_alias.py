from pathlib import Path
from unittest.mock import Mock

import pytest

import src.serving.app as serving


def test_load_model_uses_packaged_model(monkeypatch):
    """
    Production serving must load the immutable packaged model
    shipped inside the container.
    """

    loaded_model = Mock()

    load_model_mock = Mock(
        return_value=loaded_model
    )

    monkeypatch.setattr(
        serving.mlflow.pyfunc,
        "load_model",
        load_model_mock,
    )

    monkeypatch.setattr(
        serving,
        "model",
        None,
    )

    monkeypatch.setattr(
        serving,
        "model_version",
        None,
    )

    monkeypatch.setattr(
        serving,
        "TEST_MODE",
        False,
    )

    result = serving.load_model()

    assert result is loaded_model

    load_model_mock.assert_called_once()

    model_uri = load_model_mock.call_args.args[0]

    assert str(model_uri).endswith(
        "exported_model"
    )

    assert serving.model is loaded_model

    assert serving.model_version == "7"


def test_load_model_fails_when_packaged_model_missing(
    monkeypatch,
):
    """
    Serving must fail explicitly if the packaged production
    model cannot be loaded.
    """

    load_model_mock = Mock(
        side_effect=OSError(
            "No such file or directory: exported_model"
        )
    )

    monkeypatch.setattr(
        serving.mlflow.pyfunc,
        "load_model",
        load_model_mock,
    )

    monkeypatch.setattr(
        serving,
        "model",
        None,
    )

    monkeypatch.setattr(
        serving,
        "model_version",
        None,
    )

    monkeypatch.setattr(
        serving,
        "TEST_MODE",
        False,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to load packaged production model",
    ):
        serving.load_model()


def test_load_model_reuses_existing_packaged_model(
    monkeypatch,
):
    """
    If the packaged model is already loaded and its version
    has not changed, serving should reuse the existing model
    instead of loading it again.
    """

    existing_model = Mock()

    load_model_mock = Mock()

    monkeypatch.setattr(
        serving,
        "mlflow",
        serving.mlflow,
    )

    monkeypatch.setattr(
        serving.mlflow.pyfunc,
        "load_model",
        load_model_mock,
    )

    monkeypatch.setattr(
        serving,
        "model",
        existing_model,
    )

    monkeypatch.setattr(
        serving,
        "model_version",
        "7",
    )

    monkeypatch.setattr(
        serving,
        "TEST_MODE",
        False,
    )

    result = serving.load_model()

    assert result is existing_model

    load_model_mock.assert_not_called()


def test_load_model_force_reload(
    monkeypatch,
):
    """
    force_reload=True must reload the packaged model even
    when the current model is already loaded.
    """

    existing_model = Mock()
    reloaded_model = Mock()

    load_model_mock = Mock(
        return_value=reloaded_model
    )

    monkeypatch.setattr(
        serving.mlflow.pyfunc,
        "load_model",
        load_model_mock,
    )

    monkeypatch.setattr(
        serving,
        "model",
        existing_model,
    )

    monkeypatch.setattr(
        serving,
        "model_version",
        "7",
    )

    monkeypatch.setattr(
        serving,
        "TEST_MODE",
        False,
    )

    result = serving.load_model(
        force_reload=True
    )

    assert result is reloaded_model

    assert serving.model is reloaded_model

    assert serving.model_version == "7"

    load_model_mock.assert_called_once()


def test_load_model_skipped_in_test_mode(
    monkeypatch,
):
    """
    TEST_MODE must prevent production model loading.
    """

    load_model_mock = Mock()

    monkeypatch.setattr(
        serving.mlflow.pyfunc,
        "load_model",
        load_model_mock,
    )

    monkeypatch.setattr(
        serving,
        "TEST_MODE",
        True,
    )

    monkeypatch.setattr(
        serving,
        "model",
        None,
    )

    monkeypatch.setattr(
        serving,
        "model_version",
        None,
    )

    result = serving.load_model()

    assert result is None

    load_model_mock.assert_not_called()