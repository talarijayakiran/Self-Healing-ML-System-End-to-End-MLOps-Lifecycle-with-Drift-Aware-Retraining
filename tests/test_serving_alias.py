from unittest.mock import Mock

import pytest

import src.serving.app as serving


def test_load_model_uses_production_alias(monkeypatch):
    client = Mock()

    production_version = Mock()
    production_version.version = "7"

    client.get_model_version_by_alias.return_value = (
        production_version
    )

    loaded_model = Mock()

    load_model_calls = []

    def fake_load_model(uri):
        load_model_calls.append(uri)
        return loaded_model

    monkeypatch.setattr(
        serving,
        "mlflow_client",
        client,
    )

    monkeypatch.setattr(
        serving.mlflow.pyfunc,
        "load_model",
        fake_load_model,
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

    client.get_model_version_by_alias.assert_called_once_with(
        serving.MODEL_NAME,
        serving.PRODUCTION_ALIAS,
    )

    assert load_model_calls == [
        f"models:/{serving.MODEL_NAME}@"
        f"{serving.PRODUCTION_ALIAS}"
    ]

    assert serving.model_version == "7"


def test_load_model_fails_when_production_alias_missing(
    monkeypatch,
):
    client = Mock()

    client.get_model_version_by_alias.side_effect = (
        RuntimeError("alias not found")
    )

    monkeypatch.setattr(
        serving,
        "mlflow_client",
        client,
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
        match="production",
    ):
        serving.load_model()


def test_load_model_reuses_same_production_version(
    monkeypatch,
):
    client = Mock()

    production_version = Mock()
    production_version.version = "7"

    client.get_model_version_by_alias.return_value = (
        production_version
    )

    existing_model = Mock()

    monkeypatch.setattr(
        serving,
        "mlflow_client",
        client,
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

    serving.mlflow.pyfunc.load_model.assert_not_called()


def test_load_model_reuses_same_production_version(
    monkeypatch,
):
    client = Mock()

    production_version = Mock()
    production_version.version = "7"

    client.get_model_version_by_alias.return_value = (
        production_version
    )

    existing_model = Mock()
    load_model_mock = Mock()

    monkeypatch.setattr(
        serving,
        "mlflow_client",
        client,
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

    monkeypatch.setattr(
        serving.mlflow.pyfunc,
        "load_model",
        load_model_mock,
    )

    result = serving.load_model()

    assert result is existing_model
    load_model_mock.assert_not_called()