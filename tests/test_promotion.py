from unittest.mock import Mock

import pytest

from src.registry.promotion import (
    PRODUCTION_ALIAS,
    promote_model,
)


def build_candidate_model_version(
    version: str = "7",
    *,
    quality_gate_passed: str = "true",
):
    model_version = Mock()

    model_version.version = version

    model_version.tags = {
        "model_role": "candidate",
        "quality_gate_passed": quality_gate_passed,
    }

    return model_version


def test_promotion_assigns_production_alias():
    client = Mock()

    model_version = build_candidate_model_version(
        version="7",
    )

    client.get_model_version.return_value = model_version

    result = promote_model(
        "7",
        client=client,
    )

    assert result.model_name == "retail_demand_forecaster"
    assert result.model_version == "7"
    assert result.alias == PRODUCTION_ALIAS
    assert result.promoted is True

    client.get_model_version.assert_called_once_with(
        name="retail_demand_forecaster",
        version="7",
    )

    client.set_registered_model_alias.assert_called_once_with(
        name="retail_demand_forecaster",
        alias="production",
        version="7",
    )


def test_promotion_rejects_empty_version():
    client = Mock()

    with pytest.raises(
        ValueError,
        match="Model version must be provided",
    ):
        promote_model(
            "",
            client=client,
        )

    client.get_model_version.assert_not_called()


def test_promotion_rejects_whitespace_version():
    client = Mock()

    with pytest.raises(
        ValueError,
        match="Model version must be provided",
    ):
        promote_model(
            "   ",
            client=client,
        )

    client.get_model_version.assert_not_called()


def test_promotion_fails_when_model_version_cannot_be_loaded():
    client = Mock()

    client.get_model_version.side_effect = Exception(
        "MLflow unavailable"
    )

    with pytest.raises(
        RuntimeError,
        match="Unable to retrieve registered model version",
    ):
        promote_model(
            "7",
            client=client,
        )

    client.set_registered_model_alias.assert_not_called()


def test_promotion_rejects_non_candidate_model():
    client = Mock()

    model_version = Mock()
    model_version.version = "7"
    model_version.tags = {
        "model_role": "production",
        "quality_gate_passed": "true",
    }

    client.get_model_version.return_value = model_version

    with pytest.raises(
        RuntimeError,
        match="not marked as a candidate",
    ):
        promote_model(
            "7",
            client=client,
        )

    client.set_registered_model_alias.assert_not_called()


def test_promotion_rejects_failed_quality_gate():
    client = Mock()

    model_version = build_candidate_model_version(
        version="7",
        quality_gate_passed="false",
    )

    client.get_model_version.return_value = model_version

    with pytest.raises(
        RuntimeError,
        match="did not pass the quality gate",
    ):
        promote_model(
            "7",
            client=client,
        )

    client.set_registered_model_alias.assert_not_called()