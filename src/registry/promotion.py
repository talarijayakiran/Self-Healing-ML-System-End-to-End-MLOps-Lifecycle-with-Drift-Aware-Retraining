"""
Model promotion boundary for the retail demand forecasting system.

Responsibilities
----------------
1. Identify a registered candidate model version.
2. Verify that the candidate passed the model quality gate.
3. Promote only an eligible candidate using an MLflow alias.
4. Keep production model selection separate from training.
5. Avoid hard-coded production model versions.
6. Return a structured promotion result.

Promotion is intentionally explicit and controlled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlflow.tracking import MlflowClient


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "retail_demand_forecaster"

CANDIDATE_ALIAS = "candidate"
PRODUCTION_ALIAS = "production"

QUALITY_GATE_PASSED_TAG = "quality_gate_passed"
MODEL_ROLE_TAG = "model_role"


# ============================================================
# RESULT CONTRACT
# ============================================================


@dataclass(frozen=True)
class PromotionResult:
    """
    Immutable result returned after model promotion.
    """

    model_name: str
    model_version: str
    alias: str
    promoted: bool


# ============================================================
# MODEL VERSION LOOKUP
# ============================================================


def get_registered_model_version(
    client: MlflowClient,
    model_name: str,
    version: str,
) -> Any:
    """
    Retrieve a specific registered MLflow model version.

    Parameters
    ----------
    client:
        MLflow tracking client.

    model_name:
        Registered model name.

    version:
        Registered model version.

    Returns
    -------
    Any
        MLflow model version object.
    """

    return client.get_model_version(
        name=model_name,
        version=version,
    )


# ============================================================
# ELIGIBILITY CHECKS
# ============================================================


def _quality_gate_passed(model_version: Any) -> bool:
    """
    Determine whether the registered model version passed
    the model quality gate.

    The quality gate result is stored as an MLflow model-version
    tag and is treated as part of the promotion contract.
    """

    tags = getattr(model_version, "tags", {}) or {}

    value = tags.get(
        QUALITY_GATE_PASSED_TAG,
        "",
    )

    return str(value).strip().lower() == "true"


def _is_candidate(model_version: Any) -> bool:
    """
    Determine whether the registered model version is explicitly
    marked as a candidate model.
    """

    tags = getattr(model_version, "tags", {}) or {}

    value = tags.get(
        MODEL_ROLE_TAG,
        "",
    )

    return str(value).strip().lower() == "candidate"


def _validate_promotion_eligibility(
    model_version: Any,
    version: str,
) -> None:
    """
    Validate all promotion eligibility requirements.

    A model can be promoted only when:

    1. It is explicitly marked as a candidate.
    2. It passed the model quality gate.
    """

    if not _is_candidate(model_version):
        raise RuntimeError(
            f"Model version {version} is not marked as a candidate."
        )

    if not _quality_gate_passed(model_version):
        raise RuntimeError(
            f"Model version {version} did not pass the quality gate."
        )


# ============================================================
# MODEL PROMOTION
# ============================================================


def promote_model(
    version: str,
    *,
    model_name: str = MODEL_NAME,
    client: MlflowClient | None = None,
) -> PromotionResult:
    """
    Promote an eligible registered candidate model version.

    Promotion requires BOTH:

        model_role == "candidate"
        quality_gate_passed == "true"

    The model is promoted using the MLflow ``production`` alias.

    Parameters
    ----------
    version:
        Registered MLflow model version to promote.

    model_name:
        Registered MLflow model name.

    client:
        Optional MLflow client.

        This is primarily useful for testing and dependency
        injection. In production, the default MlflowClient is used.

    Returns
    -------
    PromotionResult
        Structured result describing the promotion.

    Raises
    ------
    ValueError
        If the model version is empty or contains only whitespace.

    RuntimeError
        If the model version cannot be retrieved.

    RuntimeError
        If the model is not marked as a candidate.

    RuntimeError
        If the model failed the quality gate.
    """

    # ---------------------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------------------

    if not version or not version.strip():
        raise ValueError(
            "Model version must be provided."
        )

    normalized_version = version.strip()

    # ---------------------------------------------------------
    # CLIENT
    # ---------------------------------------------------------

    client = client or MlflowClient()

    # ---------------------------------------------------------
    # LOAD REGISTERED MODEL VERSION
    # ---------------------------------------------------------

    try:
        model_version = get_registered_model_version(
            client=client,
            model_name=model_name,
            version=normalized_version,
        )

    except Exception as exc:
        raise RuntimeError(
            "Unable to retrieve registered model version "
            f"{model_name}:{normalized_version}."
        ) from exc

    # ---------------------------------------------------------
    # PROMOTION ELIGIBILITY
    # ---------------------------------------------------------

    _validate_promotion_eligibility(
        model_version=model_version,
        version=normalized_version,
    )

    # ---------------------------------------------------------
    # PROMOTE USING MLflow ALIAS
    # ---------------------------------------------------------

    client.set_registered_model_alias(
        name=model_name,
        alias=PRODUCTION_ALIAS,
        version=str(model_version.version),
    )

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------

    result = PromotionResult(
        model_name=model_name,
        model_version=str(model_version.version),
        alias=PRODUCTION_ALIAS,
        promoted=True,
    )

    # ---------------------------------------------------------
    # OPERATIONAL OUTPUT
    # ---------------------------------------------------------

    print(
        "MODEL PROMOTION SUCCEEDED"
    )

    print(
        f"Model: {result.model_name}"
    )

    print(
        f"Version: {result.model_version}"
    )

    print(
        f"Alias: {result.alias}"
    )

    print(
        f"Promoted: {result.promoted}"
    )

    return result