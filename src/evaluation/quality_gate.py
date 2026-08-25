"""
Model quality gate for the retail demand forecasting pipeline.

The quality gate is responsible for deciding whether a newly
trained candidate model satisfies the minimum evaluation policy
required for registration.

This module deliberately contains no training or MLflow logic.
It only evaluates model metrics against the runtime quality policy.

Runtime configuration is owned by src.config.settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import settings as settings_module


# ============================================================
# BACKWARD-COMPATIBILITY CONSTANT
# ============================================================

# Runtime configuration is the source of truth.
#
# This constant is retained so existing callers/tests that import
# DEFAULT_MAX_RMSE do not break.
#
# New application code should use:
#
#     settings_module.settings.max_rmse
#
# instead of defining another hard-coded threshold.
DEFAULT_MAX_RMSE = (
    settings_module.settings.max_rmse
)


# ============================================================
# QUALITY GATE RESULT
# ============================================================


@dataclass(frozen=True)
class QualityGateResult:
    """
    Immutable result returned by the model quality gate.
    """

    passed: bool
    rmse: float
    max_rmse: float
    reason: str


# ============================================================
# VALIDATION
# ============================================================


def _validate_rmse(
    rmse: float,
) -> None:
    """
    Validate the candidate model RMSE.

    RMSE must be:

    - finite
    - greater than or equal to zero
    """

    if not math.isfinite(rmse):
        raise ValueError(
            "RMSE must be finite."
        )

    if rmse < 0:
        raise ValueError(
            "RMSE cannot be negative."
        )


def _validate_max_rmse(
    max_rmse: float,
) -> None:
    """
    Validate the maximum allowed RMSE threshold.

    The threshold must be:

    - finite
    - greater than zero
    """

    if not math.isfinite(max_rmse):
        raise ValueError(
            "Maximum RMSE threshold must be finite."
        )

    if max_rmse <= 0:
        raise ValueError(
            "Maximum RMSE threshold must be greater than 0."
        )


# ============================================================
# QUALITY GATE
# ============================================================


def evaluate_model_quality(
    rmse: float,
    max_rmse: float | None = None,
) -> QualityGateResult:
    """
    Evaluate whether a candidate model satisfies the RMSE policy.

    Parameters
    ----------
    rmse:
        Validation RMSE produced by the candidate model.

    max_rmse:
        Optional maximum acceptable RMSE.

        When omitted, the threshold is resolved from the
        centralized runtime configuration:

            settings_module.settings.max_rmse

        Explicit values are supported for deterministic unit
        tests and controlled policy overrides.

    Returns
    -------
    QualityGateResult
        Structured quality-gate decision.

    Raises
    ------
    ValueError
        If RMSE or the configured threshold is invalid.
    """

    # --------------------------------------------------------
    # RESOLVE RUNTIME POLICY
    # --------------------------------------------------------

    effective_max_rmse = (
        settings_module.settings.max_rmse
        if max_rmse is None
        else max_rmse
    )

    # --------------------------------------------------------
    # VALIDATE INPUTS
    # --------------------------------------------------------

    _validate_rmse(
        rmse
    )

    _validate_max_rmse(
        effective_max_rmse
    )

    # --------------------------------------------------------
    # QUALITY DECISION
    # --------------------------------------------------------

    if rmse <= effective_max_rmse:

        return QualityGateResult(
            passed=True,
            rmse=rmse,
            max_rmse=effective_max_rmse,
            reason=(
                f"Candidate passed quality gate: "
                f"RMSE {rmse:.4f} <= threshold "
                f"{effective_max_rmse:.4f}."
            ),
        )

    return QualityGateResult(
        passed=False,
        rmse=rmse,
        max_rmse=effective_max_rmse,
        reason=(
            f"Candidate failed quality gate: "
            f"RMSE {rmse:.4f} > threshold "
            f"{effective_max_rmse:.4f}."
        ),
    )