# src/evaluation/quality_gate.py

"""
Model quality gate for the retail demand forecasting pipeline.

The quality gate is responsible for deciding whether a newly
trained candidate model satisfies the minimum evaluation policy
required for registration.

This module deliberately contains no training or MLflow logic.
It only evaluates model metrics against explicit quality criteria.
"""

from dataclasses import dataclass


# ------------------------------------------------------------------
# QUALITY POLICY
# ------------------------------------------------------------------

DEFAULT_MAX_RMSE = 3.0


# ------------------------------------------------------------------
# QUALITY GATE RESULT
# ------------------------------------------------------------------


@dataclass(frozen=True)
class QualityGateResult:
    """
    Immutable result returned by the model quality gate.
    """

    passed: bool
    rmse: float
    max_rmse: float
    reason: str


# ------------------------------------------------------------------
# QUALITY GATE
# ------------------------------------------------------------------


def evaluate_model_quality(
    rmse: float,
    max_rmse: float = DEFAULT_MAX_RMSE,
) -> QualityGateResult:
    """
    Evaluate whether a candidate model satisfies the RMSE policy.

    Args:
        rmse:
            Validation RMSE produced by the candidate model.

        max_rmse:
            Maximum acceptable RMSE.

    Returns:
        QualityGateResult describing whether the candidate passed.

    Raises:
        ValueError:
            If RMSE or the configured threshold is invalid.
    """

    if rmse < 0:
        raise ValueError("RMSE cannot be negative.")

    if max_rmse <= 0:
        raise ValueError("Maximum RMSE threshold must be greater than 0.")

    if rmse <= max_rmse:
        return QualityGateResult(
            passed=True,
            rmse=rmse,
            max_rmse=max_rmse,
            reason=(
                f"Candidate passed quality gate: "
                f"RMSE {rmse:.4f} <= threshold {max_rmse:.4f}."
            ),
        )

    return QualityGateResult(
        passed=False,
        rmse=rmse,
        max_rmse=max_rmse,
        reason=(
            f"Candidate failed quality gate: "
            f"RMSE {rmse:.4f} > threshold {max_rmse:.4f}."
        ),
    )