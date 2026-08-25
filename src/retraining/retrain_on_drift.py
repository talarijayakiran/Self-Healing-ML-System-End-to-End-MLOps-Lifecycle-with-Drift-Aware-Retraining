"""
Automated retraining controller for the retail demand forecasting system.

Responsibilities
----------------
1. Load the persisted drift report.
2. Validate the drift decision.
3. Trigger the training pipeline when drift is detected.
4. Keep retraining separate from model promotion.
5. Surface training failures to the caller.

This module is an orchestration boundary.

It does NOT:
- promote models,
- modify the production alias,
- bypass the quality gate,
- directly manipulate MLflow model versions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DRIFT_REPORT_PATH = Path(
    "data/monitoring/drift_report.json"
)

TRAINING_MODULE = (
    "src.training.train_model"
)


# ============================================================
# RESULT CONTRACT
# ============================================================


@dataclass(frozen=True)
class RetrainingResult:
    """
    Structured result of a retraining decision.
    """

    drift_detected: bool
    retraining_triggered: bool
    training_succeeded: bool


# ============================================================
# DRIFT REPORT LOADING
# ============================================================


def _load_drift_report() -> dict:
    """
    Load the persisted drift report.

    Raises
    ------
    FileNotFoundError
        If the drift report does not exist.

    ValueError
        If the report is invalid or malformed.
    """

    if not DRIFT_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Drift report not found: "
            f"{DRIFT_REPORT_PATH}"
        )

    try:
        with DRIFT_REPORT_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            report = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Drift report contains invalid JSON."
        ) from exc

    if not isinstance(report, dict):
        raise ValueError(
            "Drift report must contain a JSON object."
        )

    return report


# ============================================================
# DRIFT DECISION
# ============================================================


def _drift_detected(report: dict) -> bool:
    """
    Determine whether the persisted drift report requires
    retraining.

    The canonical decision is stored under:

        report["_summary"]["drift_detected"]
    """

    summary = report.get("_summary")

    if not isinstance(summary, dict):
        raise ValueError(
            "Drift report is missing a valid '_summary' section."
        )

    drift_detected = summary.get(
        "drift_detected"
    )

    if not isinstance(
        drift_detected,
        bool,
    ):
        raise ValueError(
            "'_summary.drift_detected' must be boolean."
        )

    return drift_detected


# ============================================================
# TRAINING TRIGGER
# ============================================================


def _run_training() -> None:
    """
    Execute the canonical training pipeline.

    Training is intentionally invoked through Python's module
    interface so that the same entry point can be used by:

    - local execution,
    - Docker,
    - CI/CD,
    - scheduled jobs,
    - future orchestration systems.
    """

    subprocess.run(
        [
            sys.executable,
            "-m",
            TRAINING_MODULE,
        ],
        check=True,
    )


# ============================================================
# RETRAINING CONTROLLER
# ============================================================


def retrain_model() -> RetrainingResult:
    """
    Evaluate the drift report and trigger retraining when required.

    Returns
    -------
    RetrainingResult
        Structured outcome of the retraining decision.

    Raises
    ------
    FileNotFoundError
        If the drift report does not exist.

    ValueError
        If the drift report is malformed.

    subprocess.CalledProcessError
        If the training pipeline fails.

    Important
    ---------
    This function never promotes a model.

    A successful retraining run produces a candidate model
    through the existing training pipeline. Production promotion
    remains an explicit downstream operation.
    """

    report = _load_drift_report()

    drift_detected = _drift_detected(
        report
    )

    # --------------------------------------------------------
    # NO DRIFT
    # --------------------------------------------------------

    if not drift_detected:

        print(
            "No drift detected. "
            "Retraining skipped."
        )

        return RetrainingResult(
            drift_detected=False,
            retraining_triggered=False,
            training_succeeded=False,
        )

    # --------------------------------------------------------
    # DRIFT DETECTED
    # --------------------------------------------------------

    print(
        "Drift detected. "
        "Starting retraining pipeline..."
    )

    _run_training()

    print(
        "Retraining completed successfully."
    )

    return RetrainingResult(
        drift_detected=True,
        retraining_triggered=True,
        training_succeeded=True,
    )


# ============================================================
# CLI ENTRY POINT
# ============================================================


if __name__ == "__main__":
    retrain_model()