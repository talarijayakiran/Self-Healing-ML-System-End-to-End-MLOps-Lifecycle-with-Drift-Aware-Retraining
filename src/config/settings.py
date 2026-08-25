"""
Runtime configuration for the retail MLOps system.

This module is the single runtime configuration boundary.

Design principles
-----------------
1. Configuration comes from environment variables.
2. Safe development defaults are provided.
3. Values are parsed and validated at startup.
4. ML/data contracts remain in schema.py.
5. Secrets must never be hard-coded here.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# HELPERS
# ============================================================


def _get_float(
    name: str,
    default: float,
) -> float:
    """
    Read a finite positive floating-point configuration value.
    """

    raw_value = os.getenv(
        name,
        str(default),
    )

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a valid number."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than 0."
        )

    return value


def _get_int(
    name: str,
    default: int,
) -> int:
    """
    Read a positive integer configuration value.
    """

    raw_value = os.getenv(
        name,
        str(default),
    )

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a valid integer."
        ) from exc

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than 0."
        )

    return value


def _get_path(
    name: str,
    default: str,
) -> Path:
    """
    Read a filesystem path configuration value.
    """

    value = os.getenv(
        name,
        default,
    ).strip()

    if not value:
        raise ValueError(
            f"{name} must not be empty."
        )

    return Path(value)


def _get_string(
    name: str,
    default: str,
) -> str:
    """
    Read a non-empty string configuration value.
    """

    value = os.getenv(
        name,
        default,
    ).strip()

    if not value:
        raise ValueError(
            f"{name} must not be empty."
        )

    return value


# ============================================================
# APPLICATION SETTINGS
# ============================================================


@dataclass(frozen=True)
class Settings:
    """
    Validated runtime configuration.

    This object contains deployment/runtime configuration only.

    ML data contracts belong in schema.py.
    """

    # --------------------------------------------------------
    # MODEL QUALITY
    # --------------------------------------------------------

    max_rmse: float

    # --------------------------------------------------------
    # DRIFT MONITORING
    # --------------------------------------------------------

    drift_threshold: float

    observation_window_size: int

    min_observations: int

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    prediction_log_path: Path

    drift_report_path: Path

    reference_data_path: Path

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow_tracking_uri: str

    mlflow_model_name: str

    mlflow_production_alias: str


# ============================================================
# SETTINGS FACTORY
# ============================================================


def load_settings() -> Settings:
    """
    Load and validate runtime configuration.

    Defaults are intentionally suitable for local development.
    Production deployments should provide explicit environment
    variables.
    """

    max_rmse = _get_float(
        "MAX_RMSE",
        3.0,
    )

    drift_threshold = _get_float(
        "DRIFT_THRESHOLD",
        0.20,
    )

    observation_window_size = _get_int(
        "OBSERVATION_WINDOW_SIZE",
        50,
    )

    min_observations = _get_int(
        "MIN_OBSERVATIONS",
        10,
    )

    if min_observations > observation_window_size:
        raise ValueError(
            "MIN_OBSERVATIONS cannot be greater than "
            "OBSERVATION_WINDOW_SIZE."
        )

    prediction_log_path = _get_path(
        "PREDICTION_LOG_PATH",
        "data/monitoring/predictions.csv",
    )

    drift_report_path = _get_path(
        "DRIFT_REPORT_PATH",
        "data/monitoring/drift_report.json",
    )

    reference_data_path = _get_path(
        "REFERENCE_DATA_PATH",
        "data/processed/processed_train.csv",
    )

    mlflow_tracking_uri = _get_string(
        "MLFLOW_TRACKING_URI",
        "sqlite:///mlflow.db",
    )

    mlflow_model_name = _get_string(
        "MLFLOW_MODEL_NAME",
        "retail_demand_forecaster",
    )

    mlflow_production_alias = _get_string(
        "MLFLOW_PRODUCTION_ALIAS",
        "production",
    )

    return Settings(
        max_rmse=max_rmse,
        drift_threshold=drift_threshold,
        observation_window_size=observation_window_size,
        min_observations=min_observations,
        prediction_log_path=prediction_log_path,
        drift_report_path=drift_report_path,
        reference_data_path=reference_data_path,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_model_name=mlflow_model_name,
        mlflow_production_alias=mlflow_production_alias,
    )


# ============================================================
# APPLICATION-WIDE SETTINGS INSTANCE
# ============================================================


settings = load_settings()