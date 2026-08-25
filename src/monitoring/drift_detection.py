"""
Feature drift detection for the retail demand forecasting system.

Responsibilities
----------------
1. Load reference training data.
2. Load observed production prediction data.
3. Select a recent observation window.
4. Validate monitored numerical features.
5. Calculate drift statistics.
6. Build auditable observation-window metadata.
7. Produce a structured drift report.
8. Persist the report for downstream orchestration.

Runtime configuration
--------------------
Runtime configuration is owned by src.config.settings.

This module detects drift only.

It does NOT trigger retraining.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pandas as pd

from src.config import settings as settings_module


# ============================================================
# STATIC DOMAIN CONFIGURATION
# ============================================================

MONITORED_FEATURES = [
    "price",
    "promo",
]

TIMESTAMP_COLUMN = "timestamp"


# ============================================================
# RESULT CONTRACT
# ============================================================


@dataclass(frozen=True)
class FeatureDriftResult:
    """
    Drift result for one monitored feature.
    """

    reference_mean: float
    live_mean: float
    drift_ratio: float
    threshold: float
    drift_detected: bool


# ============================================================
# RUNTIME CONFIGURATION
# ============================================================


def _get_runtime_settings():
    """
    Return the current validated runtime settings.

    The settings module is imported rather than the Settings
    instance directly so tests and controlled runtime environments
    can replace the application-wide settings object safely.
    """

    return settings_module.settings


# ============================================================
# DATA LOADING
# ============================================================


def _load_reference_data() -> pd.DataFrame:
    """
    Load the canonical training/reference dataset.
    """

    reference_data_path = (
        _get_runtime_settings()
        .reference_data_path
    )

    if not reference_data_path.exists():
        raise FileNotFoundError(
            "Reference dataset not found: "
            f"{reference_data_path}"
        )

    return pd.read_csv(
        reference_data_path
    )


def _load_live_data() -> pd.DataFrame:
    """
    Load observed production prediction data.

    The prediction log path is the runtime-configured source
    for production observations.
    """

    prediction_log_path = (
        _get_runtime_settings()
        .prediction_log_path
    )

    if not prediction_log_path.exists():
        raise FileNotFoundError(
            "Live prediction data not found: "
            f"{prediction_log_path}"
        )

    return pd.read_csv(
        prediction_log_path
    )


# ============================================================
# OBSERVATION WINDOW
# ============================================================


def _select_observation_window(
    live_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the most recent production observations.

    The complete prediction history is retained on disk, but
    drift detection operates only on the most recent configured
    observation window.

    Requirements
    ------------
    1. Timestamp column must exist.
    2. Timestamps must be parseable.
    3. At least the configured minimum observations must exist.
    4. The newest configured observation-window records are used.

    Returns
    -------
    pd.DataFrame
        Chronologically ordered recent observation window.
    """

    runtime_settings = _get_runtime_settings()

    observation_window_size = (
        runtime_settings.observation_window_size
    )

    min_observations = (
        runtime_settings.min_observations
    )

    if TIMESTAMP_COLUMN not in live_df.columns:
        raise ValueError(
            "Live prediction data must contain "
            f"'{TIMESTAMP_COLUMN}' column."
        )

    if live_df.empty:
        raise ValueError(
            "Live prediction dataset is empty."
        )

    window = live_df.copy()

    window[TIMESTAMP_COLUMN] = pd.to_datetime(
        window[TIMESTAMP_COLUMN],
        errors="coerce",
        utc=True,
    )

    if window[TIMESTAMP_COLUMN].isnull().any():
        raise ValueError(
            "Live prediction data contains invalid "
            f"'{TIMESTAMP_COLUMN}' values."
        )

    if len(window) < min_observations:
        raise ValueError(
            "Insufficient live observations for drift "
            "detection. Required at least "
            f"{min_observations}, found {len(window)}."
        )

    window = (
        window
        .sort_values(
            TIMESTAMP_COLUMN,
            ascending=True,
        )
        .tail(observation_window_size)
        .reset_index(drop=True)
    )

    return window


# ============================================================
# OBSERVATION WINDOW METADATA
# ============================================================


def _build_observation_window_metadata(
    observation_window: pd.DataFrame,
) -> dict:
    """
    Build metadata describing the observations used for
    drift detection.

    The metadata makes the drift decision auditable by recording:

    - observation count
    - configured window size
    - minimum required observations
    - whether the configured window was completely filled
    - oldest observation timestamp
    - newest observation timestamp
    - model versions represented in the window
    """

    runtime_settings = _get_runtime_settings()

    observation_window_size = (
        runtime_settings.observation_window_size
    )

    min_observations = (
        runtime_settings.min_observations
    )

    if observation_window.empty:
        raise ValueError(
            "Observation window cannot be empty."
        )

    if TIMESTAMP_COLUMN not in observation_window.columns:
        raise ValueError(
            "Observation window must contain "
            f"'{TIMESTAMP_COLUMN}' column."
        )

    timestamps = observation_window[
        TIMESTAMP_COLUMN
    ]

    model_versions: list[str] = []

    if "model_version" in observation_window.columns:
        model_versions = sorted(
            observation_window[
                "model_version"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

    return {
        "observation_count": int(
            len(observation_window)
        ),
        "observation_window_size": int(
            observation_window_size
        ),
        "minimum_observations": int(
            min_observations
        ),
        "window_complete": (
            len(observation_window)
            == observation_window_size
        ),
        "oldest_observation_timestamp": (
            timestamps.min().isoformat()
        ),
        "newest_observation_timestamp": (
            timestamps.max().isoformat()
        ),
        "model_versions": model_versions,
    }


# ============================================================
# NUMERICAL VALIDATION
# ============================================================


def _validate_numeric_feature(
    df: pd.DataFrame,
    feature: str,
    dataset_name: str,
) -> None:
    """
    Validate that a monitored feature is numeric and contains
    no null values.
    """

    if feature not in df.columns:
        raise ValueError(
            f"Feature '{feature}' is missing "
            f"from {dataset_name} data."
        )

    if not pd.api.types.is_numeric_dtype(
        df[feature]
    ):
        raise ValueError(
            f"Feature '{feature}' in "
            f"{dataset_name} data must be numeric."
        )

    if df[feature].isnull().any():
        raise ValueError(
            f"Feature '{feature}' in "
            f"{dataset_name} data contains null values."
        )


# ============================================================
# DRIFT CALCULATION
# ============================================================


def _calculate_drift_ratio(
    reference_mean: float,
    live_mean: float,
) -> float:
    """
    Calculate normalized mean difference.

    Formula:

        |live - reference| / |reference|

    A zero reference mean is handled explicitly.
    """

    if reference_mean == 0:

        if live_mean == 0:
            return 0.0

        return float("inf")

    return abs(
        live_mean - reference_mean
    ) / abs(reference_mean)


def _calculate_feature_drift(
    reference: pd.Series,
    live: pd.Series,
) -> FeatureDriftResult:
    """
    Calculate drift statistics for one numerical feature.

    The drift threshold comes from validated runtime configuration.
    """

    drift_threshold = (
        _get_runtime_settings()
        .drift_threshold
    )

    reference_mean = float(
        reference.mean()
    )

    live_mean = float(
        live.mean()
    )

    drift_ratio = _calculate_drift_ratio(
        reference_mean,
        live_mean,
    )

    drift_detected = (
        drift_ratio >= drift_threshold
    )

    return FeatureDriftResult(
        reference_mean=reference_mean,
        live_mean=live_mean,
        drift_ratio=drift_ratio,
        threshold=drift_threshold,
        drift_detected=drift_detected,
    )


# ============================================================
# DRIFT DETECTION
# ============================================================


def detect_drift(
    *,
    save: bool = True,
) -> dict:
    """
    Detect feature drift between reference training data
    and the recent production observation window.

    Runtime paths and drift policy are resolved from the
    validated application settings.

    Returns
    -------
    dict
        Structured drift report.

    Raises
    ------
    FileNotFoundError
        If reference or live data is unavailable.

    ValueError
        If monitored features are missing, invalid, non-numeric,
        contain null values, timestamps are invalid, or there
        are insufficient live observations.
    """

    runtime_settings = _get_runtime_settings()

    reference_df = _load_reference_data()

    live_df = _load_live_data()

    # --------------------------------------------------------
    # SELECT RECENT PRODUCTION WINDOW
    # --------------------------------------------------------

    live_window = _select_observation_window(
        live_df
    )

    # --------------------------------------------------------
    # BUILD OBSERVATION WINDOW METADATA
    # --------------------------------------------------------

    observation_metadata = (
        _build_observation_window_metadata(
            live_window
        )
    )

    report: dict[str, dict] = {}

    # --------------------------------------------------------
    # VALIDATE AND CALCULATE DRIFT
    # --------------------------------------------------------

    for feature in MONITORED_FEATURES:

        _validate_numeric_feature(
            reference_df,
            feature,
            "reference",
        )

        _validate_numeric_feature(
            live_window,
            feature,
            "live",
        )

        result = _calculate_feature_drift(
            reference=reference_df[feature],
            live=live_window[feature],
        )

        report[feature] = asdict(
            result
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    drift_detected = any(
        feature_report["drift_detected"]
        for feature_report in report.values()
    )

    report["_summary"] = {
        "drift_detected": drift_detected,
        "monitored_features": len(
            MONITORED_FEATURES
        ),
        **observation_metadata,
    }

    # --------------------------------------------------------
    # PERSIST REPORT
    # --------------------------------------------------------

    if save:

        drift_report_path = (
            runtime_settings.drift_report_path
        )

        drift_report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with drift_report_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=2,
        )

        print(
            "Drift report saved to "
            f"{drift_report_path}"
        )

    return report


# ============================================================
# CLI ENTRYPOINT
# ============================================================


if __name__ == "__main__":
    detect_drift()