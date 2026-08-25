"""
Feature drift detection for the retail demand forecasting system.

Responsibilities
----------------
1. Load reference training data.
2. Load observed production prediction data.
3. Select a recent observation window.
4. Validate monitored numerical features.
5. Calculate drift statistics.
6. Produce a structured drift report.
7. Persist the report for downstream orchestration.

This module detects drift only.

It does NOT trigger retraining.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

REFERENCE_DATA_PATH = Path(
    "data/processed/processed_train.csv"
)

LIVE_DATA_PATH = Path(
    "data/monitoring/predictions.csv"
)

DRIFT_REPORT_PATH = Path(
    "data/monitoring/drift_report.json"
)

DRIFT_THRESHOLD = 0.20

MONITORED_FEATURES = [
    "price",
    "promo",
]

# ============================================================
# OBSERVATION WINDOW
# ============================================================

OBSERVATION_WINDOW_SIZE = 50

MIN_OBSERVATIONS = 10

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
# DATA LOADING
# ============================================================


def _load_reference_data() -> pd.DataFrame:
    """
    Load the canonical training/reference dataset.
    """

    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: "
            f"{REFERENCE_DATA_PATH}"
        )

    return pd.read_csv(
        REFERENCE_DATA_PATH
    )


def _load_live_data() -> pd.DataFrame:
    """
    Load observed production prediction data.
    """

    if not LIVE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Live prediction data not found: "
            f"{LIVE_DATA_PATH}"
        )

    return pd.read_csv(
        LIVE_DATA_PATH
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
    3. At least MIN_OBSERVATIONS must be available.
    4. The newest OBSERVATION_WINDOW_SIZE observations are used.

    Returns
    -------
    pd.DataFrame
        Chronologically ordered recent observation window.
    """

    if TIMESTAMP_COLUMN not in live_df.columns:
        raise ValueError(
            f"Live prediction data must contain "
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
            f"Live prediction data contains invalid "
            f"'{TIMESTAMP_COLUMN}' values."
        )

    if len(window) < MIN_OBSERVATIONS:
        raise ValueError(
            "Insufficient live observations for drift "
            f"detection. Required at least "
            f"{MIN_OBSERVATIONS}, found {len(window)}."
        )

    window = (
        window
        .sort_values(
            TIMESTAMP_COLUMN,
            ascending=True,
        )
        .tail(OBSERVATION_WINDOW_SIZE)
        .reset_index(drop=True)
    )

    return window


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
    """

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
        drift_ratio >= DRIFT_THRESHOLD
    )

    return FeatureDriftResult(
        reference_mean=reference_mean,
        live_mean=live_mean,
        drift_ratio=drift_ratio,
        threshold=DRIFT_THRESHOLD,
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

    reference_df = _load_reference_data()

    live_df = _load_live_data()

    # --------------------------------------------------------
    # SELECT RECENT PRODUCTION WINDOW
    # --------------------------------------------------------

    live_window = _select_observation_window(
        live_df
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
        "observation_count": len(
            live_window
        ),
        "observation_window_size": (
            OBSERVATION_WINDOW_SIZE
        ),
        "latest_observation": (
            live_window[
                TIMESTAMP_COLUMN
            ]
            .max()
            .isoformat()
        ),
        "earliest_observation": (
            live_window[
                TIMESTAMP_COLUMN
            ]
            .min()
            .isoformat()
        ),
    }

    # --------------------------------------------------------
    # PERSIST REPORT
    # --------------------------------------------------------

    if save:

        DRIFT_REPORT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with DRIFT_REPORT_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=2,
            )

        print(
            f"Drift report saved to "
            f"{DRIFT_REPORT_PATH}"
        )

    return report