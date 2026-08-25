"""
Feature drift detection for the retail demand forecasting system.

Responsibilities
----------------
1. Load reference training data.
2. Load observed production prediction data.
3. Validate monitored numerical features.
4. Compare configured numerical features.
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
# FEATURE VALIDATION
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

    # Threshold is inclusive.
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
    and observed production prediction data.

    Returns
    -------
    dict
        Structured drift report.

    Raises
    ------
    FileNotFoundError
        If reference or live data is unavailable.

    ValueError
        If monitored features are missing, non-numeric,
        contain null values, or live data is empty.
    """

    reference_df = _load_reference_data()

    live_df = _load_live_data()

    if live_df.empty:
        raise ValueError(
            "Live prediction dataset is empty."
        )

    report: dict[str, dict] = {}

    for feature in MONITORED_FEATURES:

        # ----------------------------------------------------
        # REFERENCE FEATURE CONTRACT
        # ----------------------------------------------------

        if feature not in reference_df.columns:
            raise ValueError(
                f"Feature '{feature}' is missing "
                "from reference data."
            )

        # ----------------------------------------------------
        # LIVE FEATURE CONTRACT
        # ----------------------------------------------------

        if feature not in live_df.columns:
            raise ValueError(
                f"Feature '{feature}' is missing "
                "from live prediction data."
            )

        # ----------------------------------------------------
        # REFERENCE VALIDATION
        # ----------------------------------------------------

        _validate_numeric_feature(
            reference_df,
            feature,
            "reference",
        )

        # ----------------------------------------------------
        # LIVE VALIDATION
        # ----------------------------------------------------

        _validate_numeric_feature(
            live_df,
            feature,
            "live",
        )

        # ----------------------------------------------------
        # DRIFT CALCULATION
        # ----------------------------------------------------

        result = _calculate_feature_drift(
            reference=reference_df[feature],
            live=live_df[feature],
        )

        report[feature] = asdict(
            result
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    drift_detected = any(
        feature_report["drift_detected"]
        for feature_report in report.values()
    )

    report["_summary"] = {
        "drift_detected": drift_detected,
        "monitored_features": len(
            MONITORED_FEATURES
        ),
    }

    # ========================================================
    # PERSIST REPORT
    # ========================================================

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