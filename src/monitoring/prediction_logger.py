"""
Production prediction logging for the retail demand forecasting system.

Responsibilities
----------------
1. Persist production prediction events.
2. Enforce a stable monitoring schema.
3. Capture model and request metadata.
4. Validate monitoring data before persistence.
5. Serialize concurrent writes safely.
6. Keep prediction logging independent from drift detection.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

LOG_PATH = Path(
    "data/monitoring/predictions.csv"
)

PREDICTION_COLUMNS = [
    "timestamp",
    "request_id",
    "model_version",
    "date",
    "category",
    "region",
    "price",
    "promo",
    "prediction",
]


# ============================================================
# PROCESS-LOCAL WRITE LOCK
# ============================================================

# Protects concurrent writes from multiple FastAPI requests
# handled by the same application process.
#
# NOTE:
# This is sufficient for the current single-container
# deployment. In a multi-replica production deployment,
# CSV should be replaced by a shared durable datastore.
LOG_LOCK = Lock()


# ============================================================
# INITIALIZATION
# ============================================================

LOG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _validate_required_string(
    value: object,
    field_name: str,
) -> None:
    """
    Validate that a required field is a non-empty string.
    """

    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string."
        )

    if not value.strip():
        raise ValueError(
            f"{field_name} must be provided."
        )


def _validate_finite_number(
    value: object,
    field_name: str,
) -> None:
    """
    Validate that a numeric monitoring value is finite.
    """

    try:
        numeric_value = float(value)

    except (TypeError, ValueError) as exc:

        raise ValueError(
            f"{field_name} must be numeric."
        ) from exc

    if not math.isfinite(numeric_value):

        raise ValueError(
            f"{field_name} must be finite."
        )


# ============================================================
# PREDICTION LOGGING
# ============================================================

def log_prediction(
    *,
    request_id: str,
    model_version: str,
    date: str,
    category: str,
    region: str,
    price: float,
    promo: int,
    prediction: float,
) -> None:
    """
    Persist one production prediction event.

    The function accepts explicit parameters rather than an
    arbitrary dictionary so that the monitoring contract
    remains stable.

    Canonical schema:

        timestamp
        request_id
        model_version
        date
        category
        region
        price
        promo
        prediction
    """

    # ========================================================
    # REQUIRED STRING VALIDATION
    # ========================================================

    _validate_required_string(
        request_id,
        "request_id",
    )

    _validate_required_string(
        model_version,
        "model_version",
    )

    _validate_required_string(
        date,
        "date",
    )

    _validate_required_string(
        category,
        "category",
    )

    _validate_required_string(
        region,
        "region",
    )

    # ========================================================
    # NUMERIC VALIDATION
    # ========================================================

    _validate_finite_number(
        price,
        "price",
    )

    _validate_finite_number(
        prediction,
        "prediction",
    )

    if float(price) <= 0:

        raise ValueError(
            "price must be greater than 0."
        )

    # ========================================================
    # PROMOTION VALIDATION
    # ========================================================

    if promo not in (0, 1):

        raise ValueError(
            "promo must be either 0 or 1."
        )

    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    # ========================================================
    # RECORD
    # ========================================================

    record = {
        "timestamp": timestamp,
        "request_id": request_id,
        "model_version": model_version,
        "date": date,
        "category": category,
        "region": region,
        "price": float(price),
        "promo": int(promo),
        "prediction": float(prediction),
    }

    # ========================================================
    # DATAFRAME
    # ========================================================

    row = pd.DataFrame(
        [record],
        columns=PREDICTION_COLUMNS,
    )

    # ========================================================
    # PERSISTENCE
    # ========================================================

    with LOG_LOCK:

        file_exists = LOG_PATH.exists()

        row.to_csv(
            LOG_PATH,
            mode="a",
            header=not file_exists,
            index=False,
        )