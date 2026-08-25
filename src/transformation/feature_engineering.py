# src/transformation/feature_engineering.py

"""
Feature engineering for the retail demand forecasting pipeline.

This module owns the transformation from validated raw data into the
canonical model feature representation.

The final feature schema is defined centrally in:
    src.config.schema.MODEL_FEATURES

Training and inference must consume the same ordered feature contract.
"""

from pathlib import Path

import pandas as pd

from src.config.schema import (
    DERIVED_COLUMNS,
    MODEL_FEATURES,
    TARGET_COLUMN,
)


INPUT_PATH = Path("data/processed/validated_data.csv")
TRAIN_PATH = Path("data/processed/processed_train.csv")
INFER_PATH = Path("data/processed/processed_inference.csv")


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform validated raw data into the canonical model feature matrix.

    The returned DataFrame contains exactly MODEL_FEATURES in exactly
    the same order.
    """

    data = df.copy()

    # ---------------------------------------------------------
    # DATE FEATURES
    # ---------------------------------------------------------

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    data["day"] = data["date"].dt.day
    data["month"] = data["date"].dt.month

    data = data.drop(columns=["date"])

    # ---------------------------------------------------------
    # CATEGORICAL ENCODING
    # ---------------------------------------------------------

    data = pd.get_dummies(
        data,
        columns=["category", "region"],
        dtype=int,
    )

    # ---------------------------------------------------------
    # TARGET SEPARATION
    # ---------------------------------------------------------

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing "
            "from the validated dataset."
        )

    features = data.drop(
        columns=[TARGET_COLUMN],
    )

    # ---------------------------------------------------------
    # CANONICAL FEATURE CONTRACT
    # ---------------------------------------------------------

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Feature contract violation. "
            f"Missing model features: {missing_features}"
        )

    # Ignore unexpected columns and enforce the canonical order.
    features = features.reindex(
        columns=MODEL_FEATURES,
        fill_value=0,
    )

    # ---------------------------------------------------------
    # FINAL CONTRACT VALIDATION
    # ---------------------------------------------------------

    if list(features.columns) != MODEL_FEATURES:
        raise ValueError(
            "Feature contract violation. "
            "Generated feature columns do not match MODEL_FEATURES."
        )

    if features.isnull().any().any():
        raise ValueError(
            "Feature engineering produced null values."
        )

    return features


def run_feature_engineering() -> None:
    """
    Execute the feature engineering stage.

    Produces:

        processed_train.csv
            canonical features + target

        processed_inference.csv
            canonical features only
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Validated input data not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    features = _build_features(df)

    target = df[TARGET_COLUMN].copy()

    train_df = pd.concat(
        [features, target],
        axis=1,
    )

    TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        TRAIN_PATH,
        index=False,
    )

    features.to_csv(
        INFER_PATH,
        index=False,
    )

    print("Feature engineering completed successfully.")
    print(f"Train shape: {train_df.shape}")
    print(f"Inference shape: {features.shape}")
    print(f"Feature count: {len(MODEL_FEATURES)}")
    print("Canonical model features:")
    for index, feature in enumerate(MODEL_FEATURES, start=1):
        print(f"  {index:02d}. {feature}")


if __name__ == "__main__":
    run_feature_engineering()