# src/training/train_model.py

"""
Model training pipeline for the retail demand forecasting system.

Responsibilities
----------------
1. Load the canonical training dataset.
2. Validate the model feature contract.
3. Split the data deterministically.
4. Train a candidate RandomForest model.
5. Evaluate the candidate model.
6. Apply the model quality gate.
7. Log training metadata and metrics to MLflow.
8. Register the candidate model only if the quality gate passes.
9. Return a structured TrainingResult.

Model promotion is intentionally NOT handled here.

The quality gate decides whether the candidate satisfies the
minimum model quality policy. Production promotion belongs to
a separate lifecycle stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.config.schema import MODEL_FEATURES, TARGET_COLUMN
from src.evaluation.quality_gate import evaluate_model_quality


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/processed/processed_train.csv"
)

EXPERIMENT_NAME = (
    "Retail Demand Forecasting"
)

MODEL_NAME = (
    "retail_demand_forecaster"
)

RANDOM_STATE = 42

TEST_SIZE = 0.25

N_ESTIMATORS = 200


# ============================================================
# TRAINING RESULT
# ============================================================


@dataclass(frozen=True)
class TrainingResult:
    """
    Immutable result returned by the training pipeline.
    """

    run_id: str
    model_name: str
    model_version: str | None
    rmse: float
    quality_gate_passed: bool
    training_rows: int
    validation_rows: int
    feature_count: int


# ============================================================
# DATA LOADING
# ============================================================


def load_training_data() -> pd.DataFrame:
    """
    Load the canonical training dataset.

    Returns
    -------
    pd.DataFrame
        Training dataset containing MODEL_FEATURES and
        TARGET_COLUMN.

    Raises
    ------
    FileNotFoundError
        If the processed training dataset does not exist.

    ValueError
        If the dataset violates the training feature contract.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        *MODEL_FEATURES,
        TARGET_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Training dataset violates the model feature contract. "
            f"Missing columns: {missing_columns}"
        )

    unexpected_columns = [
        column
        for column in df.columns
        if column not in required_columns
    ]

    if unexpected_columns:
        raise ValueError(
            "Training dataset contains unexpected columns: "
            f"{unexpected_columns}"
        )

    # Enforce canonical feature ordering.
    df = df[
        [
            *MODEL_FEATURES,
            TARGET_COLUMN,
        ]
    ]

    if df.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if df.isnull().any().any():
        raise ValueError(
            "Training dataset contains null values."
        )

    return df


# ============================================================
# MODEL CREATION
# ============================================================


def build_model() -> RandomForestRegressor:
    """
    Construct the candidate model.

    The model configuration is intentionally deterministic so that
    experiments are reproducible.
    """

    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# TRAINING
# ============================================================


def train_and_log() -> TrainingResult:
    """
    Train, evaluate, quality-check, and register a candidate model.

    The candidate model is registered only when it passes the
    configured model quality gate.

    This function does NOT promote the candidate to production.

    Returns
    -------
    TrainingResult
        Structured metadata describing the candidate training run.

    Raises
    ------
    RuntimeError
        If the candidate fails the model quality gate.
    """

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    df = load_training_data()

    X = df[MODEL_FEATURES]
    y = df[TARGET_COLUMN]

    # ---------------------------------------------------------
    # DATASET SIZE SAFETY
    # ---------------------------------------------------------

    if len(df) < 2:
        raise ValueError(
            "At least two training samples are required."
        )

    effective_test_size = (
        0.4
        if len(df) < 10
        else TEST_SIZE
    )

    # ---------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # ---------------------------------------------------------

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X,
            y,
            test_size=effective_test_size,
            random_state=RANDOM_STATE,
        )
    )

    if X_train.empty:
        raise ValueError(
            "Training split contains no samples."
        )

    if X_validation.empty:
        raise ValueError(
            "Validation split contains no samples."
        )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    model = build_model()

    # ---------------------------------------------------------
    # MLFLOW EXPERIMENT
    # ---------------------------------------------------------

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run() as run:

        # -----------------------------------------------------
        # LOG PARAMETERS
        # -----------------------------------------------------

        mlflow.log_params(
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": N_ESTIMATORS,
                "random_state": RANDOM_STATE,
                "test_size": effective_test_size,
                "feature_count": len(MODEL_FEATURES),
                "training_rows": len(X_train),
                "validation_rows": len(X_validation),
            }
        )

        # -----------------------------------------------------
        # TRAIN
        # -----------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        # -----------------------------------------------------
        # EVALUATE
        # -----------------------------------------------------

        predictions = model.predict(
            X_validation
        )

        rmse = float(
            mean_squared_error(
                y_validation,
                predictions,
            )
            ** 0.5
        )

        # -----------------------------------------------------
        # METRICS
        # -----------------------------------------------------

        mlflow.log_metric(
            "rmse",
            rmse,
        )

        # -----------------------------------------------------
        # MODEL QUALITY GATE
        # -----------------------------------------------------
        #
        # IMPORTANT:
        #
        # The candidate must pass the quality gate BEFORE
        # registration.
        #
        # This prevents an invalid candidate from entering
        # the model registry.
        # -----------------------------------------------------

        quality_result = evaluate_model_quality(
            rmse=rmse,
        )

        mlflow.log_metric(
            "quality_gate_passed",
            int(quality_result.passed),
        )

        mlflow.log_metric(
            "max_allowed_rmse",
            quality_result.max_rmse,
        )

        mlflow.set_tag(
            "quality_gate_status",
            "passed"
            if quality_result.passed
            else "failed",
        )

        mlflow.set_tag(
            "quality_gate_reason",
            quality_result.reason,
        )

        if not quality_result.passed:
            print(
                "MODEL QUALITY GATE FAILED"
            )

            print(
                quality_result.reason
            )

            raise RuntimeError(
                "Candidate model rejected by quality gate: "
                f"{quality_result.reason}"
            )

        print(
            "MODEL QUALITY GATE PASSED"
        )

        print(
            quality_result.reason
        )

        # -----------------------------------------------------
        # FEATURE CONTRACT METADATA
        # -----------------------------------------------------

        mlflow.set_tag(
            "feature_contract",
            "MODEL_FEATURES",
        )

        mlflow.set_tag(
            "model_role",
            "candidate",
        )

        mlflow.set_tag(
            "pipeline_stage",
            "training",
        )

        # -----------------------------------------------------
        # MODEL REGISTRATION
        # -----------------------------------------------------
        #
        # Registration happens ONLY after the quality gate
        # has passed.
        # -----------------------------------------------------

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
        )

        # -----------------------------------------------------
        # TRAINING RESULT
        # -----------------------------------------------------

        run_id = run.info.run_id

        model_version = None

        registered_version = getattr(
            model_info,
            "registered_model_version",
            None,
        )

        if registered_version:
            model_version = str(
                registered_version
            )

        result = TrainingResult(
            run_id=run_id,
            model_name=MODEL_NAME,
            model_version=model_version,
            rmse=rmse,
            quality_gate_passed=quality_result.passed,
            training_rows=len(X_train),
            validation_rows=len(X_validation),
            feature_count=len(MODEL_FEATURES),
        )

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    print(
        "Candidate model trained successfully."
    )

    print(
        f"Run ID: {result.run_id}"
    )

    print(
        f"Model: {result.model_name}"
    )

    print(
        f"Model version: {result.model_version}"
    )

    print(
        f"RMSE: {result.rmse:.4f}"
    )

    print(
        f"Quality gate passed: "
        f"{result.quality_gate_passed}"
    )

    print(
        f"Training rows: "
        f"{result.training_rows}"
    )

    print(
        f"Validation rows: "
        f"{result.validation_rows}"
    )

    print(
        f"Feature count: "
        f"{result.feature_count}"
    )

    return result


# ============================================================
# CLI ENTRYPOINT
# ============================================================


if __name__ == "__main__":
    train_and_log()