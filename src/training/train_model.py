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
7. Register only quality-approved candidates.
8. Persist promotion eligibility metadata on the MLflow
   model version.
9. Return a structured TrainingResult.

Model promotion is intentionally NOT handled here.

Promotion belongs to src.registry.promotion.

Configuration ownership
-----------------------
Runtime/deployment configuration is owned by
src.config.settings.

ML/data contracts are owned by src.config.schema.

Training algorithm parameters remain explicit training-policy
constants in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.config.schema import (
    MODEL_FEATURES,
    TARGET_COLUMN,
)
from src.config.settings import settings
from src.evaluation.quality_gate import (
    evaluate_model_quality,
)


# ============================================================
# TRAINING POLICY
# ============================================================

EXPERIMENT_NAME = (
    "Retail Demand Forecasting"
)

RANDOM_STATE = 42

TEST_SIZE = 0.25

N_ESTIMATORS = 200


# ============================================================
# MLflow MODEL-VERSION TAGS
# ============================================================

QUALITY_GATE_PASSED_TAG = (
    "quality_gate_passed"
)

MODEL_ROLE_TAG = (
    "model_role"
)

MODEL_ROLE_CANDIDATE = (
    "candidate"
)


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
# RUNTIME CONFIGURATION
# ============================================================


def configure_mlflow() -> None:
    """
    Configure MLflow using the current runtime settings.

    MLflow tracking configuration belongs to deployment/runtime
    configuration rather than the training algorithm itself.
    """

    mlflow.set_tracking_uri(
        settings.mlflow_tracking_uri
    )


# ============================================================
# DATA LOADING
# ============================================================


def load_training_data() -> pd.DataFrame:
    """
    Load and validate the canonical training dataset.

    The dataset path comes from runtime configuration.

    The required columns and their ordering come from
    src.config.schema.
    """

    data_path = (
        settings.reference_data_path
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {data_path}"
        )

    df = pd.read_csv(
        data_path
    )

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
            "Training dataset violates the model "
            "feature contract. "
            f"Missing columns: {missing_columns}"
        )

    unexpected_columns = [
        column
        for column in df.columns
        if column not in required_columns
    ]

    if unexpected_columns:
        raise ValueError(
            "Training dataset contains unexpected "
            f"columns: {unexpected_columns}"
        )

    # --------------------------------------------------------
    # Canonical feature ordering
    # --------------------------------------------------------

    df = df[
        [
            *MODEL_FEATURES,
            TARGET_COLUMN,
        ]
    ]

    # --------------------------------------------------------
    # Dataset safety
    # --------------------------------------------------------

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
    Construct the deterministic candidate model.

    These parameters are training-policy decisions and are
    intentionally not sourced from deployment configuration.
    """

    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# MODEL-VERSION METADATA
# ============================================================


def persist_model_version_metadata(
    *,
    model_name: str,
    model_version: str,
    quality_gate_passed: bool,
    client: MlflowClient,
) -> None:
    """
    Persist lifecycle eligibility metadata directly onto
    the registered MLflow model version.

    This is intentionally different from mlflow.set_tag(),
    which creates a run-level tag.

    Promotion reads model-version metadata, so the eligibility
    decision must be persisted at the model-version level.
    """

    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key=QUALITY_GATE_PASSED_TAG,
        value=str(
            quality_gate_passed
        ).lower(),
    )

    if quality_gate_passed:

        client.set_model_version_tag(
            name=model_name,
            version=model_version,
            key=MODEL_ROLE_TAG,
            value=MODEL_ROLE_CANDIDATE,
        )


# ============================================================
# TRAINING
# ============================================================


def train_and_log() -> TrainingResult:
    """
    Train, evaluate and register a candidate model.

    A model is registered as a candidate only after it passes
    the quality gate.

    Production promotion is NOT performed here.
    """

    # --------------------------------------------------------
    # RUNTIME CONFIGURATION
    # --------------------------------------------------------

    configure_mlflow()

    model_name = (
        settings.mlflow_model_name
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_training_data()

    X = df[MODEL_FEATURES]

    y = df[TARGET_COLUMN]

    # --------------------------------------------------------
    # DATASET SIZE SAFETY
    # --------------------------------------------------------

    if len(df) < 2:
        raise ValueError(
            "At least two training samples are required."
        )

    effective_test_size = (
        0.4
        if len(df) < 10
        else TEST_SIZE
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X,
        y,
        test_size=effective_test_size,
        random_state=RANDOM_STATE,
    )

    if X_train.empty:
        raise ValueError(
            "Training split contains no samples."
        )

    if X_validation.empty:
        raise ValueError(
            "Validation split contains no samples."
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # MLFLOW EXPERIMENT
    # --------------------------------------------------------

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run() as run:

        # ----------------------------------------------------
        # PARAMETERS
        # ----------------------------------------------------

        mlflow.log_params(
            {
                "model_type": (
                    "RandomForestRegressor"
                ),
                "n_estimators": N_ESTIMATORS,
                "random_state": RANDOM_STATE,
                "test_size": effective_test_size,
                "feature_count": len(
                    MODEL_FEATURES
                ),
                "training_rows": len(
                    X_train
                ),
                "validation_rows": len(
                    X_validation
                ),
                "quality_gate_threshold": (
                    settings.max_rmse
                ),
                "training_data_path": str(
                    settings.reference_data_path
                ),
                "mlflow_model_name": model_name,
            }
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        predictions = model.predict(
            X_validation
        )

        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------

        rmse = float(
            mean_squared_error(
                y_validation,
                predictions,
            )
            ** 0.5
        )

        # ----------------------------------------------------
        # QUALITY GATE
        # ----------------------------------------------------

        quality_result = (
            evaluate_model_quality(
                rmse=rmse,
            )
        )

        # ----------------------------------------------------
        # LOG QUALITY-GATE RESULT AT RUN LEVEL
        # ----------------------------------------------------

        mlflow.log_metric(
            "rmse",
            rmse,
        )

        mlflow.set_tag(
            "feature_contract",
            "MODEL_FEATURES",
        )

        mlflow.set_tag(
            "model_role",
            MODEL_ROLE_CANDIDATE,
        )

        mlflow.set_tag(
            "pipeline_stage",
            "training",
        )

        mlflow.set_tag(
            "quality_gate_passed",
            str(
                quality_result.passed
            ).lower(),
        )

        # ----------------------------------------------------
        # QUALITY GATE FAILURE
        # ----------------------------------------------------

        if not quality_result.passed:

            print(
                "MODEL QUALITY GATE FAILED"
            )

            print(
                "Candidate model rejected: "
                f"RMSE {rmse:.4f} > "
                "threshold "
                f"{quality_result.max_rmse:.4f}."
            )

            raise RuntimeError(
                "Candidate model failed the "
                "model quality gate."
            )

        # ----------------------------------------------------
        # QUALITY GATE SUCCESS
        # ----------------------------------------------------

        print(
            "MODEL QUALITY GATE PASSED"
        )

        print(
            "Candidate passed quality gate: "
            f"RMSE {rmse:.4f} <= "
            "threshold "
            f"{quality_result.max_rmse:.4f}."
        )

        # ----------------------------------------------------
        # MODEL REGISTRATION
        # ----------------------------------------------------

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=model_name,
        )

        # ----------------------------------------------------
        # REGISTERED MODEL VERSION
        # ----------------------------------------------------

        registered_version = getattr(
            model_info,
            "registered_model_version",
            None,
        )

        if not registered_version:
            raise RuntimeError(
                "Model registration succeeded but "
                "MLflow did not return a model version."
            )

        model_version = str(
            registered_version
        )

        # ----------------------------------------------------
        # PERSIST MODEL-VERSION METADATA
        # ----------------------------------------------------

        mlflow_client = MlflowClient()

        persist_model_version_metadata(
            model_name=model_name,
            model_version=model_version,
            quality_gate_passed=True,
            client=mlflow_client,
        )

        # ----------------------------------------------------
        # STRUCTURED RESULT
        # ----------------------------------------------------

        result = TrainingResult(
            run_id=run.info.run_id,
            model_name=model_name,
            model_version=model_version,
            rmse=rmse,
            quality_gate_passed=True,
            training_rows=len(
                X_train
            ),
            validation_rows=len(
                X_validation
            ),
            feature_count=len(
                MODEL_FEATURES
            ),
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

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
        "Quality gate passed: "
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