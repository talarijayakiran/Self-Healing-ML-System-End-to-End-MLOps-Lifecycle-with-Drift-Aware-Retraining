import os
import time
import uuid
from contextlib import asynccontextmanager
from threading import Lock

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from mlflow.tracking import MlflowClient
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from src.monitoring.prediction_logger import log_prediction
from src.serving.schemas import (
    HealthResponse,
    PredictionInput,
    PredictionResponse,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "retail_demand_forecaster"

# Single source of truth for production model selection.
#
# Training creates candidate models.
# Quality gate determines eligibility.
# Promotion assigns this alias.
# Serving consumes this alias.
PRODUCTION_ALIAS = "production"

TEST_MODE = (
    os.getenv(
        "TEST_MODE",
        "false",
    ).lower()
    == "true"
)


# ============================================================
# MODEL STATE
# ============================================================

model = None
model_version = None

model_lock = Lock()

mlflow_client = MlflowClient()


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(force_reload: bool = False):
    """
    Load the model currently assigned to the MLflow production
    alias.

    Production model selection is intentionally controlled by
    the model promotion boundary.

    Lifecycle:

        candidate
            ↓
        quality gate
            ↓
        promotion
            ↓
        production alias
            ↓
        serving

    The serving layer MUST NOT select arbitrary latest model
    versions.

    Parameters
    ----------
    force_reload:
        Force a model reload even if the production alias still
        points to the currently loaded version.

    Returns
    -------
    object | None
        Loaded MLflow pyfunc model, or None in TEST_MODE.

    Raises
    ------
    RuntimeError
        If the production alias cannot be resolved or the
        production model cannot be loaded.
    """

    global model
    global model_version

    # --------------------------------------------------------
    # TEST MODE
    # --------------------------------------------------------

    if TEST_MODE:
        print(
            "TEST_MODE enabled - model loading skipped"
        )

        return None

    # --------------------------------------------------------
    # MODEL LOCK
    # --------------------------------------------------------

    with model_lock:

        # ----------------------------------------------------
        # RESOLVE PRODUCTION ALIAS
        # ----------------------------------------------------

        try:
            production_version = (
                mlflow_client.get_model_version_by_alias(
                    MODEL_NAME,
                    PRODUCTION_ALIAS,
                )
            )

        except Exception as exc:
            raise RuntimeError(
                "Production model alias could not be resolved. "
                f"Model='{MODEL_NAME}', "
                f"alias='{PRODUCTION_ALIAS}'."
            ) from exc

        # ----------------------------------------------------
        # RESOLVE VERSION
        # ----------------------------------------------------

        resolved_version = str(
            production_version.version
        )

        # ----------------------------------------------------
        # HOT-RELOAD CHECK
        # ----------------------------------------------------

        if (
            model is not None
            and model_version == resolved_version
            and not force_reload
        ):
            return model

        # ----------------------------------------------------
        # PRODUCTION MODEL URI
        # ----------------------------------------------------

        model_uri = (
            f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}"
        )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        try:
            loaded_model = mlflow.pyfunc.load_model(
                model_uri
            )

        except Exception as exc:
            raise RuntimeError(
                "Failed to load production model. "
                f"Model='{MODEL_NAME}', "
                f"version='{resolved_version}'."
            ) from exc

        # ----------------------------------------------------
        # UPDATE MODEL STATE
        # ----------------------------------------------------

        model = loaded_model
        model_version = resolved_version

        print(
            "Loaded production model: "
            f"{MODEL_NAME}:{model_version}"
        )

        return model


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print(
        "Starting Self-Healing ML Inference API..."
    )

    try:
        load_model()

        print(
            "Model initialization successful "
            f"(production_version={model_version})"
        )

    except Exception as exc:

        print(
            "Model initialization failed: "
            f"{exc}"
        )

        # Do not terminate the process.
        #
        # /ready will return HTTP 503.
        #
        # This allows Kubernetes, AWS load balancers,
        # container orchestration and health-check systems
        # to correctly determine that this instance is not
        # ready to receive traffic.

    yield

    print(
        "Shutting down Self-Healing ML Inference API..."
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Self-Healing ML Inference API",
    version="1.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PROMETHEUS METRICS
# ============================================================

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    [
        "method",
        "endpoint",
        "http_status",
    ],
)


REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"],
)


PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Model prediction latency",
)


# ============================================================
# FEATURE CONTRACT
# ============================================================

FEATURE_COLUMNS = pd.read_csv(
    "data/processed/processed_inference.csv"
).columns.tolist()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_version=(
            str(model_version)
            if model_version is not None
            else None
        ),
    )


# ============================================================
# READINESS CHECK
# ============================================================

@app.get(
    "/ready",
    response_model=HealthResponse,
)
def readiness():

    if model is None:

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail="Model is not ready",
        )

    return HealthResponse(
        status="ready",
        model_loaded=True,
        model_version=str(model_version),
    )


# ============================================================
# FEATURE BUILDER
# ============================================================

def build_feature_vector(
    input_data: PredictionInput,
) -> pd.DataFrame:
    """
    Build the inference feature vector using the canonical
    feature ordering defined by processed_inference.csv.
    """

    X = pd.DataFrame(
        0,
        columns=FEATURE_COLUMNS,
        index=[0],
    )

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    dt = input_data.date

    if "day" in X.columns:
        X.at[
            0,
            "day",
        ] = dt.day

    if "month" in X.columns:
        X.at[
            0,
            "month",
        ] = dt.month

    # --------------------------------------------------------
    # NUMERICAL FEATURES
    # --------------------------------------------------------

    if "price" in X.columns:
        X.at[
            0,
            "price",
        ] = input_data.price

    if "promo" in X.columns:
        X.at[
            0,
            "promo",
        ] = input_data.promo

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    category_column = (
        f"category_{input_data.category}"
    )

    if category_column in X.columns:
        X.at[
            0,
            category_column,
        ] = 1

    # --------------------------------------------------------
    # REGION
    # --------------------------------------------------------

    region_column = (
        f"region_{input_data.region}"
    )

    if region_column in X.columns:
        X.at[
            0,
            region_column,
        ] = 1

    return X


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    input_data: PredictionInput,
):

    start_time = time.time()

    status_code = 200

    request_id = str(
        uuid.uuid4()
    )

    try:

        # ====================================================
        # ENSURE PRODUCTION MODEL IS AVAILABLE
        # ====================================================

        try:
            current_model = load_model()

        except Exception as exc:

            status_code = 503

            raise HTTPException(
                status_code=(
                    status.HTTP_503_SERVICE_UNAVAILABLE
                ),
                detail=(
                    "Prediction service is not ready"
                ),
            ) from exc

        if current_model is None:

            status_code = 503

            raise HTTPException(
                status_code=(
                    status.HTTP_503_SERVICE_UNAVAILABLE
                ),
                detail=(
                    "Prediction model is not available"
                ),
            )

        # ====================================================
        # BUILD FEATURES
        # ====================================================

        features = build_feature_vector(
            input_data
        )

        # ====================================================
        # MODEL INFERENCE
        # ====================================================

        prediction_start = time.time()

        try:

            prediction = current_model.predict(
                features
            )[0]

        except Exception as exc:

            status_code = 500

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail="Model prediction failed",
            ) from exc

        # ----------------------------------------------------
        # RECORD PREDICTION LATENCY
        # ----------------------------------------------------

        PREDICTION_LATENCY.observe(
            time.time()
            - prediction_start
        )

        # ====================================================
        # CANONICAL PREDICTION LOGGING
        # ====================================================

        log_prediction(
            request_id=request_id,
            model_version=str(model_version),
            date=input_data.date.isoformat(),
            category=input_data.category,
            region=input_data.region,
            price=float(input_data.price),
            promo=int(input_data.promo),
            prediction=float(prediction),
        )

        # ====================================================
        # API RESPONSE
        # ====================================================

        return PredictionResponse(
            predicted_sales=round(
                float(prediction),
                2,
            ),
            model_version=str(
                model_version
            ),
            request_id=request_id,
        )

    finally:

        # ====================================================
        # REQUEST LATENCY
        # ====================================================

        REQUEST_LATENCY.labels(
            endpoint="/predict",
        ).observe(
            time.time()
            - start_time
        )

        # ====================================================
        # REQUEST COUNT
        # ====================================================

        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict",
            http_status=status_code,
        ).inc()


# ============================================================
# PROMETHEUS METRICS ENDPOINT
# ============================================================

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )