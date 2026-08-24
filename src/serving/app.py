import os
import time
import uuid
from contextlib import asynccontextmanager
from threading import Lock

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, status
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


# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "retail_demand_forecaster"

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/app/model",
)

TEST_MODE = (
    os.getenv(
        "TEST_MODE",
        "false",
    ).lower()
    == "true"
)


# =====================================================
# MODEL STATE
# =====================================================

model = None
model_version = None

model_lock = Lock()

mlflow_client = MlflowClient()


# =====================================================
# LOAD MODEL
# HOT-RELOAD SAFE
# =====================================================

def load_model(force_reload: bool = False):
    global model, model_version

    if TEST_MODE:
        print(
            "⚠ TEST_MODE enabled — model loading skipped"
        )
        return None

    with model_lock:

        versions = mlflow_client.get_latest_versions(
            MODEL_NAME,
            stages=["None"],
        )

        if not versions:
            raise RuntimeError(
                "No registered models found in MLflow"
            )

        latest = versions[0]

        if (
            model is not None
            and model_version == latest.version
            and not force_reload
        ):
            return model

        model_uri = (
            f"models:/{MODEL_NAME}/{latest.version}"
        )

        model = mlflow.pyfunc.load_model(
            model_uri
        )

        model_version = latest.version

        print(
            f"Loaded model version: "
            f"{model_version}"
        )

        return model


# =====================================================
# FASTAPI LIFESPAN
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting Self-Healing ML Inference API...")

    try:
        load_model()
        print(
            f"Model initialization successful "
            f"(version={model_version})"
        )

    except Exception as exc:
        print(
            f"Model initialization failed: {exc}"
        )

        # Do not crash the process here.
        #
        # The application can still start and
        # /ready will correctly report 503.
        #
        # This allows Kubernetes/AWS/load balancers
        # to detect that the instance is not ready.

    yield

    print(
        "Shutting down Self-Healing ML Inference API..."
    )


# =====================================================
# FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="Self-Healing ML Inference API",
    version="1.0",
    lifespan=lifespan,
)


# =====================================================
# PROMETHEUS METRICS
# =====================================================

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


# =====================================================
# FEATURE TEMPLATE
# =====================================================

FEATURE_COLUMNS = pd.read_csv(
    "data/processed/processed_inference.csv"
).columns.tolist()


# =====================================================
# HEALTH CHECK
# =====================================================

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


# =====================================================
# READINESS CHECK
# =====================================================

@app.get(
    "/ready",
    response_model=HealthResponse,
)
def readiness():

    if model is None:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready",
        )

    return HealthResponse(
        status="ready",
        model_loaded=True,
        model_version=str(model_version),
    )


# =====================================================
# FEATURE BUILDER
# =====================================================

def build_feature_vector(
    input_data: PredictionInput,
) -> pd.DataFrame:

    X = pd.DataFrame(
        0,
        columns=FEATURE_COLUMNS,
        index=[0],
    )

    dt = input_data.date

    if "day" in X.columns:
        X.at[0, "day"] = dt.day

    if "month" in X.columns:
        X.at[0, "month"] = dt.month

    if "price" in X.columns:
        X.at[0, "price"] = input_data.price

    if "promo" in X.columns:
        X.at[0, "promo"] = input_data.promo

    cat_col = (
        f"category_{input_data.category}"
    )

    if cat_col in X.columns:
        X.at[0, cat_col] = 1

    reg_col = (
        f"region_{input_data.region}"
    )

    if reg_col in X.columns:
        X.at[0, reg_col] = 1

    return X


# =====================================================
# PREDICT
# =====================================================

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

        # ---------------------------------------------
        # Ensure model is available
        # ---------------------------------------------

        try:
            current_model = load_model()

        except Exception as exc:

            status_code = 503

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service is not ready",
            ) from exc

        if current_model is None:

            status_code = 503

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction model is not available",
            )

        # ---------------------------------------------
        # Build feature vector
        # ---------------------------------------------

        features = build_feature_vector(
            input_data
        )

        # ---------------------------------------------
        # Model prediction
        # ---------------------------------------------

        prediction_start = time.time()

        try:

            prediction = current_model.predict(
                features
            )[0]

        except Exception as exc:

            status_code = 500

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model prediction failed",
            ) from exc

        PREDICTION_LATENCY.observe(
            time.time()
            - prediction_start
        )

        # ---------------------------------------------
        # Prediction logging
        # ---------------------------------------------

        log_prediction(
            features=input_data.model_dump(
                mode="json"
            ),
            prediction=float(prediction),
        )

        # ---------------------------------------------
        # API response
        # ---------------------------------------------

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

        REQUEST_LATENCY.labels(
            endpoint="/predict",
        ).observe(
            time.time()
            - start_time
        )

        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict",
            http_status=status_code,
        ).inc()


# =====================================================
# METRICS
# =====================================================

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )