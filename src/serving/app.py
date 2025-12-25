import os
import time
from datetime import datetime

import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from src.monitoring.prediction_logger import log_prediction

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

model = None

# =====================================================
# LOAD MODEL (ONCE)
# =====================================================
def load_model():
    global model
    if model is not None:
        return model

    if TEST_MODE:
        print("TEST_MODE enabled — skipping model load")
        return None

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model directory not found: {MODEL_PATH}")

    model = mlflow.pyfunc.load_model(MODEL_PATH)
    print("Model loaded successfully")
    return model


# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(
    title="Self-Healing ML Inference API",
    version="1.0",
)

@app.on_event("startup")
def startup_event():
    load_model()


# =====================================================
# METRICS
# =====================================================
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"],
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
# SCHEMA
# =====================================================
class PredictionInput(BaseModel):
    date: str
    category: str
    region: str
    price: float
    promo: int

# =====================================================
# FEATURE BUILDER
# =====================================================
def build_feature_vector(input_data: PredictionInput) -> pd.DataFrame:
    X = pd.DataFrame(0, columns=FEATURE_COLUMNS, index=[0])

    dt = datetime.strptime(input_data.date, "%Y-%m-%d")
    if "day" in X.columns:
        X.at[0, "day"] = dt.day
    if "month" in X.columns:
        X.at[0, "month"] = dt.month

    X.at[0, "price"] = input_data.price
    X.at[0, "promo"] = input_data.promo

    cat = f"category_{input_data.category}"
    if cat in X.columns:
        X.at[0, cat] = 1

    reg = f"region_{input_data.region}"
    if reg in X.columns:
        X.at[0, reg] = 1

    return X

# =====================================================
# HEALTH
# =====================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }

# =====================================================
# PREDICT
# =====================================================
@app.post("/predict")
def predict(input_data: PredictionInput):
    start = time.time()
    status = 200

    try:
        if model is None:
            raise RuntimeError("Model not loaded")

        features = build_feature_vector(input_data)

        pred_start = time.time()
        prediction = model.predict(features)[0]
        PREDICTION_LATENCY.observe(time.time() - pred_start)

        log_prediction(
            features=input_data.dict(),
            prediction=float(prediction),
        )

        return {"predicted_sales": round(float(prediction), 2)}

    except Exception as e:
        status = 500
        raise e

    finally:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict",
            http_status=status,
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