import time
import os
from datetime import datetime
from pathlib import Path

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
# MODEL LOADING (FILE SYSTEM ONLY – CONTAINER SAFE)
# =====================================================
MODEL_PATH = os.getenv("MODEL_PATH", "exported_model")
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

model = None

def load_model():
    global model
    if model is not None:
        return model

    if TEST_MODE:
        return None  # Skip model loading in CI/tests

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model directory not found at {MODEL_PATH}")

    model = mlflow.pyfunc.load_model(MODEL_PATH)
    return model
# =====================================================
# FEATURE TEMPLATE (ORDER LOCK)
# =====================================================
FEATURE_TEMPLATE_PATH = "data/processed/processed_inference.csv"
FEATURE_COLUMNS = pd.read_csv(FEATURE_TEMPLATE_PATH).columns.tolist()

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(
    title="Retail Demand Forecasting API",
    version="1.0",
)

# =====================================================
# PROMETHEUS METRICS (DEFINE ONCE)
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
# INPUT SCHEMA
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

    cat_col = f"category_{input_data.category}"
    if cat_col in X.columns:
        X.at[0, cat_col] = 1

    reg_col = f"region_{input_data.region}"
    if reg_col in X.columns:
        X.at[0, reg_col] = 1

    return X

# =====================================================
# PREDICT ENDPOINT
# =====================================================
@app.post("/predict")
def predict(input_data: PredictionInput):
    start_time = time.time()
    status_code = 200

    try:
        final_df = build_feature_vector(input_data)
        prediction = model.predict(final_df)[0]

        log_prediction(
            features=input_data.dict(),
            prediction=float(prediction)
        )

        return {
            "predicted_sales": round(float(prediction), 2)
        }

    except Exception:
        status_code = 500
        raise

    finally:
        latency = time.time() - start_time

        REQUEST_LATENCY.labels(
            endpoint="/predict"
        ).observe(latency)

        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict",
            http_status=status_code
        ).inc()# =====================================================
# PROMETHEUS METRICS ENDPOINT
# =====================================================
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )