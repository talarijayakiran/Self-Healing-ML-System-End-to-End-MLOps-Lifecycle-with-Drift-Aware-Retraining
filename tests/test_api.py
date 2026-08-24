import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.serving.app import app


VALID_PAYLOAD = {
    "date": "2024-01-10",
    "category": "Electronics",
    "region": "North",
    "price": 1000.0,
    "promo": 1,
}


# =====================================================
# TEST CLIENT
# =====================================================

@pytest.fixture
def client():
    """
    Creates an isolated FastAPI test client.

    The real MLflow model is NOT loaded.
    A mocked model is injected into the application state.
    """

    mock_model = MagicMock()

    mock_model.predict.return_value = [123.456]

    with patch(
        "src.serving.app.load_model",
        return_value=mock_model,
    ):
        with patch(
            "src.serving.app.model",
            mock_model,
        ):
            with patch(
                "src.serving.app.model_version",
                "test-version",
            ):
                with TestClient(app) as test_client:
                    yield test_client


# =====================================================
# HEALTH
# =====================================================

def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "model_version" in data


# =====================================================
# READINESS
# =====================================================

def test_ready(client):

    response = client.get("/ready")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["model_version"] == "test-version"


# =====================================================
# VALID PREDICTION
# =====================================================

def test_valid_prediction(client):

    with patch(
        "src.serving.app.log_prediction"
    ):

        response = client.post(
            "/predict",
            json=VALID_PAYLOAD,
        )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_sales" in data
    assert data["predicted_sales"] == 123.46

    assert "model_version" in data
    assert data["model_version"] == "test-version"

    assert "request_id" in data
    assert data["request_id"]


# =====================================================
# INVALID DATE
# =====================================================

def test_invalid_date(client):

    payload = VALID_PAYLOAD.copy()

    payload["date"] = "not-a-date"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


# =====================================================
# INVALID CATEGORY
# =====================================================

def test_invalid_category(client):

    payload = VALID_PAYLOAD.copy()

    payload["category"] = "Unknown"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


# =====================================================
# INVALID REGION
# =====================================================

def test_invalid_region(client):

    payload = VALID_PAYLOAD.copy()

    payload["region"] = "Unknown"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


# =====================================================
# INVALID PRICE
# =====================================================

def test_invalid_price(client):

    payload = VALID_PAYLOAD.copy()

    payload["price"] = -100

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


# =====================================================
# INVALID PROMO
# =====================================================

def test_invalid_promo(client):

    payload = VALID_PAYLOAD.copy()

    payload["promo"] = 2

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


# =====================================================
# MODEL UNAVAILABLE
# =====================================================

def test_model_unavailable(client):

    with patch(
        "src.serving.app.load_model",
        return_value=None,
    ):

        response = client.post(
            "/predict",
            json=VALID_PAYLOAD,
        )

    assert response.status_code == 503

    data = response.json()

    assert data["detail"] == (
        "Prediction model is not available"
    )


# =====================================================
# MODEL LOADING FAILURE
# =====================================================

def test_model_loading_failure(client):

    with patch(
        "src.serving.app.load_model",
        side_effect=RuntimeError(
            "MLflow unavailable"
        ),
    ):

        response = client.post(
            "/predict",
            json=VALID_PAYLOAD,
        )

    assert response.status_code == 503

    data = response.json()

    assert data["detail"] == (
        "Prediction service is not ready"
    )


# =====================================================
# MODEL PREDICTION FAILURE
# =====================================================

def test_prediction_failure(client):

    failing_model = MagicMock()

    failing_model.predict.side_effect = (
        RuntimeError("Prediction failed")
    )

    with patch(
        "src.serving.app.load_model",
        return_value=failing_model,
    ):

        with patch(
            "src.serving.app.log_prediction"
        ):

            response = client.post(
                "/predict",
                json=VALID_PAYLOAD,
            )

    assert response.status_code == 500

    data = response.json()

    assert data["detail"] == (
        "Model prediction failed"
    )


# =====================================================
# METRICS
# =====================================================

def test_metrics(client):

    response = client.get("/metrics")

    assert response.status_code == 200

    assert (
        "http_requests_total"
        in response.text
    )

    assert (
        "prediction_latency_seconds"
        in response.text
    )