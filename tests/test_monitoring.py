

import pandas as pd
import pytest

from src.monitoring import drift_detection
from src.monitoring.drift_detection import (
    _calculate_drift_ratio,
    detect_drift,
)
from src.monitoring.prediction_logger import (
    PREDICTION_COLUMNS,
    log_prediction,
)


# ============================================================
# DRIFT RATIO TESTS
# ============================================================

def test_drift_ratio_is_zero_when_means_match():
    result = _calculate_drift_ratio(
        100.0,
        100.0,
    )

    assert result == 0.0


def test_drift_ratio_is_calculated_correctly():
    result = _calculate_drift_ratio(
        100.0,
        120.0,
    )

    assert result == pytest.approx(0.20)


def test_zero_reference_mean_is_handled():
    result = _calculate_drift_ratio(
        0.0,
        10.0,
    )

    assert result == float("inf")


def test_drift_at_threshold_is_detected():
    result = drift_detection._calculate_feature_drift(
        reference=pd.Series(
            [100.0, 100.0]
        ),
        live=pd.Series(
            [120.0, 120.0]
        ),
    )

    assert result.drift_ratio == pytest.approx(
        0.20
    )

    assert result.drift_detected is True


# ============================================================
# PREDICTION LOGGER TESTS
# ============================================================

def test_prediction_logger_creates_contract(
    tmp_path,
    monkeypatch,
):
    log_path = (
        tmp_path /
        "predictions.csv"
    )

    monkeypatch.setattr(
        "src.monitoring.prediction_logger.LOG_PATH",
        log_path,
    )

    log_prediction(
        request_id="request-123",
        model_version="7",
        date="2024-01-10",
        category="Electronics",
        region="North",
        price=1000.0,
        promo=1,
        prediction=125.5,
    )

    df = pd.read_csv(
        log_path
    )

    assert list(
        df.columns
    ) == PREDICTION_COLUMNS

    assert len(df) == 1

    assert (
        df.loc[0, "request_id"]
        == "request-123"
    )

    assert (
        str(df.loc[0, "model_version"])
        == "7"
    )

    assert (
        df.loc[0, "price"]
        == 1000.0
    )

    assert (
        df.loc[0, "promo"]
        == 1
    )

    assert (
        df.loc[0, "prediction"]
        == 125.5
    )


def test_prediction_logger_rejects_invalid_price(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.monitoring.prediction_logger.LOG_PATH",
        tmp_path / "predictions.csv",
    )

    with pytest.raises(
        ValueError,
        match="price must be greater than 0",
    ):
        log_prediction(
            request_id="request-123",
            model_version="7",
            date="2024-01-10",
            category="Electronics",
            region="North",
            price=0,
            promo=1,
            prediction=125.5,
        )


def test_prediction_logger_rejects_invalid_promo(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.monitoring.prediction_logger.LOG_PATH",
        tmp_path / "predictions.csv",
    )

    with pytest.raises(
        ValueError,
        match="promo must be either 0 or 1",
    ):
        log_prediction(
            request_id="request-123",
            model_version="7",
            date="2024-01-10",
            category="Electronics",
            region="North",
            price=1000,
            promo=2,
            prediction=125.5,
        )


def test_prediction_logger_rejects_non_finite_prediction(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.monitoring.prediction_logger.LOG_PATH",
        tmp_path / "predictions.csv",
    )

    with pytest.raises(
        ValueError,
        match="prediction must be finite",
    ):
        log_prediction(
            request_id="request-123",
            model_version="7",
            date="2024-01-10",
            category="Electronics",
            region="North",
            price=1000.0,
            promo=1,
            prediction=float("nan"),
        )


def test_prediction_logger_rejects_non_finite_price(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.monitoring.prediction_logger.LOG_PATH",
        tmp_path / "predictions.csv",
    )

    with pytest.raises(
        ValueError,
        match="price must be finite",
    ):
        log_prediction(
            request_id="request-123",
            model_version="7",
            date="2024-01-10",
            category="Electronics",
            region="North",
            price=float("inf"),
            promo=1,
            prediction=125.5,
        )


def test_prediction_logger_rejects_missing_category(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.monitoring.prediction_logger.LOG_PATH",
        tmp_path / "predictions.csv",
    )

    with pytest.raises(
        ValueError,
        match="category must be provided",
    ):
        log_prediction(
            request_id="request-123",
            model_version="7",
            date="2024-01-10",
            category="",
            region="North",
            price=1000.0,
            promo=1,
            prediction=125.5,
        )


# ============================================================
# DRIFT DETECTION TESTS
# ============================================================

def test_detect_drift_uses_reference_and_live_data(
    tmp_path,
    monkeypatch,
):
    reference_path = (
        tmp_path /
        "reference.csv"
    )

    live_path = (
        tmp_path /
        "live.csv"
    )

    report_path = (
        tmp_path /
        "drift_report.json"
    )

    reference = pd.DataFrame(
        {
            "price": [100, 100, 100],
            "promo": [0, 0, 0],
        }
    )

    live = pd.DataFrame(
        {
            "price": [150, 150, 150],
            "promo": [1, 1, 1],
        }
    )

    reference.to_csv(
        reference_path,
        index=False,
    )

    live.to_csv(
        live_path,
        index=False,
    )

    monkeypatch.setattr(
        drift_detection,
        "REFERENCE_DATA_PATH",
        reference_path,
    )

    monkeypatch.setattr(
        drift_detection,
        "LIVE_DATA_PATH",
        live_path,
    )

    monkeypatch.setattr(
        drift_detection,
        "DRIFT_REPORT_PATH",
        report_path,
    )

    report = detect_drift(
        save=True,
    )

    assert (
        report["price"]["drift_detected"]
        is True
    )

    assert (
        report["promo"]["drift_detected"]
        is True
    )

    assert (
        report["_summary"]["drift_detected"]
        is True
    )

    assert report_path.exists()


def test_detect_drift_rejects_non_numeric_feature(
    tmp_path,
    monkeypatch,
):
    reference_path = (
        tmp_path /
        "reference.csv"
    )

    live_path = (
        tmp_path /
        "live.csv"
    )

    reference = pd.DataFrame(
        {
            "price": [
                100,
                "not-a-number",
                100,
            ],
            "promo": [0, 0, 0],
        }
    )

    live = pd.DataFrame(
        {
            "price": [
                150,
                150,
                150,
            ],
            "promo": [1, 1, 1],
        }
    )

    reference.to_csv(
        reference_path,
        index=False,
    )

    live.to_csv(
        live_path,
        index=False,
    )

    monkeypatch.setattr(
        drift_detection,
        "REFERENCE_DATA_PATH",
        reference_path,
    )

    monkeypatch.setattr(
        drift_detection,
        "LIVE_DATA_PATH",
        live_path,
    )

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        detect_drift(
            save=False,
        )


def test_detect_drift_rejects_null_feature(
    tmp_path,
    monkeypatch,
):
    reference_path = (
        tmp_path /
        "reference.csv"
    )

    live_path = (
        tmp_path /
        "live.csv"
    )

    reference = pd.DataFrame(
        {
            "price": [100, 100, 100],
            "promo": [0, 0, 0],
        }
    )

    live = pd.DataFrame(
        {
            "price": [
                150,
                None,
                150,
            ],
            "promo": [1, 1, 1],
        }
    )

    reference.to_csv(
        reference_path,
        index=False,
    )

    live.to_csv(
        live_path,
        index=False,
    )

    monkeypatch.setattr(
        drift_detection,
        "REFERENCE_DATA_PATH",
        reference_path,
    )

    monkeypatch.setattr(
        drift_detection,
        "LIVE_DATA_PATH",
        live_path,
    )

    with pytest.raises(
        ValueError,
        match="contains null values",
    ):
        detect_drift(
            save=False,
        )


def test_detect_drift_fails_when_live_data_is_missing(
    tmp_path,
    monkeypatch,
):
    reference_path = (
        tmp_path /
        "reference.csv"
    )

    reference = pd.DataFrame(
        {
            "price": [100],
            "promo": [0],
        }
    )

    reference.to_csv(
        reference_path,
        index=False,
    )

    monkeypatch.setattr(
        drift_detection,
        "REFERENCE_DATA_PATH",
        reference_path,
    )

    monkeypatch.setattr(
        drift_detection,
        "LIVE_DATA_PATH",
        tmp_path / "missing.csv",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Live prediction data not found",
    ):
        detect_drift(
            save=False,
        )