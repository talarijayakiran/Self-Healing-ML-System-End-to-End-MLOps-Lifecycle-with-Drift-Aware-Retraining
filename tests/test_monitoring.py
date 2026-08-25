import json
from dataclasses import replace

import pandas as pd
import pytest

from src.config import settings as settings_module
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
# TEST RUNTIME CONFIGURATION HELPER
# ============================================================


def _configure_drift_settings(
    monkeypatch,
    *,
    reference_data_path,
    prediction_log_path,
    drift_report_path,
    drift_threshold=None,
    observation_window_size=None,
    min_observations=None,
):
    """
    Replace the application-wide Settings object with a
    test-specific immutable configuration.

    Settings is a frozen dataclass, so individual fields cannot
    be mutated directly. dataclasses.replace() creates a new
    validated configuration object while preserving all
    unrelated runtime settings.
    """

    current = settings_module.settings

    overrides = {
        "reference_data_path": reference_data_path,
        "prediction_log_path": prediction_log_path,
        "drift_report_path": drift_report_path,
    }

    if drift_threshold is not None:
        overrides["drift_threshold"] = drift_threshold

    if observation_window_size is not None:
        overrides[
            "observation_window_size"
        ] = observation_window_size

    if min_observations is not None:
        overrides[
            "min_observations"
        ] = min_observations

    monkeypatch.setattr(
        settings_module,
        "settings",
        replace(
            current,
            **overrides,
        ),
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

    assert result == pytest.approx(
        0.20
    )


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

    assert result.threshold == pytest.approx(
        settings_module.settings.drift_threshold
    )

    assert result.drift_detected is True


# ============================================================
# RUNTIME DRIFT THRESHOLD TESTS
# ============================================================


def test_drift_detection_uses_runtime_drift_threshold(
    monkeypatch,
):
    current = settings_module.settings

    monkeypatch.setattr(
        settings_module,
        "settings",
        replace(
            current,
            drift_threshold=0.30,
        ),
    )

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

    assert result.threshold == pytest.approx(
        0.30
    )

    assert result.drift_detected is False


def test_drift_detection_rejects_using_runtime_threshold(
    monkeypatch,
):
    current = settings_module.settings

    monkeypatch.setattr(
        settings_module,
        "settings",
        replace(
            current,
            drift_threshold=0.10,
        ),
    )

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

    assert result.threshold == pytest.approx(
        0.10
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


def test_detect_drift_uses_runtime_paths(
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
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=10,
                freq="h",
            ),
            "model_version": ["7"] * 10,
            "price": [150] * 10,
            "promo": [1] * 10,
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

    _configure_drift_settings(
        monkeypatch,
        reference_data_path=reference_path,
        prediction_log_path=live_path,
        drift_report_path=report_path,
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

    assert (
        report["_summary"]["observation_count"]
        == 10
    )

    assert (
        report["_summary"]["observation_window_size"]
        == settings_module.settings.observation_window_size
    )

    assert (
        report["_summary"]["minimum_observations"]
        == settings_module.settings.min_observations
    )

    assert (
        report["_summary"]["window_complete"]
        is False
    )

    assert (
        report["_summary"]["model_versions"]
        == ["7"]
    )

    assert report_path.exists()


def test_detect_drift_report_contains_observation_metadata(
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

    timestamps = pd.date_range(
        "2024-01-01",
        periods=10,
        freq="h",
    )

    live = pd.DataFrame(
        {
            "timestamp": timestamps,
            "model_version": ["7"] * 10,
            "price": [150] * 10,
            "promo": [1] * 10,
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

    _configure_drift_settings(
        monkeypatch,
        reference_data_path=reference_path,
        prediction_log_path=live_path,
        drift_report_path=report_path,
    )

    detect_drift(
        save=True,
    )

    with report_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        persisted_report = json.load(file)

    summary = persisted_report["_summary"]

    assert (
        summary["observation_count"]
        == 10
    )

    assert (
        summary["observation_window_size"]
        == settings_module.settings.observation_window_size
    )

    assert (
        summary["minimum_observations"]
        == settings_module.settings.min_observations
    )

    assert (
        summary["window_complete"]
        is False
    )

    assert (
        summary["model_versions"]
        == ["7"]
    )

    assert (
        summary[
            "oldest_observation_timestamp"
        ]
        == "2024-01-01T00:00:00+00:00"
    )

    assert (
        summary[
            "newest_observation_timestamp"
        ]
        == "2024-01-01T09:00:00+00:00"
    )


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
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=10,
                freq="h",
            ),
            "price": [150] * 10,
            "promo": [1] * 10,
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

    _configure_drift_settings(
        monkeypatch,
        reference_data_path=reference_path,
        prediction_log_path=live_path,
        drift_report_path=(
            tmp_path /
            "drift_report.json"
        ),
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
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=10,
                freq="h",
            ),
            "price": [
                150,
                None,
                150,
                150,
                150,
                150,
                150,
                150,
                150,
                150,
            ],
            "promo": [1] * 10,
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

    _configure_drift_settings(
        monkeypatch,
        reference_data_path=reference_path,
        prediction_log_path=live_path,
        drift_report_path=(
            tmp_path /
            "drift_report.json"
        ),
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

    _configure_drift_settings(
        monkeypatch,
        reference_data_path=reference_path,
        prediction_log_path=(
            tmp_path /
            "missing.csv"
        ),
        drift_report_path=(
            tmp_path /
            "drift_report.json"
        ),
    )

    with pytest.raises(
        FileNotFoundError,
        match="Live prediction data not found",
    ):
        detect_drift(
            save=False,
        )


# ============================================================
# OBSERVATION WINDOW TESTS
# ============================================================


def test_observation_window_selects_latest_records():
    timestamps = pd.date_range(
        "2024-01-01",
        periods=60,
        freq="h",
    )

    live = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": range(60),
            "promo": [0] * 60,
        }
    )

    window = drift_detection._select_observation_window(
        live
    )

    assert len(window) == (
        settings_module.settings.observation_window_size
    )

    expected_start = (
        timestamps[
            -settings_module.settings.observation_window_size
        ]
    )

    expected_end = timestamps[-1]

    assert (
        window["timestamp"].min()
        == pd.Timestamp(
            expected_start,
            tz="UTC",
        )
    )

    assert (
        window["timestamp"].max()
        == pd.Timestamp(
            expected_end,
            tz="UTC",
        )
    )


def test_observation_window_orders_by_timestamp():
    timestamps = pd.date_range(
        "2024-01-01",
        periods=10,
        freq="h",
    )

    live = pd.DataFrame(
        {
            "timestamp": list(
                reversed(timestamps)
            ),
            "price": [100] * 10,
            "promo": [0] * 10,
        }
    )

    window = drift_detection._select_observation_window(
        live
    )

    assert (
        window["timestamp"]
        .is_monotonic_increasing
    )


def test_observation_window_rejects_insufficient_data():
    min_observations = (
        settings_module.settings.min_observations
    )

    timestamps = pd.date_range(
        "2024-01-01",
        periods=min_observations - 1,
        freq="h",
    )

    live = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": [100] * (
                min_observations - 1
            ),
            "promo": [0] * (
                min_observations - 1
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="Insufficient live observations",
    ):
        drift_detection._select_observation_window(
            live
        )


def test_observation_window_rejects_invalid_timestamp():
    live = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "invalid-timestamp",
            ],
            "price": [100, 100],
            "promo": [0, 0],
        }
    )

    with pytest.raises(
        ValueError,
        match="invalid.*timestamp",
    ):
        drift_detection._select_observation_window(
            live
        )


# ============================================================
# RUNTIME OBSERVATION WINDOW CONFIGURATION TESTS
# ============================================================


def test_observation_window_uses_runtime_configuration(
    monkeypatch,
):
    current = settings_module.settings

    monkeypatch.setattr(
        settings_module,
        "settings",
        replace(
            current,
            observation_window_size=20,
            min_observations=5,
        ),
    )

    timestamps = pd.date_range(
        "2024-01-01",
        periods=30,
        freq="h",
    )

    live = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": range(30),
            "promo": [0] * 30,
        }
    )

    window = (
        drift_detection
        ._select_observation_window(live)
    )

    assert len(window) == 20

    assert (
        window["timestamp"].min()
        == pd.Timestamp(
            "2024-01-01 10:00:00",
            tz="UTC",
        )
    )

    assert (
        window["timestamp"].max()
        == pd.Timestamp(
            "2024-01-02 05:00:00",
            tz="UTC",
        )
    )


def test_observation_window_uses_runtime_minimum_observations(
    monkeypatch,
):
    current = settings_module.settings

    monkeypatch.setattr(
        settings_module,
        "settings",
        replace(
            current,
            observation_window_size=20,
            min_observations=15,
        ),
    )

    timestamps = pd.date_range(
        "2024-01-01",
        periods=10,
        freq="h",
    )

    live = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": [100] * 10,
            "promo": [0] * 10,
        }
    )

    with pytest.raises(
        ValueError,
        match="Required at least 15",
    ):
        drift_detection._select_observation_window(
            live
        )


# ============================================================
# OBSERVATION WINDOW METADATA TESTS
# ============================================================


def test_observation_window_metadata_reports_partial_window():
    timestamps = pd.date_range(
        "2024-01-01",
        periods=10,
        freq="h",
    )

    live = pd.DataFrame(
        {
            "timestamp": timestamps,
            "model_version": ["7"] * 10,
            "price": [100] * 10,
            "promo": [0] * 10,
        }
    )

    window = drift_detection._select_observation_window(
        live
    )

    metadata = (
        drift_detection._build_observation_window_metadata(
            window
        )
    )

    assert (
        metadata["observation_count"]
        == 10
    )

    assert (
        metadata["observation_window_size"]
        == settings_module.settings.observation_window_size
    )

    assert (
        metadata["minimum_observations"]
        == settings_module.settings.min_observations
    )

    assert (
        metadata["window_complete"]
        is False
    )

    assert (
        metadata["model_versions"]
        == ["7"]
    )

    assert (
        metadata[
            "oldest_observation_timestamp"
        ]
        == "2024-01-01T00:00:00+00:00"
    )

    assert (
        metadata[
            "newest_observation_timestamp"
        ]
        == "2024-01-01T09:00:00+00:00"
    )


def test_observation_window_metadata_reports_complete_window():
    observation_window_size = (
        settings_module.settings.observation_window_size
    )

    timestamps = pd.date_range(
        "2024-01-01",
        periods=observation_window_size,
        freq="h",
    )

    live = pd.DataFrame(
        {
            "timestamp": timestamps,
            "model_version": ["7"] * observation_window_size,
            "price": [100] * observation_window_size,
            "promo": [0] * observation_window_size,
        }
    )

    window = drift_detection._select_observation_window(
        live
    )

    metadata = (
        drift_detection._build_observation_window_metadata(
            window
        )
    )

    assert (
        metadata["observation_count"]
        == observation_window_size
    )

    assert (
        metadata["observation_window_size"]
        == observation_window_size
    )

    assert (
        metadata["window_complete"]
        is True
    )

    assert (
        metadata["model_versions"]
        == ["7"]
    )


def test_observation_window_metadata_tracks_multiple_model_versions():
    observation_window_size = (
        settings_module.settings.observation_window_size
    )

    timestamps = pd.date_range(
        "2024-01-01",
        periods=observation_window_size,
        freq="h",
    )

    first_version_count = (
        observation_window_size // 2
    )

    second_version_count = (
        observation_window_size
        - first_version_count
    )

    live = pd.DataFrame(
        {
            "timestamp": timestamps,
            "model_version": (
                ["7"] * first_version_count
                + ["8"] * second_version_count
            ),
            "price": [100] * observation_window_size,
            "promo": [0] * observation_window_size,
        }
    )

    window = drift_detection._select_observation_window(
        live
    )

    metadata = (
        drift_detection._build_observation_window_metadata(
            window
        )
    )

    assert (
        metadata["model_versions"]
        == ["7", "8"]
    )


def test_observation_window_metadata_handles_missing_model_version():
    timestamps = pd.date_range(
        "2024-01-01",
        periods=10,
        freq="h",
    )

    live = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": [100] * 10,
            "promo": [0] * 10,
        }
    )

    window = drift_detection._select_observation_window(
        live
    )

    metadata = (
        drift_detection._build_observation_window_metadata(
            window
        )
    )

    assert (
        metadata["model_versions"]
        == []
    )


def test_observation_window_metadata_rejects_empty_window():
    live = pd.DataFrame(
        {
            "timestamp": pd.Series(
                dtype="datetime64[ns]"
            ),
            "price": pd.Series(
                dtype="float64"
            ),
            "promo": pd.Series(
                dtype="int64"
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="Observation window cannot be empty",
    ):
        drift_detection._build_observation_window_metadata(
            live
        )


# ============================================================
# DRIFT DETECTION WITH RECENT OBSERVATION WINDOW
# ============================================================


def test_detect_drift_uses_only_recent_observation_window(
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

    timestamps = pd.date_range(
        "2024-01-01",
        periods=60,
        freq="h",
    )

    live = pd.DataFrame(
        {
            "timestamp": timestamps,
            "model_version": (
                ["7"] * 10
                + ["8"] * 50
            ),
            "price": (
                [100] * 10
                + [120] * 50
            ),
            "promo": [0] * 60,
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

    _configure_drift_settings(
        monkeypatch,
        reference_data_path=reference_path,
        prediction_log_path=live_path,
        drift_report_path=(
            tmp_path /
            "drift_report.json"
        ),
    )

    report = drift_detection.detect_drift(
        save=False,
    )

    assert (
        report["price"]["live_mean"]
        == pytest.approx(120.0)
    )

    assert (
        report["price"]["drift_detected"]
        is True
    )

    assert (
        report["_summary"]["observation_count"]
        == settings_module.settings.observation_window_size
    )

    assert (
        report["_summary"]["observation_window_size"]
        == settings_module.settings.observation_window_size
    )

    assert (
        report["_summary"]["minimum_observations"]
        == settings_module.settings.min_observations
    )

    assert (
        report["_summary"]["window_complete"]
        is True
    )

    assert (
        report["_summary"]["model_versions"]
        == ["8"]
    )

    expected_start = (
        timestamps[
            -settings_module.settings.observation_window_size
        ]
    )

    expected_end = timestamps[-1]

    assert (
        report["_summary"][
            "oldest_observation_timestamp"
        ]
        == pd.Timestamp(
            expected_start,
            tz="UTC",
        ).isoformat()
    )

    assert (
        report["_summary"][
            "newest_observation_timestamp"
        ]
        == pd.Timestamp(
            expected_end,
            tz="UTC",
        ).isoformat()
    )