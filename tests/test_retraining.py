import json
import subprocess

import pytest

from src.retraining import retrain_on_drift


def test_retraining_stops_when_drift_report_is_missing(
    tmp_path,
    monkeypatch,
):
    report_path = (
        tmp_path /
        "drift_report.json"
    )

    monkeypatch.setattr(
        retrain_on_drift,
        "DRIFT_REPORT_PATH",
        report_path,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Drift report not found",
    ):
        retrain_on_drift.retrain_model()


def test_retraining_skips_when_no_drift_is_detected(
    tmp_path,
    monkeypatch,
):
    report_path = (
        tmp_path /
        "drift_report.json"
    )

    report = {
        "_summary": {
            "drift_detected": False,
            "monitored_features": 2,
        }
    }

    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        retrain_on_drift,
        "DRIFT_REPORT_PATH",
        report_path,
    )

    training_mock = monkeypatch.setattr(
        retrain_on_drift,
        "_run_training",
        lambda: pytest.fail(
            "Training must not run when no drift is detected."
        ),
    )

    result = (
        retrain_on_drift.retrain_model()
    )

    assert result.drift_detected is False
    assert result.retraining_triggered is False
    assert result.training_succeeded is False


def test_retraining_triggers_training_when_drift_is_detected(
    tmp_path,
    monkeypatch,
):
    report_path = (
        tmp_path /
        "drift_report.json"
    )

    report = {
        "price": {
            "drift_detected": True,
        },
        "promo": {
            "drift_detected": False,
        },
        "_summary": {
            "drift_detected": True,
            "monitored_features": 2,
        },
    }

    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        retrain_on_drift,
        "DRIFT_REPORT_PATH",
        report_path,
    )

    calls = []

    def fake_training():
        calls.append(True)

    monkeypatch.setattr(
        retrain_on_drift,
        "_run_training",
        fake_training,
    )

    result = (
        retrain_on_drift.retrain_model()
    )

    assert result.drift_detected is True
    assert result.retraining_triggered is True
    assert result.training_succeeded is True
    assert calls == [True]


def test_retraining_propagates_training_failure(
    tmp_path,
    monkeypatch,
):
    report_path = (
        tmp_path /
        "drift_report.json"
    )

    report = {
        "_summary": {
            "drift_detected": True,
            "monitored_features": 2,
        }
    }

    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        retrain_on_drift,
        "DRIFT_REPORT_PATH",
        report_path,
    )

    def failing_training():
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=[
                "python",
                "-m",
                "src.training.train_model",
            ],
        )

    monkeypatch.setattr(
        retrain_on_drift,
        "_run_training",
        failing_training,
    )

    with pytest.raises(
        subprocess.CalledProcessError,
    ):
        retrain_on_drift.retrain_model()


def test_retraining_rejects_missing_summary(
    tmp_path,
    monkeypatch,
):
    report_path = (
        tmp_path /
        "drift_report.json"
    )

    report = {
        "price": {
            "drift_detected": True,
        }
    }

    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        retrain_on_drift,
        "DRIFT_REPORT_PATH",
        report_path,
    )

    with pytest.raises(
        ValueError,
        match="missing a valid '_summary'",
    ):
        retrain_on_drift.retrain_model()


def test_retraining_rejects_non_boolean_drift_decision(
    tmp_path,
    monkeypatch,
):
    report_path = (
        tmp_path /
        "drift_report.json"
    )

    report = {
        "_summary": {
            "drift_detected": "true",
            "monitored_features": 2,
        }
    }

    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        retrain_on_drift,
        "DRIFT_REPORT_PATH",
        report_path,
    )

    with pytest.raises(
        ValueError,
        match="'_summary.drift_detected' must be boolean",
    ):
        retrain_on_drift.retrain_model()


def test_retraining_rejects_invalid_json(
    tmp_path,
    monkeypatch,
):
    report_path = (
        tmp_path /
        "drift_report.json"
    )

    report_path.write_text(
        "{invalid-json",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        retrain_on_drift,
        "DRIFT_REPORT_PATH",
        report_path,
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        retrain_on_drift.retrain_model()