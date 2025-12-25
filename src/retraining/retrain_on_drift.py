import json
import subprocess
import sys
from pathlib import Path

DRIFT_REPORT_PATH = Path("data/monitoring/drift_report.json")


def retrain_model():
    """
    Entry point for automated retraining.
    Safe to call from:
    - Docker exec
    - Cron
    - CI/CD
    - API background task
    """

    if not DRIFT_REPORT_PATH.exists():
        print("❌ Drift report not found. Retraining aborted.")
        return False

    with open(DRIFT_REPORT_PATH, "r") as f:
        report = json.load(f)

    drift_found = False
    for feature, stats in report.items():
        if stats.get("drift_detected") is True:
            print(f"⚠ Drift detected in feature: {feature}")
            drift_found = True

    if not drift_found:
        print("✅ No drift detected. Retraining skipped.")
        return False

    print("🚀 Drift confirmed. Starting retraining pipeline...")

    subprocess.run(
        [sys.executable, "-m", "src.training.train_model"],
        check=True
    )

    print("✅ Retraining completed successfully.")
    return True