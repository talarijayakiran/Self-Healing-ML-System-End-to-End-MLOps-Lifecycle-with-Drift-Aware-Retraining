import json
import subprocess
import sys
from pathlib import Path

DRIFT_REPORT_PATH = Path("data/monitoring/drift_report.json")

def should_retrain() -> bool:
    if not DRIFT_REPORT_PATH.exists():
        print("❌ Drift report not found. Skipping retraining.")
        return False

    with open(DRIFT_REPORT_PATH, "r") as f:
        report = json.load(f)

    for feature, stats in report.items():
        if stats["drift_detected"] is True:
            print(f"⚠ Drift detected in feature: {feature}")
            return True

    print("✅ No drift detected. Retraining not required.")
    return False


def trigger_retraining():
    print("🚀 Triggering model retraining...")
    subprocess.run(
        [sys.executable, "-m", "src.training.train_model"],
        check=True
    )
    print("✅ Retraining completed and model registered.")


if __name__ == "__main__":
    if should_retrain():
        trigger_retraining()
    else:
        print("🛑 Retraining skipped.")