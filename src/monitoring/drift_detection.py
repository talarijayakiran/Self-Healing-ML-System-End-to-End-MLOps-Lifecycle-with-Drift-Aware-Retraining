import json
from pathlib import Path
import pandas as pd

DRIFT_REPORT_PATH = Path("data/monitoring/drift_report.json")


def detect_drift(save: bool = True):
    # your existing drift logic
    drift_report = {
        "price": {
            "train_mean": 718.5,
            "live_mean": 1000.0,
            "drift_ratio": 0.392,
            "drift_detected": True,
        },
        "promo": {
            "train_mean": 0.5,
            "live_mean": 1.0,
            "drift_ratio": 1.0,
            "drift_detected": True,
        },
    }

    if save:
        DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DRIFT_REPORT_PATH, "w") as f:
            json.dump(drift_report, f, indent=2)

        print(f" Drift report saved to {DRIFT_REPORT_PATH}")


    return drift_report
