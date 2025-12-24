import pandas as pd
import json
from pathlib import Path

TRAIN_DATA_PATH = "data/processed/processed_train.csv"
PREDICTION_LOG_PATH = "data/monitoring/predictions.csv"
REPORT_PATH = "data/monitoring/drift_report.json"

NUMERICAL_COLUMNS = ["price", "promo"]


def detect_drift(threshold: float = 0.2):
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    pred_df = pd.read_csv(PREDICTION_LOG_PATH)

    report = {}

    for col in NUMERICAL_COLUMNS:
        train_mean = train_df[col].mean()
        live_mean = pred_df[col].mean()

        drift_ratio = abs(train_mean - live_mean) / train_mean

        report[col] = {
            "train_mean": round(train_mean, 3),
            "live_mean": round(live_mean, 3),
            "drift_ratio": round(drift_ratio, 3),
            "drift_detected": bool(drift_ratio > threshold)
        }

    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    return report


if __name__ == "__main__":
    output = detect_drift()
    print(json.dumps(output, indent=4))
