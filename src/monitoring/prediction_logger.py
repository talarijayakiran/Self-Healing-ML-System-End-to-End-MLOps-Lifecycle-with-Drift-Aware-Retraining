
import pandas as pd
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("data/monitoring/predictions.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_prediction(features: dict, prediction: float):
    record = {
        **features,
        "prediction": prediction,
        "timestamp": datetime.utcnow().isoformat()
    }

    df = pd.DataFrame([record])

    if LOG_PATH.exists():
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)
