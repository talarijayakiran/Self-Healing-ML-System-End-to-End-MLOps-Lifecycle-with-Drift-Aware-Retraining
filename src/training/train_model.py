# src/training/train_model.py

import pandas as pd
import mlflow
import mlflow.sklearn
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.config.schema import TARGET_COLUMN

DATA_PATH = "data/processed/processed_train.csv"
EXPERIMENT_NAME = "Retail Demand Forecasting"
MODEL_NAME = "retail_demand_forecaster"


def train_and_log():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # SAFETY: ensure enough test samples
    test_size = 0.4 if len(df) < 10 else 0.25

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds) ** 0.5

        # 🔒 SAFE METRIC LOGGING
        mlflow.log_metric("rmse", rmse, step=0)
        time.sleep(1)  # prevent timestamp collision

        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME
        )

        print("✅ MODEL TRAINED & REGISTERED")
        print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    train_and_log()