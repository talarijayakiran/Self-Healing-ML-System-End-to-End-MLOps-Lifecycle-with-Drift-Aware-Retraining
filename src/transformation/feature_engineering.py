# src/transformation/feature_engineering.py

import pandas as pd
from src.config.schema import TARGET_COLUMN

INPUT_PATH = "data/processed/validated_data.csv"
TRAIN_PATH = "data/processed/processed_train.csv"
INFER_PATH = "data/processed/processed_inference.csv"


def run_feature_engineering():
    df = pd.read_csv(INPUT_PATH)

    # -------------------------
    # Date features (created HERE, nowhere else)
    # -------------------------
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month

    # Drop raw date
    df = df.drop(columns=["date"])

    # -------------------------
    # One-hot encoding
    # -------------------------
    df = pd.get_dummies(df, columns=["category", "region"])

    # -------------------------
    # Split target
    # -------------------------
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # -------------------------
    # Save outputs
    # -------------------------
    train_df = pd.concat([X, y], axis=1)

    train_df.to_csv(TRAIN_PATH, index=False)
    X.to_csv(INFER_PATH, index=False)

    print("✅ Feature engineering completed")
    print("Train shape:", train_df.shape)
    print("Inference shape:", X.shape)
    print("Columns:", list(X.columns))


if __name__ == "__main__":
    run_feature_engineering()