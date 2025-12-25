# src/validation/data_validation.py

import pandas as pd
from src.config.schema import RAW_COLUMNS

INPUT_PATH = "data/processed/raw_loaded.csv"
OUTPUT_PATH = "data/processed/validated_data.csv"


def validate(df: pd.DataFrame):
    # -------------------------
    # Column presence check
    # -------------------------
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # -------------------------
    # Null checks
    # -------------------------
    if df.isnull().any().any():
        raise ValueError("Null values found in raw data")

    # -------------------------
    # Basic type sanity
    # -------------------------
    if not pd.api.types.is_numeric_dtype(df["price"]):
        raise TypeError("price must be numeric")

    if not pd.api.types.is_numeric_dtype(df["promo"]):
        raise TypeError("promo must be numeric")

    if not pd.api.types.is_numeric_dtype(df["sales"]):
        raise TypeError("sales must be numeric")


def run_validation():
    df = pd.read_csv(INPUT_PATH)
    validate(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Data validation passed successfully")
    print(df.head())


if __name__ == "__main__":
    run_validation()