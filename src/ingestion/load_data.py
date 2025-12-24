# src/ingestion/load_data.py

import pandas as pd
from src.config.schema import (
    TARGET_COLUMN,
    DATE_COLUMN,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS
)

RAW_DATA_PATH = "data/raw/retail_sales.csv"
OUTPUT_PATH = "data/processed/raw_loaded.csv"


def load_raw_data():
    df = pd.read_csv(RAW_DATA_PATH)

    required_columns = (
        [DATE_COLUMN]
        + CATEGORICAL_COLUMNS
        + NUMERICAL_COLUMNS
        + [TARGET_COLUMN]
    )

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw data: {missing}")

    df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Raw data loaded successfully")
    print(df.head())

    return df


if __name__ == "__main__":
    load_raw_data()