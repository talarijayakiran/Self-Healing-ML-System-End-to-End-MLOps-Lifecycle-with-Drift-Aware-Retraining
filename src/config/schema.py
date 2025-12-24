# src/config/schema.py

"""
Single Source of Truth
Each section belongs to ONE pipeline stage.
"""

# =========================
# RAW / INGESTION SCHEMA
# =========================
DATE_COLUMN = "date"

CATEGORICAL_COLUMNS = [
    "category",
    "region"
]

NUMERICAL_COLUMNS = [
    "price",
    "promo"
]

TARGET_COLUMN = "sales"

RAW_COLUMNS = (
    [DATE_COLUMN]
    + CATEGORICAL_COLUMNS
    + NUMERICAL_COLUMNS
    + [TARGET_COLUMN]
)

# =========================
# FEATURE ENGINEERING
# =========================
DERIVED_COLUMNS = [
    "day",
    "month"
]

# =========================
# MODEL FEATURES (FINAL)
# =========================
MODEL_FEATURES = [
    # numerical
    "price",
    "promo",
    "day",
    "month",

    # category
    "category_Electronics",
    "category_Furniture",
    "category_Grocery",

    # region
    "region_North",
    "region_South",
    "region_East",
    "region_West",
]