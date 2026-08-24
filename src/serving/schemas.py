# src/serving/schemas.py

from datetime import date

from pydantic import BaseModel, Field, field_validator

from src.config.schema import (
    SUPPORTED_CATEGORIES,
    SUPPORTED_REGIONS,
)


class PredictionInput(BaseModel):
    """
    Input contract for the retail demand prediction API.
    """

    date: date

    category: str = Field(
        min_length=1,
        description="Retail product category",
    )

    region: str = Field(
        min_length=1,
        description="Retail sales region",
    )

    price: float = Field(
        gt=0,
        description="Product price; must be greater than zero",
    )

    promo: int = Field(
        ge=0,
        le=1,
        description="Promotion flag: 0 = no promotion, 1 = promotion",
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        if value not in SUPPORTED_CATEGORIES:
            raise ValueError(
                f"Unsupported category: {value}. "
                f"Supported categories: {SUPPORTED_CATEGORIES}"
            )

        return value

    @field_validator("region")
    @classmethod
    def validate_region(cls, value: str) -> str:
        if value not in SUPPORTED_REGIONS:
            raise ValueError(
                f"Unsupported region: {value}. "
                f"Supported regions: {SUPPORTED_REGIONS}"
            )

        return value


class PredictionResponse(BaseModel):
    """
    Standard response returned by the prediction endpoint.
    """

    predicted_sales: float
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    """
    Liveness/health response.
    """

    status: str
    model_loaded: bool
    model_version: str | None = None