import pandas as pd
from datetime import datetime

from src.serving.app import build_feature_vector, PredictionInput
from src.serving.app import model


def test_feature_builder():
    sample = PredictionInput(
        date="2024-01-10",
        category="Electronics",
        region="North",
        price=1000.0,
        promo=1
    )

    df = build_feature_vector(sample)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert "price" in df.columns


def test_model_loaded():
    assert model is not None


def test_model_prediction():
    sample = PredictionInput(
        date="2024-01-10",
        category="Electronics",
        region="North",
        price=1000.0,
        promo=1
    )

    df = build_feature_vector(sample)
    prediction = model.predict(df)

    assert prediction is not None
    assert len(prediction) == 1