from pydantic import BaseModel

class PredictionRequest(BaseModel):
    store_id: int
    product_id: int
    category_encoded: int
    day: int
    month: int
    year: int

class PredictionResponse(BaseModel):
    predicted_sales: float
