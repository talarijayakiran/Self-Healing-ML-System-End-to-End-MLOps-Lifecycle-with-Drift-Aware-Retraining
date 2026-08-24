export interface PredictionRequest {
  date: string;
  category: string;
  region: string;
  price: number;
  promo: number;
}

export interface PredictionResponse {
  predicted_sales: number;
  model_version: string;
  request_id: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version: string | null;
}