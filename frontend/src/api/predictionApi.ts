import axios from "axios";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 15000,
});

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

export async function predictDemand(
  payload: PredictionRequest,
): Promise<PredictionResponse> {
  const response = await apiClient.post<PredictionResponse>(
    "/predict",
    payload,
  );

  return response.data;
}

export async function checkHealth() {
  const response = await apiClient.get("/health");

  return response.data;
}

export async function checkReadiness() {
  const response = await apiClient.get("/ready");

  return response.data;
}