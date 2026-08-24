import { apiClient } from "./client";

import type {
  PredictionRequest,
  PredictionResponse,
} from "../types/prediction";

export async function predictDemand(
  payload: PredictionRequest,
): Promise<PredictionResponse> {
  const response = await apiClient.post<PredictionResponse>(
    "/predict",
    payload,
  );

  return response.data;
}