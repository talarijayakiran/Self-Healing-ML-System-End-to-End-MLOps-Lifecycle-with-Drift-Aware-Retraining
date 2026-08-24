import { apiClient, normalizeApiError } from "./client";

import type {
  PredictionRequest,
  PredictionResponse,
} from "../types/prediction";

export async function predictDemand(
  payload: PredictionRequest,
): Promise<PredictionResponse> {
  try {
    const response =
      await apiClient.post<PredictionResponse>(
        "/predict",
        payload,
      );

    return response.data;
  } catch (error) {
    throw normalizeApiError(error);
  }
}