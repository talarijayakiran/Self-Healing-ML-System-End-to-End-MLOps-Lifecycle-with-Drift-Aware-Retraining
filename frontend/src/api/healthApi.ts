import { apiClient, normalizeApiError } from "./client";

import type {
  HealthResponse,
  ReadinessResponse,
} from "../types/prediction";

export async function checkHealth(): Promise<HealthResponse> {
  try {
    const response =
      await apiClient.get<HealthResponse>("/health");

    return response.data;
  } catch (error) {
    throw normalizeApiError(error);
  }
}

export async function checkReadiness(): Promise<ReadinessResponse> {
  try {
    const response =
      await apiClient.get<ReadinessResponse>("/ready");

    return response.data;
  } catch (error) {
    throw normalizeApiError(error);
  }
}