import { apiClient } from "./client";

import type {
  HealthResponse,
  ReadinessResponse,
} from "../types/prediction";

export async function checkHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>("/health");

  return response.data;
}

export async function checkReadiness(): Promise<ReadinessResponse> {
  const response = await apiClient.get<ReadinessResponse>("/ready");

  return response.data;
}