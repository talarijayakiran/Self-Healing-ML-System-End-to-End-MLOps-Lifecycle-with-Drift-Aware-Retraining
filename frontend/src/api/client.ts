import axios, {
  AxiosError,
  type AxiosInstance,
} from "axios";

import {
  createApiError,
  type ApiError,
} from "./errors";

const API_BASE_URL = import.meta.env.PROD
  ? "https://os6isp2hj9.execute-api.ap-south-2.amazonaws.com/api"
  : import.meta.env.VITE_API_BASE_URL ||
    "http://127.0.0.1:8000";

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 15000,
});

export function normalizeApiError(
  error: unknown,
): ApiError {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{
      detail?: string;
    }>;

    const status = axiosError.response?.status ?? null;

    const detail =
      axiosError.response?.data?.detail;

    if (typeof detail === "string") {
      return createApiError(
        detail,
        status,
        axiosError.code ?? null,
      );
    }

    if (axiosError.code === "ECONNABORTED") {
      return createApiError(
        "The forecasting service request timed out.",
        null,
        axiosError.code,
      );
    }

    if (!axiosError.response) {
      return createApiError(
        "Unable to connect to the forecasting service.",
        null,
        axiosError.code ?? null,
      );
    }

    return createApiError(
      "The forecasting service returned an unexpected error.",
      status,
      axiosError.code ?? null,
    );
  }

  if (error instanceof Error) {
    return createApiError(error.message);
  }

  return createApiError(
    "An unexpected application error occurred.",
  );
}