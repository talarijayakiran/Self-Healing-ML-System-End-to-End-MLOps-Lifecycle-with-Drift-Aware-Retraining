export interface ApiError {
  message: string;
  status: number | null;
  code: string | null;
}

export function createApiError(
  message: string,
  status: number | null = null,
  code: string | null = null,
): ApiError {
  return {
    message,
    status,
    code,
  };
}

export function isApiError(
  error: unknown,
): error is ApiError {
  return (
    typeof error === "object" &&
    error !== null &&
    "message" in error &&
    "status" in error &&
    "code" in error
  );
}