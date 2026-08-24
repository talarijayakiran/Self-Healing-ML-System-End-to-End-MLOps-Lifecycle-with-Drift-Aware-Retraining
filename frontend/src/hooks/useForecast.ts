import { useState } from "react";

import { predictDemand } from "../api/predictionApi";
import type {
  PredictionRequest,
  PredictionResponse,
} from "../types/prediction";

interface UseForecastResult {
  prediction: PredictionResponse | null;
  loading: boolean;
  error: string;
  generateForecast: (
    payload: PredictionRequest,
  ) => Promise<void>;
  resetForecast: () => void;
}

export function useForecast(): UseForecastResult {
  const [prediction, setPrediction] =
    useState<PredictionResponse | null>(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  async function generateForecast(
    payload: PredictionRequest,
  ): Promise<void> {
    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const result = await predictDemand(payload);

      setPrediction(result);
    } catch (err: unknown) {
      console.error(
        "Forecast request failed:",
        err,
      );

      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError(
          "Unable to generate the forecast. Please try again.",
        );
      }
    } finally {
      setLoading(false);
    }
  }

  function resetForecast(): void {
    setPrediction(null);
    setError("");
  }

  return {
    prediction,
    loading,
    error,
    generateForecast,
    resetForecast,
  };
}