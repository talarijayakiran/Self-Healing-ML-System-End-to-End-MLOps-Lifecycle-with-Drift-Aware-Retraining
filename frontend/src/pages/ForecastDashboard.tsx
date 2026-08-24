import { ArrowRight } from "lucide-react";
import { useState } from "react";

import ForecastForm from "../components/forecast/ForecastForm";
import ForecastHeader from "../components/forecast/ForecastHeader";
import ForecastResult from "../components/forecast/ForecastResult";

import {
  DEFAULT_FORECAST_CATEGORY,
  DEFAULT_FORECAST_DATE,
  DEFAULT_FORECAST_PRICE,
  DEFAULT_FORECAST_PROMO,
  DEFAULT_FORECAST_REGION,
  FORECAST_CATEGORIES,
  FORECAST_REGIONS,
} from "../constants/forecast";

import { useForecast } from "../hooks/useForecast";
import type { PredictionRequest } from "../types/prediction";

export default function ForecastDashboard() {
  /*
   * ---------------------------------------------------------
   * FORM STATE
   * ---------------------------------------------------------
   *
   * The dashboard owns the forecast scenario state.
   *
   * Child components remain presentational:
   *
   * ForecastForm
   *     -> renders inputs
   *     -> emits user interactions
   *
   * ForecastResult
   *     -> renders model output
   *
   * This page remains the orchestration/container layer.
   */

  const [date, setDate] = useState(DEFAULT_FORECAST_DATE);

  const [category, setCategory] = useState(
    DEFAULT_FORECAST_CATEGORY,
  );

  const [region, setRegion] = useState(DEFAULT_FORECAST_REGION);

  const [price, setPrice] = useState(DEFAULT_FORECAST_PRICE);

  const [promo, setPromo] = useState(DEFAULT_FORECAST_PROMO);

  /*
   * ---------------------------------------------------------
   * FORECAST REQUEST STATE
   * ---------------------------------------------------------
   *
   * The complete request lifecycle is owned by useForecast.
   *
   * Dashboard does not directly manage:
   *
   * - loading state
   * - prediction state
   * - API errors
   * - API request execution
   */

  const {
    prediction,
    loading,
    error,
    generateForecast,
  } = useForecast();

  /*
   * ---------------------------------------------------------
   * PRICE VALIDATION
   * ---------------------------------------------------------
   *
   * Validation remains at the container level because the
   * validated numeric value becomes part of PredictionRequest.
   */

  const numericPrice = Number(price);

  const priceError =
    price !== "" &&
    (!Number.isFinite(numericPrice) || numericPrice <= 0)
      ? "Price must be greater than 0."
      : "";

  /*
   * ---------------------------------------------------------
   * FORECAST SUBMISSION
   * ---------------------------------------------------------
   *
   * The page translates UI state into the API contract.
   */

  async function handlePrediction(): Promise<void> {
    if (!date) {
      return;
    }

    if (
      price === "" ||
      !Number.isFinite(numericPrice) ||
      numericPrice <= 0
    ) {
      return;
    }

    const payload: PredictionRequest = {
      date,
      category,
      region,
      price: numericPrice,
      promo: Number(promo),
    };

    await generateForecast(payload);
  }

  /*
   * ---------------------------------------------------------
   * DERIVED DISPLAY STATE
   * ---------------------------------------------------------
   */

  const promotionLabel =
    promo === "1" ? "Promotion active" : "No promotion";

  /*
   * ---------------------------------------------------------
   * RENDER
   * ---------------------------------------------------------
   */

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#eeeee9] text-neutral-950">
      {/* =====================================================
          PREMIUM BACKGROUND
      ===================================================== */}

      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -left-40 -top-40 h-[520px] w-[520px] rounded-full bg-white/80 blur-3xl" />

        <div className="absolute -right-40 top-20 h-[480px] w-[480px] rounded-full bg-[#e5e3dc]/80 blur-3xl" />

        <div className="absolute bottom-[-240px] left-1/3 h-[500px] w-[500px] rounded-full bg-white/60 blur-3xl" />
      </div>

      <div className="relative mx-auto min-h-screen max-w-7xl px-5 py-6 sm:px-8 lg:px-10">
        {/* =====================================================
            HEADER
        ===================================================== */}

        <ForecastHeader />

        {/* =====================================================
            MAIN CONTENT
        ===================================================== */}

        <div className="grid gap-6 py-8 lg:grid-cols-[minmax(0,1fr)_minmax(360px,0.8fr)]">
          {/* ===================================================
              FORECAST FORM
          =================================================== */}

          <ForecastForm
            date={date}
            category={category}
            region={region}
            price={price}
            promo={promo}
            categories={FORECAST_CATEGORIES}
            regions={FORECAST_REGIONS}
            priceError={priceError}
            error={error}
            loading={loading}
            onDateChange={setDate}
            onCategoryChange={setCategory}
            onRegionChange={setRegion}
            onPriceChange={setPrice}
            onPromoChange={setPromo}
            onSubmit={handlePrediction}
          />

          {/* ===================================================
              FORECAST RESULT
          =================================================== */}

          <ForecastResult
            prediction={prediction}
            loading={loading}
            category={category}
            region={region}
            price={price}
            promotionLabel={promotionLabel}
          />
        </div>

        {/* =====================================================
            INFERENCE ARCHITECTURE
        ===================================================== */}

        <section className="border-t border-black/[0.08] pt-6 pb-8">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-[11px] font-semibold tracking-[0.2em] text-neutral-400">
                PRODUCTION INFERENCE
              </p>

              <p className="mt-1 text-sm text-neutral-500">
                The forecast request flows through the deployed ML
                serving layer.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2 text-xs font-medium text-neutral-500">
              <span className="rounded-lg border border-black/[0.07] bg-white/80 px-3 py-2 shadow-sm">
                React
              </span>

              <ArrowRight className="h-3.5 w-3.5 text-neutral-300" />

              <span className="rounded-lg border border-black/[0.07] bg-white/80 px-3 py-2 shadow-sm">
                FastAPI
              </span>

              <ArrowRight className="h-3.5 w-3.5 text-neutral-300" />

              <span className="rounded-lg border border-black/[0.07] bg-white/80 px-3 py-2 shadow-sm">
                MLflow
              </span>

              <ArrowRight className="h-3.5 w-3.5 text-neutral-300" />

              <span className="rounded-lg border border-black/[0.07] bg-white/80 px-3 py-2 shadow-sm">
                Forecast
              </span>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}