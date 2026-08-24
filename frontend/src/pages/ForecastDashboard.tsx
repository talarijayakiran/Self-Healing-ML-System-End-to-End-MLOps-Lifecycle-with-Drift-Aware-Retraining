import { useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Loader2,
  Sparkles,
  TrendingUp,
} from "lucide-react";

import { useForecast } from "../hooks/useForecast";
import type { PredictionRequest } from "../types/prediction";

const CATEGORIES = [
  "Electronics",
  "Furniture",
  "Grocery",
];

const REGIONS = [
  "North",
  "South",
  "East",
  "West",
];

const DEFAULT_DATE = "2024-01-10";
const DEFAULT_PRICE = "1000";
const DEFAULT_PROMO = "1";

export default function ForecastDashboard() {
  /*
   * ---------------------------------------------------------
   * FORM STATE
   * ---------------------------------------------------------
   */

  const [date, setDate] = useState(DEFAULT_DATE);
  const [category, setCategory] = useState("Electronics");
  const [region, setRegion] = useState("North");
  const [price, setPrice] = useState(DEFAULT_PRICE);
  const [promo, setPromo] = useState(DEFAULT_PROMO);

  /*
   * ---------------------------------------------------------
   * FORECAST REQUEST STATE
   * ---------------------------------------------------------
   *
   * Request lifecycle is owned by the custom hook.
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
   */

  const numericPrice = Number(price);

  const priceError =
    price !== "" &&
    (!Number.isFinite(numericPrice) || numericPrice <= 0)
      ? "Price must be greater than 0."
      : "";

  /*
   * ---------------------------------------------------------
   * PREDICTION
   * ---------------------------------------------------------
   */

  async function handlePrediction() {
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

  const promotionLabel =
    promo === "1"
      ? "Promotion active"
      : "No promotion";

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

        <header className="flex items-start justify-between border-b border-black/[0.08] pb-6">

          <div>
            <div className="flex items-center gap-3">

              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-neutral-950 text-white shadow-sm">
                <TrendingUp className="h-4 w-4" />
              </div>

              <div>
                <p className="text-[11px] font-semibold tracking-[0.24em] text-neutral-600">
                  NOVACART
                </p>

                <p className="mt-0.5 text-xs text-neutral-400">
                  Retail Demand Intelligence
                </p>
              </div>

            </div>

            <h1 className="mt-8 text-3xl font-semibold tracking-[-0.04em] sm:text-4xl">
              Forecast Dashboard
            </h1>

            <p className="mt-2 max-w-xl text-sm leading-6 text-neutral-500">
              Generate a demand forecast from the retail signals
              used by the production ML system.
            </p>
          </div>

          <div className="hidden items-center gap-2 rounded-full border border-black/[0.08] bg-white/80 px-3.5 py-2 text-xs font-medium text-neutral-600 shadow-sm backdrop-blur sm:flex">
            <CheckCircle2 className="h-3.5 w-3.5" />
            Production ML Platform
          </div>

        </header>

        {/* =====================================================
            MAIN CONTENT
        ===================================================== */}

        <div className="grid gap-6 py-8 lg:grid-cols-[minmax(0,1fr)_minmax(360px,0.8fr)]">

          {/* ===================================================
              FORECAST SCENARIO
          =================================================== */}

          <section className="rounded-3xl border border-black/[0.07] bg-white/95 p-6 shadow-[0_20px_60px_rgba(0,0,0,0.06)] backdrop-blur sm:p-8">

            <div className="flex items-start justify-between gap-6">

              <div>
                <div className="flex items-center gap-2 text-[11px] font-semibold tracking-[0.2em] text-neutral-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-neutral-950" />
                  FORECAST SCENARIO
                </div>

                <h2 className="mt-3 text-2xl font-semibold tracking-[-0.03em]">
                  Define the retail scenario
                </h2>

                <p className="mt-2 max-w-xl text-sm leading-6 text-neutral-500">
                  Adjust the commercial signals that influence the
                  expected demand for this scenario.
                </p>
              </div>

              <div className="hidden h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-neutral-200 bg-neutral-50 sm:flex">
                <Sparkles className="h-4 w-4 text-neutral-500" />
              </div>

            </div>

            {/* INPUT GRID */}

            <div className="mt-8 grid gap-5 sm:grid-cols-2">

              {/* DATE */}

              <div>
                <label
                  htmlFor="forecast-date"
                  className="mb-2 block text-sm font-medium text-neutral-800"
                >
                  Forecast date
                </label>

                <input
                  id="forecast-date"
                  type="date"
                  value={date}
                  onChange={(event) =>
                    setDate(event.target.value)
                  }
                  className="h-12 w-full rounded-xl border border-neutral-200 bg-neutral-50 px-4 text-sm text-neutral-900 outline-none transition focus:border-neutral-950 focus:bg-white focus:ring-4 focus:ring-neutral-950/5"
                />

                <p className="mt-1.5 text-xs text-neutral-400">
                  Date used for the demand forecast.
                </p>
              </div>

              {/* CATEGORY */}

              <div>
                <label
                  htmlFor="category"
                  className="mb-2 block text-sm font-medium text-neutral-800"
                >
                  Product category
                </label>

                <select
                  id="category"
                  value={category}
                  onChange={(event) =>
                    setCategory(event.target.value)
                  }
                  className="h-12 w-full rounded-xl border border-neutral-200 bg-neutral-50 px-4 text-sm text-neutral-900 outline-none transition focus:border-neutral-950 focus:bg-white focus:ring-4 focus:ring-neutral-950/5"
                >
                  {CATEGORIES.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>

                <p className="mt-1.5 text-xs text-neutral-400">
                  Product segment being forecast.
                </p>
              </div>

              {/* REGION */}

              <div>
                <label
                  htmlFor="region"
                  className="mb-2 block text-sm font-medium text-neutral-800"
                >
                  Sales region
                </label>

                <select
                  id="region"
                  value={region}
                  onChange={(event) =>
                    setRegion(event.target.value)
                  }
                  className="h-12 w-full rounded-xl border border-neutral-200 bg-neutral-50 px-4 text-sm text-neutral-900 outline-none transition focus:border-neutral-950 focus:bg-white focus:ring-4 focus:ring-neutral-950/5"
                >
                  {REGIONS.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>

                <p className="mt-1.5 text-xs text-neutral-400">
                  Regional demand segment.
                </p>
              </div>

              {/* PRICE */}

              <div>
                <label
                  htmlFor="product-price"
                  className="mb-2 block text-sm font-medium text-neutral-800"
                >
                  Product price
                </label>

                <div className="relative">

                  <input
                    id="product-price"
                    type="number"
                    min="0.01"
                    step="0.01"
                    value={price}
                    onChange={(event) =>
                      setPrice(event.target.value)
                    }
                    placeholder="1000"
                    aria-invalid={Boolean(priceError)}
                    aria-describedby={
                      priceError
                        ? "product-price-error"
                        : "product-price-help"
                    }
                    className={`h-12 w-full rounded-xl border bg-neutral-50 px-4 pr-12 text-sm text-neutral-900 outline-none transition focus:bg-white focus:ring-4 ${
                      priceError
                        ? "border-red-400 bg-red-50/40 focus:border-red-500 focus:ring-red-500/10"
                        : "border-neutral-200 focus:border-neutral-950 focus:ring-neutral-950/5"
                    }`}
                  />

                  {priceError && (
                    <AlertCircle className="absolute right-4 top-1/2 h-4 w-4 -translate-y-1/2 text-red-500" />
                  )}

                </div>

                {priceError ? (
                  <p
                    id="product-price-error"
                    className="mt-1.5 flex items-center gap-1.5 text-xs font-medium text-red-600"
                  >
                    <AlertCircle className="h-3.5 w-3.5" />
                    Price must be greater than 0.
                  </p>
                ) : (
                  <p
                    id="product-price-help"
                    className="mt-1.5 text-xs text-neutral-400"
                  >
                    Price signal supplied to the model.
                  </p>
                )}
              </div>

              {/* PROMOTION */}

              <div className="sm:col-span-2">
                <span className="mb-2 block text-sm font-medium text-neutral-800">
                  Promotion
                </span>

                <div className="grid gap-3 sm:grid-cols-2">

                  <button
                    type="button"
                    onClick={() => setPromo("0")}
                    aria-pressed={promo === "0"}
                    className={`h-12 rounded-xl border text-sm font-medium transition ${
                      promo === "0"
                        ? "border-neutral-950 bg-neutral-950 text-white shadow-sm"
                        : "border-neutral-200 bg-neutral-50 text-neutral-600 hover:border-neutral-300 hover:bg-white"
                    }`}
                  >
                    No Promotion
                  </button>

                  <button
                    type="button"
                    onClick={() => setPromo("1")}
                    aria-pressed={promo === "1"}
                    className={`h-12 rounded-xl border text-sm font-medium transition ${
                      promo === "1"
                        ? "border-neutral-950 bg-neutral-950 text-white shadow-sm"
                        : "border-neutral-200 bg-neutral-50 text-neutral-600 hover:border-neutral-300 hover:bg-white"
                    }`}
                  >
                    Promotion Active
                  </button>

                </div>
              </div>
            </div>

            {/* GENERAL ERROR */}

            {error && (
              <div
                role="alert"
                className="mt-6 flex gap-3 rounded-2xl border border-red-200 bg-red-50 p-4"
              >
                <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-600" />

                <div>
                  <p className="text-sm font-semibold text-red-800">
                    Forecast unavailable
                  </p>

                  <p className="mt-1 text-sm leading-5 text-red-700">
                    {error}
                  </p>
                </div>
              </div>
            )}

            {/* ACTION */}

            <div className="mt-8 border-t border-neutral-100 pt-6">

              <button
                type="button"
                onClick={handlePrediction}
                disabled={
                  loading ||
                  Boolean(priceError) ||
                  price === "" ||
                  !date
                }
                className="group flex h-13 w-full items-center justify-center gap-2 rounded-xl bg-neutral-950 px-5 text-sm font-semibold text-white shadow-sm transition hover:bg-neutral-800 active:scale-[0.995] disabled:cursor-not-allowed disabled:bg-neutral-300 disabled:text-neutral-500"
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Generating forecast...
                  </>
                ) : (
                  <>
                    Generate Forecast
                    <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
                  </>
                )}
              </button>

              <p className="mt-3 text-center text-xs text-neutral-400">
                Sends this scenario to the production forecasting API.
              </p>

            </div>
          </section>

          {/* ===================================================
              FORECAST RESULT
          =================================================== */}

          <section className="rounded-3xl border border-black/[0.07] bg-white/95 p-6 shadow-[0_20px_60px_rgba(0,0,0,0.06)] backdrop-blur sm:p-8">

            <div className="flex items-center justify-between">

              <div>
                <div className="flex items-center gap-2 text-[11px] font-semibold tracking-[0.2em] text-neutral-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-neutral-950" />
                  MODEL OUTPUT
                </div>

                <h2 className="mt-3 text-2xl font-semibold tracking-[-0.03em]">
                  Forecast result
                </h2>
              </div>

              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-neutral-950 text-white shadow-sm">
                <TrendingUp className="h-5 w-5" />
              </div>

            </div>

            {/* EMPTY STATE */}

            {!prediction && !loading && (
              <div className="mt-8 flex min-h-[390px] flex-col items-center justify-center rounded-2xl border border-dashed border-neutral-200 bg-neutral-50 px-6 text-center">

                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-white shadow-sm ring-1 ring-neutral-200">
                  <TrendingUp className="h-6 w-6 text-neutral-400" />
                </div>

                <h3 className="mt-5 text-sm font-semibold text-neutral-800">
                  No forecast generated yet
                </h3>

                <p className="mt-2 max-w-xs text-sm leading-5 text-neutral-400">
                  Configure a retail scenario and generate a forecast
                  to see the model output here.
                </p>

              </div>
            )}

            {/* LOADING STATE */}

            {loading && (
              <div className="mt-8 flex min-h-[390px] flex-col items-center justify-center rounded-2xl border border-neutral-200 bg-neutral-50 px-6 text-center">

                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-neutral-950 text-white">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>

                <h3 className="mt-5 text-sm font-semibold text-neutral-800">
                  Running demand forecast
                </h3>

                <p className="mt-2 max-w-xs text-sm leading-5 text-neutral-400">
                  Sending the scenario through the production
                  inference service.
                </p>

                <div className="mt-6 flex items-center gap-2 text-[11px] font-medium uppercase tracking-[0.14em] text-neutral-400">
                  <span>React</span>
                  <ArrowRight className="h-3 w-3" />
                  <span>FastAPI</span>
                  <ArrowRight className="h-3 w-3" />
                  <span>MLflow</span>
                </div>

              </div>
            )}

            {/* RESULT */}

            {prediction && !loading && (
              <div className="mt-8">

                {/* PRIMARY FORECAST */}

                <div className="rounded-2xl bg-neutral-950 p-7 text-white shadow-[0_15px_35px_rgba(0,0,0,0.12)] sm:p-8">

                  <p className="text-[11px] font-semibold tracking-[0.2em] text-neutral-400">
                    PREDICTED DEMAND
                  </p>

                  <div className="mt-4 flex items-end gap-3">

                    <span className="text-6xl font-semibold tracking-[-0.055em] sm:text-7xl">
                      {prediction.predicted_sales.toFixed(2)}
                    </span>

                    <span className="mb-2 text-sm text-neutral-400">
                      units
                    </span>

                  </div>

                  <p className="mt-4 text-sm text-neutral-400">
                    Expected demand for the selected retail scenario.
                  </p>

                </div>

                {/* SCENARIO */}

                <div className="mt-5">

                  <p className="text-[11px] font-semibold tracking-[0.2em] text-neutral-400">
                    SCENARIO
                  </p>

                  <div className="mt-3 grid grid-cols-2 gap-2">

                    <div className="rounded-xl border border-neutral-200 bg-neutral-50 px-4 py-3">
                      <p className="text-xs text-neutral-400">
                        Category
                      </p>

                      <p className="mt-1 text-sm font-semibold">
                        {category}
                      </p>
                    </div>

                    <div className="rounded-xl border border-neutral-200 bg-neutral-50 px-4 py-3">
                      <p className="text-xs text-neutral-400">
                        Region
                      </p>

                      <p className="mt-1 text-sm font-semibold">
                        {region}
                      </p>
                    </div>

                    <div className="rounded-xl border border-neutral-200 bg-neutral-50 px-4 py-3">
                      <p className="text-xs text-neutral-400">
                        Price
                      </p>

                      <p className="mt-1 text-sm font-semibold">
                        {Number(price).toLocaleString()}
                      </p>
                    </div>

                    <div className="rounded-xl border border-neutral-200 bg-neutral-50 px-4 py-3">
                      <p className="text-xs text-neutral-400">
                        Promotion
                      </p>

                      <p className="mt-1 text-sm font-semibold">
                        {promotionLabel}
                      </p>
                    </div>

                  </div>
                </div>

                {/* METADATA */}

                <div className="mt-5 border-t border-neutral-100 pt-5">

                  <p className="text-[11px] font-semibold tracking-[0.2em] text-neutral-400">
                    INFERENCE METADATA
                  </p>

                  <div className="mt-3 space-y-2">

                    <div className="flex items-center justify-between rounded-xl border border-neutral-200 px-4 py-3">
                      <span className="text-sm text-neutral-500">
                        Model version
                      </span>

                      <span className="text-sm font-semibold">
                        {prediction.model_version}
                      </span>
                    </div>

                    <div className="flex items-center justify-between gap-4 rounded-xl border border-neutral-200 px-4 py-3">

                      <span className="text-sm text-neutral-500">
                        Request ID
                      </span>

                      <span className="max-w-[190px] truncate font-mono text-xs text-neutral-700">
                        {prediction.request_id}
                      </span>

                    </div>

                  </div>
                </div>

                {/* SUCCESS */}

                <div className="mt-5 flex items-center gap-2 rounded-xl bg-neutral-50 px-4 py-3 text-xs font-medium text-neutral-500">
                  <CheckCircle2 className="h-4 w-4 text-neutral-700" />
                  Forecast generated successfully by the production API.
                </div>

              </div>
            )}
          </section>
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