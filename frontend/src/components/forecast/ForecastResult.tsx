import {
  ArrowRight,
  CheckCircle2,
  Loader2,
  TrendingUp,
} from "lucide-react";

import type { PredictionResponse } from "../../types/prediction";

interface ForecastResultProps {
  prediction: PredictionResponse | null;
  loading: boolean;
  category: string;
  region: string;
  price: string;
  promotionLabel: string;
}

export default function ForecastResult({
  prediction,
  loading,
  category,
  region,
  price,
  promotionLabel,
}: ForecastResultProps) {
  return (
    <section className="rounded-3xl border border-black/[0.07] bg-white/95 p-6 shadow-[0_20px_60px_rgba(0,0,0,0.06)] backdrop-blur sm:p-8">
      {/* =================================================
          HEADER
      ================================================= */}

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

      {/* =================================================
          EMPTY STATE
      ================================================= */}

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

      {/* =================================================
          LOADING STATE
      ================================================= */}

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

      {/* =================================================
          RESULT
      ================================================= */}

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

          {/* INFERENCE METADATA */}

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

            Forecast generated successfully by the production
            API.
          </div>
        </div>
      )}
    </section>
  );
}