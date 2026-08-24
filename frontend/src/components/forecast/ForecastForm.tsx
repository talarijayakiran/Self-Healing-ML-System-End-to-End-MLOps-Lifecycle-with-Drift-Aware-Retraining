import {
  AlertCircle,
  ArrowRight,
  Loader2,
  Sparkles,
} from "lucide-react";



interface ForecastFormProps {
  date: string;
  category: string;
  region: string;
  price: string;
  promo: string;

  categories: readonly string[];
  regions: readonly string[];

  priceError: string;
  error: string;
  loading: boolean;

  onDateChange: (value: string) => void;
  onCategoryChange: (value: string) => void;
  onRegionChange: (value: string) => void;
  onPriceChange: (value: string) => void;
  onPromoChange: (value: string) => void;

  onSubmit: () => void;
}

export default function ForecastForm({
  date,
  category,
  region,
  price,
  promo,
  categories,
  regions,
  priceError,
  error,
  loading,
  onDateChange,
  onCategoryChange,
  onRegionChange,
  onPriceChange,
  onPromoChange,
  onSubmit,
}: ForecastFormProps) {
  return (
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
              onDateChange(event.target.value)
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
              onCategoryChange(event.target.value)
            }
            className="h-12 w-full rounded-xl border border-neutral-200 bg-neutral-50 px-4 text-sm text-neutral-900 outline-none transition focus:border-neutral-950 focus:bg-white focus:ring-4 focus:ring-neutral-950/5"
          >
            {categories.map((item) => (
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
              onRegionChange(event.target.value)
            }
            className="h-12 w-full rounded-xl border border-neutral-200 bg-neutral-50 px-4 text-sm text-neutral-900 outline-none transition focus:border-neutral-950 focus:bg-white focus:ring-4 focus:ring-neutral-950/5"
          >
            {regions.map((item) => (
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
                onPriceChange(event.target.value)
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
              onClick={() => onPromoChange("0")}
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
              onClick={() => onPromoChange("1")}
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
          onClick={onSubmit}
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
  );
}