import {
  CheckCircle2,
  TrendingUp,
} from "lucide-react";

export default function ForecastHeader() {
  return (
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
  );
}