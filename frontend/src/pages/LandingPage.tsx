import {
  ArrowRight,
  BarChart3,
  Boxes,
  BrainCircuit,
  CircleCheck,
  Package,
  TrendingUp,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

const businessProblems = [
  {
    icon: Boxes,
    title: "Overstock",
    description: "Excess inventory ties up working capital.",
  },
  {
    icon: Package,
    title: "Stockouts",
    description: "Insufficient inventory creates lost sales.",
  },
  {
    icon: TrendingUp,
    title: "Uncertainty",
    description: "Poor demand visibility makes planning harder.",
  },
];

const systemStages = [
  "Retail Data",
  "Feature Engineering",
  "ML Forecast",
  "Production API",
  "Retail Decision",
];

const technologies = [
  "Python",
  "Machine Learning",
  "FastAPI",
  "MLflow",
  "MLOps",
];

export default function LandingPage() {
  const navigate = useNavigate();

  const handleContinue = () => {
    navigate("/forecast");
  };

  return (
    <main className="min-h-screen bg-[#f7f7f5] text-neutral-900">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-6 py-6 lg:px-10">
        {/* Header */}
        <header className="flex items-center justify-between border-b border-neutral-200 pb-5">
          <div>
            <p className="text-sm font-semibold tracking-[0.2em] text-neutral-900">
              NOVACART
            </p>

            <p className="mt-1 text-xs text-neutral-500">
              Retail Demand Intelligence
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs font-medium text-neutral-500">
            <CircleCheck className="h-4 w-4" />
            Production ML Platform
          </div>
        </header>

        {/* Hero */}
        <section className="grid flex-1 items-center gap-14 py-14 lg:grid-cols-[1.1fr_0.9fr]">
          {/* Left: Business Problem */}
          <div>
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-neutral-200 bg-white px-3 py-1.5 text-xs font-medium text-neutral-600 shadow-sm">
              <BrainCircuit className="h-3.5 w-3.5" />
              Machine Learning · MLOps · Retail
            </div>

            <h1 className="max-w-3xl text-5xl font-semibold leading-[1.05] tracking-[-0.04em] sm:text-6xl">
              Retail demand is uncertain.

              <span className="mt-2 block text-neutral-500">
                Your inventory decisions don't have to be.
              </span>
            </h1>

            <p className="mt-7 max-w-2xl text-base leading-7 text-neutral-600 sm:text-lg">
              NovaCart uses machine learning to forecast retail demand
              from product, pricing, promotion, date, and regional
              signals—helping businesses make better inventory decisions.
            </p>

            <button
              type="button"
              onClick={handleContinue}
              className="mt-9 inline-flex items-center gap-2 rounded-xl bg-neutral-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-neutral-700"
            >
              Open Forecasting Platform

              <ArrowRight className="h-4 w-4" />
            </button>
          </div>

          {/* Right: Business Problem Panel */}
          <div className="rounded-3xl border border-neutral-200 bg-white p-6 shadow-sm">
            <div className="mb-6 flex items-start justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-neutral-400">
                  The retail challenge
                </p>

                <h2 className="mt-2 text-xl font-semibold tracking-tight">
                  Demand uncertainty creates operational risk.
                </h2>
              </div>

              <BarChart3 className="h-6 w-6 text-neutral-400" />
            </div>

            <div className="space-y-3">
              {businessProblems.map((problem) => {
                const Icon = problem.icon;

                return (
                  <div
                    key={problem.title}
                    className="flex gap-4 rounded-2xl border border-neutral-100 bg-neutral-50 p-4"
                  >
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white shadow-sm">
                      <Icon className="h-5 w-5 text-neutral-700" />
                    </div>

                    <div>
                      <h3 className="text-sm font-semibold">
                        {problem.title}
                      </h3>

                      <p className="mt-1 text-sm leading-5 text-neutral-500">
                        {problem.description}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Architecture Flow */}
            <div className="mt-6 border-t border-neutral-200 pt-5">
              <p className="mb-4 text-xs font-semibold uppercase tracking-[0.16em] text-neutral-400">
                From retail data to business decision
              </p>

              <div className="flex flex-wrap items-center gap-2">
                {systemStages.map((stage, index) => (
                  <div
                    key={stage}
                    className="flex items-center gap-2"
                  >
                    <span className="rounded-lg border border-neutral-200 bg-white px-2.5 py-1.5 text-xs font-medium text-neutral-600">
                      {stage}
                    </span>

                    {index < systemStages.length - 1 && (
                      <ArrowRight className="h-3.5 w-3.5 text-neutral-300" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Technology Footer */}
        <footer className="border-t border-neutral-200 pt-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <p className="text-xs text-neutral-400">
              Production-oriented retail forecasting system
            </p>

            <div className="flex flex-wrap gap-2">
              {technologies.map((technology) => (
                <span
                  key={technology}
                  className="rounded-md bg-neutral-100 px-2.5 py-1 text-[11px] font-medium text-neutral-500"
                >
                  {technology}
                </span>
              ))}
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}