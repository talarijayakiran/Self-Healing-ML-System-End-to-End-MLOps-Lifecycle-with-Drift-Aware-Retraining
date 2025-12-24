\# Retail Demand Forecasting — End-to-End MLOps System



> ⚠ This project is under active development.

> The README will be finalized after completing all MLOps lifecycle phases.



---



\## 📌 Project Goal

Build a production-grade Retail Demand Forecasting system following real-world MLOps practices:

\- Data ingestion \& validation

\- Feature engineering with strict contracts

\- Model training \& MLflow registry

\- Versioned model serving via FastAPI

\- Monitoring \& lifecycle management (upcoming)



---



\##  Repository Structure (Final)



```text

retail-mlops/

│

├── README.md

├── data/

│   ├── raw/                 # Raw input data

│   ├── processed/           # Validated \& engineered data

│

├── src/

│   ├── config/              # Single source of truth (schema, contracts)

│   ├── ingestion/           # Data loading

│   ├── validation/          # Data validation

│   ├── transformation/      # Feature engineering

│   ├── training/            # Model training + MLflow

│   ├── registry/            # Model registry checks

│   ├── serving/             # FastAPI inference service

│

├── mlruns/                  # MLflow tracking data

├── venv/                    # Virtual environment

└── requirements.txt

