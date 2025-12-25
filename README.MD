#  Self-Healing Machine Learning System

### Drift-Aware, Auto-Retraining, Production-Grade MLOps Architecture

> **A continuously running ML system that monitors itself, detects data drift from live traffic, retrains automatically, and serves predictions without downtime.**

This repository represents a **real-world ML production system**, not a demo or academic project.
It is designed to stay alive **24/7**, recover from failures, and evolve as data changes.

---

##  What Makes This Project Different

Most ML projects stop at:

* Training a model once
* Exposing a prediction API
* Calling it “production”

This system goes **much further**:

- Runs continuously on cloud infrastructure
- Detects data drift from real inference traffic
- Automatically retrains when drift is confirmed
- Registers new model versions without manual steps
- Survives server reboot & container restart
- Exposes full observability (metrics, latency, health)

This is how **real ML systems** work in production environments.

---

##  System Architecture (Conceptual)

```
┌──────────────┐
│   Client     │
│ (REST Call)  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│ FastAPI Inference Server │
│  • Feature Builder       │
│  • MLflow Model          │
│  • Prediction Logger     │
│  • Prometheus Metrics    │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Live Prediction Logs     │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Drift Detection Engine   │
│  • Statistical Analysis  │
│  • Drift Report          │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Auto Retraining Pipeline │
│  • Triggered by Drift    │
│  • MLflow Tracking       │
│  • Model Versioning      │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Updated Model Artifact   │
│ Used by Inference API    │
└──────────────────────────┘
```

---

##  Core Design Principles

### 1️ Deterministic Inference

* Feature order is locked using a template
* No silent column mismatch
* No runtime surprises

### 2️ Drift-First Thinking

* Drift is treated as a **first-class signal**
* Model accuracy degradation is not ignored

### 3️ Automation Over Manual Ops

* Retraining does **not** require a human
* Decisions are driven by data, not schedules

### 4️ Observability by Default

* Every request is measured
* Every prediction is logged
* Every failure is visible

---

##  Technology Stack (Why These Choices)

| Component      | Tool       | Reason                        |
| -------------- | ---------- | ----------------------------- |
| API            | FastAPI    | Async, fast, production-ready |
| Model Registry | MLflow     | Versioning & lineage          |
| Deployment     | Docker     | Reproducibility               |
| Monitoring     | Prometheus | Industry standard             |
| Dashboards     | Grafana    | Real-time observability       |
| Scheduling     | Cron (EC2) | Simple & reliable             |
| Cloud          | AWS EC2    | Full control, low cost        |

No vendor lock-in. No magic services.

---

## 📁 Repository Structure (Intentional, Not Random)

```
.
├── src/
│   ├── serving/            # Inference API
│   ├── monitoring/         # Drift & metrics logic
│   ├── retraining/         # Auto retrain orchestration
│   └── training/           # Model training pipeline
│
├── exported_model/         # MLflow model artifact
├── data/
│   ├── processed/          # Feature templates
│   └── monitoring/         # Drift reports
│
├── docker/                 # Docker configuration
├── scripts/                # Automation scripts
├── monitoring/             # Prometheus configs
└── README.md
```

Every directory exists for a reason.

---

##  API Endpoints

### Health Check

```
GET /health
```

Response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

Used by:

* Load balancers
* Monitoring systems
* Deployment validation

---

### Prediction Endpoint

```
POST /predict
```

Request:

```json
{
  "date": "2024-01-10",
  "category": "Electronics",
  "region": "North",
  "price": 1000,
  "promo": 1
}
```

Response:

```json
{
  "predicted_sales": 34.92
}
```

---

### Metrics

```
GET /metrics
```

Exposed for Prometheus scraping.

---

##  Observability (Real, Not Fake)

Tracked metrics include:

* Request count by endpoint
* Request latency
* Prediction latency
* Error rates

Grafana dashboards visualize:

* Traffic patterns
* System health
* Model behavior over time

---

##  Self-Healing Logic (Step-by-Step)

1. API receives live traffic
2. Predictions are logged
3. Drift detection compares:

   * Training distribution
   * Live inference distribution
4. Drift report is generated
5. If drift exceeds threshold:

   * Retraining pipeline is triggered
6. New model is trained
7. Model is registered in MLflow
8. System continues serving predictions

No downtime. No manual retraining.

---

## Reliability Proof

* Survives EC2 reboot
* Survives Docker restart
* Model reloads correctly
* API continues serving traffic
* Retraining works post-deployment

This has been **validated in a live cloud environment**.

---

This project demonstrates:

* Real MLOps thinking
* Production debugging ability
* Cloud deployment experience
* Monitoring & observability
* Automated ML lifecycle design

This is **how ML engineers work**, not how students code.

---

##  Current Status

**Production-Ready**
**24/7 Cloud Deployed**
**Self-Healing Enabled**

---


##  Author Note

TALARI JAYAKIRAN
This system was built end-to-end with a **production mindset**:

> *If it breaks at 3 AM, it’s not production.*
> *If services are provide to users in 24/7 then only that system is production.*


