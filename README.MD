# Self-Healing Machine Learning System

### Drift-Aware • Auto-Retraining • Production MLOps • AWS • CI/CD • Live ML Inference

> **An end-to-end MLOps platform that closes the loop between model development, production inference, monitoring, drift detection, retraining, evaluation, model promotion, and cloud deployment.**

[![🚀 Live Application](https://img.shields.io/badge/🚀%20Live%20Application-Open%20App-black?style=for-the-badge)](https://self-healing-ml-system-end-to-end-m.vercel.app)
[![▶ Watch 60-Second Demo](https://img.shields.io/badge/▶%20Watch%2060--Second%20Demo-f43f5e?style=for-the-badge)](https://drive.google.com/file/d/1ktxPcUzhlSFS8kfTfxHkJgPaY6qI59zy/view?usp=drive_link)
[![Source Code](https://img.shields.io/badge/Source%20Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/talarijayakiran/Self-Healing-ML-System-End-to-End-MLOps-Lifecycle-with-Drift-Aware-Retraining)
[![CI](https://github.com/talarijayakiran/Self-Healing-ML-System-End-to-End-MLOps-Lifecycle-with-Drift-Aware-Retraining/actions/workflows/ci.yml/badge.svg)](https://github.com/talarijayakiran/Self-Healing-ML-System-End-to-End-MLOps-Lifecycle-with-Drift-Aware-Retraining/actions/workflows/ci.yml)
[![CD](https://github.com/talarijayakiran/Self-Healing-ML-System-End-to-End-MLOps-Lifecycle-with-Drift-Aware-Retraining/actions/workflows/cd.yml/badge.svg)](https://github.com/talarijayakiran/Self-Healing-ML-System-End-to-End-MLOps-Lifecycle-with-Drift-Aware-Retraining/actions/workflows/cd.yml)

---

## 🚀 Live Production

**Web application:** https://self-healing-ml-system-end-to-end-m.vercel.app

**Production path:**

```text
Browser
   ↓
Vercel
   ↓ HTTPS
AWS API Gateway
   ↓
AWS EC2
   ↓
Docker
   ↓
FastAPI
   ↓
Production ML Model
```

**AWS region:** `ap-south-2`

The developer workstation is not part of the production runtime path.

---

## 🎥 60-Second Product Demo

[![▶ Watch 60-Second Demo](https://img.shields.io/badge/%E2%96%B6%20Watch%2060--Second%20Demo-f43f5e?style=for-the-badge)](https://drive.google.com/file/d/1ktxPcUzhlSFS8kfTfxHkJgPaY6qI59zy/view?usp=drive_link)

> A one-minute walkthrough of the deployed ML system covering live inference, production model serving, health/readiness, monitoring, and the operational MLOps workflow.

---

# 1. Why This Project Exists

Most ML projects stop at:

```text
Dataset → Train → Evaluate → Deploy
```

Production ML requires a longer lifecycle:

```text
Build
  ↓
Validate
  ↓
Register
  ↓
Promote
  ↓
Deploy
  ↓
Serve
  ↓
Observe
  ↓
Detect Drift
  ↓
Retrain
  ↓
Evaluate
  ↓
Quality Gate
  ↓
Promote
  ↓
Deploy
  ↓
Repeat
```

Production data changes. Features can drift. Models become stale. Deployments can fail. A newer model can also perform worse than the current production model.

> **The goal is not merely to train a model. The goal is to operate the model as a production system.**

---

# 2. What I Built

## Machine Learning

- Demand-forecasting model
- Feature engineering
- Deterministic inference feature construction
- Model evaluation
- Production inference

## MLOps

- MLflow model lifecycle management
- Model versioning and promotion
- Drift-aware retraining workflow
- Evaluation and quality gates
- Production model serving
- Prediction logging

## Backend

- FastAPI inference service
- Pydantic request validation
- Health and readiness endpoints
- REST API
- Prometheus metrics

## DevOps / Cloud

- Dockerized runtime
- GitHub Actions CI/CD
- Amazon ECR
- AWS EC2
- AWS API Gateway
- Vercel deployment

The result is a **live cloud-deployed ML application with a continuous MLOps lifecycle**, rather than only a training notebook.

---

# 3. System Architecture

```text
                                  INTERNET
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │       USER          │
                           │   Browser / Client  │
                           └──────────┬──────────┘
                                      │ HTTPS
                                      ▼
                           ┌─────────────────────┐
                           │       VERCEL        │
                           │   React + Vite      │
                           │ Production Frontend │
                           └──────────┬──────────┘
                                      │ HTTPS
                                      ▼
                           ┌─────────────────────┐
                           │   AWS API GATEWAY   │
                           │      HTTP API       │
                           └──────────┬──────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────┐
                  │             AWS EC2                │
                  │                                    │
                  │       ┌────────────────────┐       │
                  │       │ Docker Container   │       │
                  │       │                    │       │
                  │       │      FastAPI       │       │
                  │       │         │          │       │
                  │       │         ▼          │       │
                  │       │ Feature Builder     │       │
                  │       │         │          │       │
                  │       │         ▼          │       │
                  │       │ Production Model    │       │
                  │       │         │          │       │
                  │       │         ▼          │       │
                  │       │ Prediction Logger   │       │
                  │       │         │          │       │
                  │       │         ▼          │       │
                  │       │ Prometheus Metrics  │       │
                  │       └────────────────────┘       │
                  └──────────────────┬─────────────────┘
                                     │
                                     ▼
                           ┌─────────────────────┐
                           │ Monitoring / Drift  │
                           │ Detection Pipeline  │
                           └──────────┬──────────┘
                                      │
                               Drift detected?
                                  /       \
                                NO         YES
                                │           │
                                │           ▼
                                │       Retraining
                                │           │
                                │           ▼
                                │       Evaluation
                                │           │
                                │           ▼
                                │      Quality Gate
                                │           │
                                │           ▼
                                │    Model Promotion
                                │           │
                                └────── Deployment
                                           │
                                           ▼
                                   Production Model
```

---

# 4. Two Operating Loops

## Online inference loop

```text
User
 ↓
Frontend
 ↓
API Gateway
 ↓
FastAPI
 ↓
Pydantic Validation
 ↓
Canonical Feature Builder
 ↓
Production Model
 ↓
Prediction
 ↓
Prediction Logging
 ↓
Prometheus Metrics
 ↓
Response
```

## Offline MLOps loop

```text
Production / Monitoring Data
          ↓
     Drift Detection
          ↓
     Drift Decision
        /      \
      NO        YES
      │          │
      │          ▼
      │      Retraining
      │          ↓
      │      Evaluation
      │          ↓
      │      Quality Gate
      │          ↓
      │     Model Promotion
      │          ↓
      └──── Deployment
                │
                ▼
        Production Model
```

---

# 5. End-to-End Prediction Flow

```text
User input
   ↓
React frontend
   ↓
AWS API Gateway
   ↓
FastAPI
   ↓
Pydantic validation
   ↓
Canonical feature vector
   ↓
Production model
   ↓
Prediction
   ↓
Prediction + model version + request ID
   ↓
Frontend
```

The inference service is designed around a deterministic feature contract to reduce schema and feature-ordering failures.

---

# 6. Production API

## Prediction

```http
POST /api/predict
```

Example request:

```json
{
  "date": "2026-08-25",
  "category": "Electronics",
  "region": "North",
  "price": 100,
  "promo": 1
}
```

Example response:

```json
{
  "predicted_sales": 42.2,
  "model_version": "7",
  "request_id": "1a929b47-ea81-4fd8-be5b-dc5f2fd9cfea"
}
```

## Health

```http
GET /api/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "7"
}
```

## Readiness

```http
GET /api/ready
```

```json
{
  "status": "ready",
  "model_loaded": true,
  "model_version": "7"
}
```

If the model is unavailable, readiness returns `503 Service Unavailable`.

## Metrics

```http
GET /api/metrics
```

Exposes request and prediction telemetry for Prometheus.

---

# 7. Deterministic Inference Contract

The production feature builder derives canonical feature ordering from the processed inference schema:

```text
data/processed/processed_inference.csv
```

This helps protect against:

- column-order mismatch
- missing features
- inconsistent categorical encoding
- accidental inference-schema changes

The inference schema is treated as a **production contract** rather than an incidental implementation detail.

---

# 8. Model Lifecycle with MLflow

```text
Candidate Model
      ↓
Evaluation
      ↓
Quality Gate
      ↓
Eligible Model
      ↓
Production Promotion
      ↓
Production Version / Alias
      ↓
Inference Service
```

The lifecycle separates:

**training → evaluation → promotion → serving**

so model freshness is not treated as proof of model quality.

---

# 9. Model Serving Strategy

## Packaged model

```text
MLflow Model Artifact
        ↓
Docker Image
        ↓
EC2
        ↓
FastAPI
        ↓
Inference
```

This allows the production runtime to carry the required model artifact instead of depending on a development workstation path.

## Registry-based strategy

```text
MLflow Registry
      ↓
Production Alias
      ↓
Production Version
      ↓
FastAPI
      ↓
Inference
```

The deployment currently favors the packaged-model approach where the development registry artifact contains filesystem dependencies unsuitable for the Linux production container.

---

# 10. Self-Healing Lifecycle

```text
Production Model
      ↓
Live Inference
      ↓
Prediction Logging
      ↓
Drift Detection
      ↓
Drift detected?
    /       \
  NO         YES
  │           │
  │           ▼
  │       Retraining
  │           ↓
  │       Evaluation
  │           ↓
  │       Quality Gate
  │           ↓
  │       Promotion
  │           ↓
  │       Deployment
  │           │
  └───────────┘
       ↓
New Production Model
```

The key safety boundary is:

```text
Train → Evaluate → Quality Gate → Promote → Deploy
```

A model is not promoted simply because it is newer.

---

# 11. CI/CD

```text
Developer
   │
   ▼
Git Push / Pull Request
   │
   ▼
GitHub
   │
   ▼
GitHub Actions
   │
   ├──── CI → install + tests
   │
   └──── CD → build image
                 ↓
              Amazon ECR
                 ↓
                EC2
                 ↓
          container deployment
                 ↓
          health verification
```

## CI

- Dependency installation
- Automated tests
- Production-independent test mode where required

## CD

1. Checkout repository
2. Configure AWS credentials
3. Authenticate with ECR
4. Build Docker image
5. Tag image with Git commit SHA
6. Push image to ECR
7. Connect to EC2
8. Pull new image
9. Replace running container
10. Start container
11. Wait for startup
12. Validate application health

---

# 12. Cloud Architecture

```text
                    AWS
                     │
                     ▼
             ┌───────────────┐
             │ API Gateway   │
             │   HTTP API    │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │     EC2       │
             │               │
             │    Docker     │
             │       ↓       │
             │    FastAPI    │
             │       ↓       │
             │  ML Model     │
             └───────────────┘


CI/CD CONTROL PLANE

Developer
   ↓
GitHub
   ↓
GitHub Actions
   ↓
Amazon ECR
   ↓
AWS EC2
   ↓
Health Verification
```

**AWS region:** `ap-south-2`

---

# 13. Observability

The inference service exposes:

### Request metrics

```text
http_requests_total
```

with request metadata such as:

- method
- endpoint
- HTTP status

### Latency metrics

```text
http_request_latency_seconds
prediction_latency_seconds
```

### Prediction metadata

Prediction records can be associated with:

- request ID
- model version
- input metadata
- prediction

This creates an operational path from request → model version → response.

---

# 14. Reliability Design

### Health vs readiness

```text
/health
   ↓
Is the application alive?

/ready
   ↓
Can the application currently serve predictions?
```

### Failure boundary

If the model cannot be loaded, readiness reports:

```text
503 Service Unavailable
```

### Container recovery

The deployment is configured for automatic container restart behavior.

### Deployment verification

A successful image build is not treated as a successful release. The deployed application is checked after startup.

---

# 15. Failure Handling

| Failure | Expected behavior |
|---|---|
| Invalid request | Validation error |
| Model unavailable | `503 Service Unavailable` |
| Prediction failure | `500 Internal Server Error` |
| Application liveness | `/health` |
| Model readiness | `/ready` |
| Deployment failure | CI/CD reports failure |

---

# 16. Production Evidence

The project has been validated as a deployed system rather than only a local build.

| Evidence | Status |
|---|---|
| Frontend | ✅ Live on Vercel |
| AWS API Gateway | ✅ Live |
| FastAPI backend | ✅ Running on EC2 |
| Docker runtime | ✅ Running |
| Production model | ✅ Loaded |
| `/health` | ✅ Verified |
| `/ready` | ✅ Verified |
| `/predict` | ✅ Verified |
| Cross-machine API access | ✅ Verified |
| CI workflow | ✅ Configured |
| CD workflow | ✅ Configured |
| Amazon ECR | ✅ Configured |
| Prometheus metrics | ✅ Implemented |

> **Measured accuracy, latency, drift magnitude, and throughput numbers should be added only from reproducible benchmark/evaluation artifacts.**

---

# 17. Model Evaluation

Do not publish invented model metrics.

Recommended evidence table:

| Metric | Result |
|---|---:|
| Production model version | `7` |
| RMSE | `Add measured result` |
| MAE | `Add measured result` |
| R² | `Add measured result` |
| Validation samples | `Add measured result` |

This is deliberate: **credible evidence is more valuable than impressive-looking numbers.**

---

# 18. Drift & Retraining Evidence

```text
Production Data
      ↓
Drift Detection
      ↓
Threshold / Decision
      ↓
Retraining Trigger
      ↓
Candidate Model
      ↓
Evaluation
      ↓
Quality Gate
      ↓
Promotion
```

For a stronger engineering portfolio, publish real evidence under:

```text
docs/
├── experiments/
│   ├── model-evaluation.json
│   ├── drift-results.json
│   └── retraining-run.json
└── screenshots/
    ├── production-ui.png
    ├── github-actions-ci.png
    ├── github-actions-cd.png
    ├── aws-api-gateway.png
    ├── mlflow-model.png
    └── drift-retraining.png
```

Only add artifacts that actually exist.

---

# 19. Performance Benchmark

Recommended production benchmark:

| Metric | Result |
|---|---:|
| Requests | `Add measured result` |
| Average latency | `Add measured result` |
| P50 latency | `Add measured result` |
| P95 latency | `Add measured result` |
| P99 latency | `Add measured result` |
| Error rate | `Add measured result` |

> **Measure first, publish second.**

---

# 20. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| ML / DL | TensorFlow + ML ecosystem |
| Data | Pandas, NumPy |
| API | FastAPI |
| Validation | Pydantic |
| Model lifecycle | MLflow |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Registry | Amazon ECR |
| Compute | AWS EC2 |
| API Gateway | AWS API Gateway |
| Metrics | Prometheus |
| Frontend | React |
| Frontend build | Vite |
| Frontend hosting | Vercel |
| Version control | Git / GitHub |

---

# 21. Repository Structure

```text
.
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── data/
│   ├── processed/
│   └── monitoring/
├── frontend/
│   ├── src/
│   ├── public/
│   └── ...
├── src/
│   ├── serving/
│   ├── monitoring/
│   ├── retraining/
│   └── training/
├── exported_model/
├── monitoring/
├── scripts/
├── tests/
├── requirements.txt
├── Dockerfile
└── README.MD
```

---

# 22. Local Development

## Prerequisites

- Python
- Node.js / npm
- Docker
- Git

## Clone

```bash
git clone https://github.com/talarijayakiran/Self-Healing-ML-System-End-to-End-MLOps-Lifecycle-with-Drift-Aware-Retraining.git
cd Self-Healing-ML-System-End-to-End-MLOps-Lifecycle-with-Drift-Aware-Retraining
```

## Python environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Backend

```powershell
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

API docs:

```text
http://localhost:8000/docs
```

Health:

```text
http://localhost:8000/health
```

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

---

# 23. Docker

```bash
docker build -t self-healing-ml-api .
```

```bash
docker run -d \
  --name self-healing-ml-api \
  -p 8000:8000 \
  --restart unless-stopped \
  self-healing-ml-api
```

Verify:

```bash
curl http://localhost:8000/health
```

---

# 24. Engineering Decisions

### Why FastAPI?

Python-native inference serving, typed validation, automatic API documentation, and straightforward integration with ML code.

### Why MLflow?

To make model versioning and lifecycle management explicit instead of treating model files as anonymous artifacts.

### Why Docker?

To package application code, model runtime dependencies, and supporting libraries into a reproducible deployment artifact.

### Why AWS EC2?

To retain direct control over the containerized serving environment.

### Why API Gateway?

To provide a managed public HTTP entry point between the deployed frontend and inference backend.

### Why GitHub Actions?

To automate testing, image publishing, deployment, and post-deployment validation.

### Why Prometheus?

To expose standardized request and latency telemetry for operational monitoring.

---

# 25. Cost-Aware Design

The deployment is intentionally optimized for **portfolio-scale traffic**.

Current architecture:

```text
Vercel
   ↓
API Gateway
   ↓
Single EC2
   ↓
Docker
   ↓
FastAPI
```

This avoids introducing distributed infrastructure before workload requirements justify it.

At higher traffic or availability requirements:

```text
CDN
  ↓
Load Balancer
  ↓
Multiple inference replicas
  ↓
Autoscaling
  ↓
Managed observability
```

> **Infrastructure should be introduced because measured workload and reliability requirements demand it—not because a more complex diagram looks impressive.**

---

# 26. Security Considerations

Sensitive credentials should never be committed to Git.

Production hardening opportunities:

- IAM least privilege
- AWS Secrets Manager / Parameter Store
- Private EC2 networking
- Restricted security groups
- Authentication and authorization
- API rate limiting
- Centralized logging
- Network isolation
- Secret rotation
- HTTPS/TLS enforcement

---

# 27. Scaling Strategy

### Current

```text
Vercel
   ↓
API Gateway
   ↓
EC2
   ↓
Docker
   ↓
FastAPI
   ↓
Production Model
```

### Future

```text
Vercel / CDN
      ↓
Load Balancer
      ↓
Autoscaling inference replicas
      ↓
Queue / cache where required
      ↓
Managed model / data infrastructure
```

Potential future additions:

- ECS / EKS
- blue-green deployment
- canary releases
- automated rollback
- Infrastructure as Code
- multi-AZ deployment
- distributed tracing
- isolated model serving

---

# 28. Scope & Limitations

This is a **portfolio-scale, production-oriented MLOps system**, not a claim of hyperscale production traffic.

Current limitations include:

- single primary EC2 inference host
- no multi-region deployment
- no horizontal inference autoscaling
- limited enterprise identity integration
- no formal blue-green/canary rollout
- no multi-region disaster recovery
- no full distributed tracing platform

Documenting these boundaries is intentional. Production maturity includes knowing what the system does **and does not** guarantee.

---

# 29. What This Project Demonstrates

## Machine Learning

- Feature engineering
- Model training
- Evaluation
- Inference
- Model versioning

## MLOps

- Model lifecycle management
- Drift detection
- Retraining workflow
- Quality gates
- Model promotion
- Production serving

## Backend Engineering

- FastAPI
- REST API
- Pydantic
- Health checks
- Readiness checks
- Error handling

## DevOps

- Docker
- Git
- GitHub Actions
- CI/CD
- Amazon ECR
- Automated EC2 deployment

## Cloud

- AWS EC2
- AWS API Gateway
- Amazon ECR
- Vercel

## Observability

- Prometheus
- Request metrics
- Latency metrics
- Prediction logging
- Model-version tracking

---

# 30. Why This Project Matters

A production ML system must answer:

```text
Can I serve the model?
Can I monitor it?
Can I detect drift?
Can I retrain it?
Can I evaluate the replacement?
Can I promote it safely?
Can I deploy it automatically?
Can I recover when something fails?
Can I identify which model served a prediction?
```

This project is structured around those questions.

> **A model that is trained but cannot be safely operated, monitored, evaluated, and improved is not a complete ML system.**

---

# 31. Recruiter Review Path

A recruiter or hiring manager can evaluate the project in this order:

```text
🚀 Live Application
        ↓
▶ 60-Second Demo
        ↓
🏗️ Architecture
        ↓
🔁 MLOps Lifecycle
        ↓
⚙️ CI/CD
        ↓
☁️ AWS Deployment
        ↓
📊 Observability
        ↓
🧠 Model Lifecycle
        ↓
🔍 Source Code
```

This is intentionally designed to make the **engineering evidence discoverable quickly**.

---

# 👤 Author

**Talari Jayakiran**

**AI Engineer | MLOps Engineer | Machine Learning Engineer**

Focus areas:

- Production ML systems
- MLOps
- Cloud deployment
- ML inference systems
- AI engineering
- Generative AI / LLMOps

- GitHub: https://github.com/talarijayakiran
- Project: https://github.com/talarijayakiran/Self-Healing-ML-System-End-to-End-MLOps-Lifecycle-with-Drift-Aware-Retraining
- Live Application: https://self-healing-ml-system-end-to-end-m.vercel.app

---

## ⭐ Final Principle

> **ML engineering does not end when the model reaches production. The real system begins there.**
