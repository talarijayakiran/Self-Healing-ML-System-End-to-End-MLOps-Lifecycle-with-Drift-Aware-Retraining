# Self-Healing Machine Learning System

### Drift-Aware • Auto-Retraining • Production MLOps • AWS • CI/CD • Live ML Inference

> An end-to-end machine learning system that closes the loop between model training, production inference, monitoring, drift detection, retraining, model promotion, and cloud deployment.

[![CI](https://github.com/talarijayakiran/Self-Healing-ML-System-End-to-End-MLOps-Lifecycle-with-Drift-Aware-Retraining/actions/workflows/ci.yml/badge.svg)](https://github.com/talarijayakiran/Self-Healing-ML-System-End-to-End-MLOps-Lifecycle-with-Drift-Aware-Retraining/actions/workflows/ci.yml)
[![CD](https://github.com/talarijayakiran/Self-Healing-ML-System-End-to-End-MLOps-Lifecycle-with-Drift-Aware-Retraining/actions/workflows/cd.yml/badge.svg)](https://github.com/talarijayakiran/Self-Healing-ML-System-End-to-End-MLOps-Lifecycle-with-Drift-Aware-Retraining/actions/workflows/cd.yml)

---

## 🚀 Live Production

**Web application:** https://self-healing-ml-system-end-to-end-m.vercel.app

**Production path:**

```text
Browser → Vercel → AWS API Gateway → AWS EC2 → Docker → FastAPI → Production ML Model
```

**AWS region:** `ap-south-2`

The developer workstation is not part of the production runtime path. Once the Vercel and AWS infrastructure is running, users can access the application from other machines without the developer PC running the application locally.

---

# 1. Why This Project Exists

Most ML projects stop here:

```text
Dataset → Train → Evaluate → Deploy
```

That is not enough for a production ML system.

Production data changes. Features drift. Models can become stale. Deployments can fail. A model can perform well during development and become less reliable after deployment.

This project treats ML as a continuous operational lifecycle:

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
Promote
 ↓
Deploy
 ↓
Repeat
```

> **The goal is not merely to train a model. The goal is to operate the model as a production system.**

---

# 2. What I Built

This repository combines:

- Machine learning model development
- Deterministic production feature engineering
- MLflow model lifecycle management
- Drift-aware retraining
- Model evaluation and quality gates
- FastAPI inference serving
- Dockerized production runtime
- Prometheus metrics
- Health and readiness checks
- GitHub Actions CI/CD
- Amazon ECR
- AWS EC2
- AWS API Gateway
- React/Vite frontend
- Vercel production deployment

The result is a live cloud-deployed ML application rather than only a training notebook.

---

# 3. High-Level Architecture

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
                                      │ HTTP
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
                  └────────────────────────────────────┘
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
Pydantic validation
 ↓
Feature builder
 ↓
Production model
 ↓
Prediction
 ↓
Prediction logging
 ↓
Metrics
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
     │      Model Promotion
     │           ↓
     │       Deployment
     │           │
     └───────────┘
          ↓
 Production Model
```

---

# 5. End-to-End Prediction Flow

```text
1. User enters forecasting parameters
              ↓
2. React frontend builds API request
              ↓
3. HTTPS request reaches AWS API Gateway
              ↓
4. API Gateway routes request to EC2
              ↓
5. FastAPI validates input
              ↓
6. Canonical feature vector is constructed
              ↓
7. Production model performs inference
              ↓
8. Prediction is logged
              ↓
9. Request/prediction latency is recorded
              ↓
10. FastAPI returns prediction + model version + request ID
              ↓
11. Frontend renders result
```

---

# 6. Production API

## Prediction

```http
POST /api/predict
```

### Request

```json
{
  "date": "2026-08-25",
  "category": "Electronics",
  "region": "North",
  "price": 100,
  "promo": 1
}
```

### Verified production response

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

Exposes request and prediction telemetry through Prometheus.

---

# 7. Deterministic Inference Contract

The inference service derives canonical feature ordering from:

```text
data/processed/processed_inference.csv
```

The feature builder constructs the inference vector using that schema.

This protects against:

- column-order mismatch
- missing feature columns
- inconsistent categorical encoding
- accidental inference schema changes

The inference schema is therefore treated as a production contract.

---

# 8. Model Lifecycle

The project uses MLflow for model lifecycle management:

```text
Candidate Model
      ↓
Model Evaluation
      ↓
Quality Gate
      ↓
Eligible Model
      ↓
Production Promotion
      ↓
Production Alias / Version
      ↓
Inference Service
```

The serving layer does not select an arbitrary "latest" model.

---

# 9. Serving Strategy

The FastAPI service supports two model sources.

### Packaged model

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

This provides an immutable deployment artifact and avoids development-machine filesystem dependencies.

### MLflow registry mode

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

The deployed configuration currently uses the packaged model strategy because the local MLflow artifact environment contains development filesystem paths that are not suitable for the Linux production container.

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

The safety boundary is:

```text
Train → Evaluate → Quality Gate → Promote → Deploy
```

A newly trained model should not become production merely because it is newer.

---

# 11. CI/CD Architecture

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
┌─────────────────────┐
│ CI — GitHub Actions │
│                     │
│ Install dependencies│
│ Run pytest          │
└──────────┬──────────┘
           │
        PASS?
       /     \
     NO       YES
     │         │
     │         ▼
     │   Build Docker Image
     │         │
     │         ▼
     │   Amazon ECR
     │         │
     │         ▼
     │       EC2
     │         │
     │         ▼
     │   Restart Container
     │         │
     │         ▼
     │    Health Check
     │         │
     │         ▼
     │   Deployment Success
     │
     └── Stop
```

### CI

The CI workflow validates the application with automated tests. The test environment uses:

```text
TEST_MODE=true
```

so tests do not depend on production model initialization.

### CD

The deployment pipeline:

1. Checks out the repository.
2. Configures AWS credentials.
3. Authenticates with Amazon ECR.
4. Builds the Docker image.
5. Tags the image using the Git commit SHA.
6. Pushes the image to ECR.
7. Connects to EC2.
8. Pulls the new image.
9. Replaces the running container.
10. Starts the new container.
11. Waits for startup.
12. Performs health verification.

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

CI/CD
  ▲
  │
┌─┴──────────────┐
│ GitHub Actions │
└───────┬────────┘
        │
        ▼
┌───────────────┐
│   Amazon ECR  │
│ Docker Images │
└───────┬───────┘
        │
        ▼
       EC2
```

---

# 13. Production Evidence

This section intentionally separates **verified evidence** from metrics that still require measured experiment artifacts.

| Evidence | Verified result |
|---|---|
| Frontend | Live on Vercel |
| API Gateway | Live |
| `/health` | HTTP 200 |
| `/ready` | HTTP 200 |
| `/predict` | HTTP 200 |
| Model loaded | `true` |
| Production model version | `7` |
| Browser CORS preflight | HTTP 200 |
| Cross-machine frontend access | Verified |
| Backend runtime | FastAPI on AWS EC2 |
| Container runtime | Docker |
| Registry | Amazon ECR |
| CI | GitHub Actions |
| CD | GitHub Actions |

### Important

Do **not** publish invented latency, accuracy, drift, cost, or business-impact numbers.

A real number backed by an experiment is valuable. A fabricated number damages credibility.

---

# 14. Model Evaluation Results

The system contains model evaluation and quality-gate components. Numerical metrics should be published only when backed by reproducible artifacts.

| Metric | Value | Evidence |
|---|---:|---|
| Production model version | `7` | Production API |
| RMSE | `TBD — measured artifact required` | Evaluation output |
| MAE | `TBD — measured artifact required` | Evaluation output |
| R² | `TBD — measured artifact required` | Evaluation output |
| Validation samples | `TBD` | Evaluation artifact |

This is deliberately conservative. The README should never manufacture model-quality numbers.

---

# 15. Drift / Retraining Evidence

The intended closed loop is:

```text
Production Data
      ↓
Drift Detection
      ↓
Drift Threshold
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

Recommended evidence artifacts:

```text
docs/
├── screenshots/
│   ├── production-ui.png
│   ├── prediction-result.png
│   ├── github-actions-ci.png
│   ├── github-actions-cd.png
│   ├── aws-api-gateway.png
│   └── mlflow-model-version.png
│
└── experiments/
    ├── model-evaluation.json
    ├── drift-results.json
    └── retraining-run.json
```

Only add files that actually exist and contain evidence from this project.

---

# 16. Performance Benchmark

A serious production README should publish measured performance.

Recommended benchmark:

| Metric | Result |
|---|---:|
| Requests | `TBD` |
| Average latency | `TBD ms` |
| P50 latency | `TBD ms` |
| P95 latency | `TBD ms` |
| P99 latency | `TBD ms` |
| Error rate | `TBD %` |
| Successful predictions | `TBD` |

The important engineering principle is:

> **Measure first, publish second.**

---

# 17. Observability

The inference service exposes:

### Request metrics

```text
http_requests_total
```

with:

- method
- endpoint
- HTTP status

### Latency metrics

```text
http_request_latency_seconds
prediction_latency_seconds
```

### Prediction metadata

Predictions can be associated with:

- request ID
- model version
- date
- category
- region
- price
- promotion flag
- prediction

---

# 18. Reliability Design

### Health vs readiness

```text
/health
   ↓
Is the application alive?

/ready
   ↓
Is the model available and ready?
```

### Model initialization

The application loads the production model during FastAPI startup.

### Readiness failure

If the model cannot be loaded:

```text
/ready → 503
```

### Container recovery

The production container is configured to restart automatically after container failure.

### Deployment validation

CI/CD verifies application health after deployment.

---

# 19. Failure Handling

| Failure | Expected behavior |
|---|---|
| Invalid request | Pydantic validation error |
| Model unavailable | `503 Service Unavailable` |
| Prediction failure | `500 Internal Server Error` |
| Application health | `/health` |
| Model readiness | `/ready` |
| Deployment failure | CI/CD reports failure |

---

# 20. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| ML | TensorFlow / ML libraries |
| Data | Pandas, NumPy |
| API | FastAPI |
| Validation | Pydantic |
| Model lifecycle | MLflow |
| Container | Docker |
| CI/CD | GitHub Actions |
| Registry | Amazon ECR |
| Compute | AWS EC2 |
| API Gateway | AWS API Gateway |
| Metrics | Prometheus |
| Frontend | React |
| Frontend build | Vite |
| Frontend hosting | Vercel |
| Source control | Git / GitHub |

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
└── README.md
```

---

# 22. Local Development

## Clone

```bash
git clone https://github.com/talarijayakiran/Self-Healing-ML-System-End-to-End-MLOps-Lifecycle-with-Drift-Aware-Retraining.git
cd Self-Healing-ML-System-End-to-End-MLOps-Lifecycle-with-Drift-Aware-Retraining
```

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run backend

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

---

# 23. Run Frontend Locally

```powershell
cd frontend
npm install
npm run dev
```

The Vite development server URL will be displayed by the terminal.

---

# 24. Docker

```bash
docker build -t self-healing-ml-api .
```

```bash
docker run -d   --name self-healing-ml-api   -p 8000:8000   --restart unless-stopped   self-healing-ml-api
```

Verify:

```bash
curl http://localhost:8000/health
```

---

# 25. Screenshots

Store real screenshots under:

```text
docs/screenshots/
```

Recommended evidence:

```text
production-ui.png
prediction-result.png
github-actions-ci.png
github-actions-cd.png
aws-api-gateway.png
mlflow-model-version.png
drift-retraining.png
```

Then embed them:

```markdown
![Production application](docs/screenshots/production-ui.png)
![Prediction result](docs/screenshots/prediction-result.png)
![CI pipeline](docs/screenshots/github-actions-ci.png)
![CD pipeline](docs/screenshots/github-actions-cd.png)
![MLflow model](docs/screenshots/mlflow-model-version.png)
```

**Do not use generated or fabricated screenshots.**

---

# 26. Engineering Trade-offs

### Why FastAPI?

- typed request validation
- automatic OpenAPI documentation
- Python-native ML integration
- simple production inference serving

### Why Docker?

- reproducible runtime
- dependency isolation
- consistent deployment artifact

### Why MLflow?

- model lifecycle management
- model versioning
- production model selection

### Why EC2?

The project requires direct control over the Docker runtime and deployment environment.

### Why API Gateway?

It provides a managed public HTTP entry point between the frontend and EC2 inference service.

### Why Vercel?

It provides straightforward production hosting for the React/Vite frontend.

---

# 27. Security

Current implementation uses GitHub Actions secrets for sensitive deployment credentials.

Sensitive credentials should never be committed to the repository.

Production hardening opportunities:

- IAM least privilege
- AWS Secrets Manager / Parameter Store
- private EC2 networking
- restricted security groups
- API authentication/authorization
- rate limiting
- centralized logging
- stronger network isolation
- custom production domain

---

# 28. What This Project Demonstrates

### Machine Learning

- model training
- feature engineering
- evaluation
- inference
- model versioning

### MLOps

- model lifecycle management
- drift-aware workflows
- retraining
- quality gates
- promotion
- production serving

### Backend Engineering

- Python
- FastAPI
- REST APIs
- Pydantic
- health/readiness
- error handling

### DevOps

- Docker
- Git
- GitHub Actions
- CI/CD
- container registry
- automated deployment

### Cloud

- AWS EC2
- AWS API Gateway
- Amazon ECR
- production networking

### Observability

- Prometheus
- request metrics
- latency metrics
- prediction logging
- model version tracking

### Frontend

- React
- Vite
- API integration
- production deployment

---

# 29. Recruiter Skill Matrix

| Requirement | Evidence |
|---|---|
| Python | ML + FastAPI |
| Machine Learning | Demand forecasting |
| Feature engineering | Canonical inference builder |
| MLflow | Model lifecycle/versioning |
| MLOps | Drift → retraining → evaluation → promotion |
| FastAPI | Production inference API |
| Docker | Containerized serving |
| GitHub Actions | CI/CD |
| AWS EC2 | Production compute |
| API Gateway | Public API layer |
| ECR | Container registry |
| Prometheus | Metrics |
| React | Production frontend |
| Vite | Frontend build |
| Git/GitHub | Version control |
| REST APIs | Prediction interface |
| Cloud deployment | Live application |
| Production debugging | API/frontend integration |

---

# 30. Limitations

This is a production-oriented portfolio system, not a hyperscale enterprise platform.

Current limitations:

- single primary EC2 inference host
- no multi-region deployment
- no horizontal autoscaling
- no Kubernetes orchestration
- no enterprise identity provider
- no advanced authentication layer
- no formal blue/green deployment
- no distributed tracing platform
- limited infrastructure-as-code coverage

These boundaries are documented intentionally rather than hidden.

---

# 31. Future Production Evolution

```text
CURRENT
  │
  ├── Vercel
  ├── API Gateway
  ├── EC2
  ├── Docker
  └── ECR
        │
        ▼
FUTURE
  │
  ├── ECS / EKS
  ├── Auto Scaling
  ├── Load Balancer
  ├── Private Networking
  ├── Secrets Manager
  ├── Centralized Logging
  ├── Distributed Tracing
  ├── Terraform
  ├── Blue/Green Deployment
  ├── Canary Releases
  ├── Automated Rollback
  └── Multi-AZ / Multi-Region
```

---

# 32. Why This Project Matters

A production ML system must answer:

```text
Can I serve it?
Can I monitor it?
Can I detect drift?
Can I retrain it?
Can I validate the replacement?
Can I promote it safely?
Can I deploy it automatically?
Can I recover when something fails?
Can I prove which model served a prediction?
```

This project is built around those questions.

> **The model is only one component. The production system around the model is the engineering problem.**

---

# 33. Production Status

| Component | Status |
|---|---|
| ML model | ✅ |
| Feature engineering | ✅ |
| MLflow lifecycle | ✅ |
| Drift/retraining workflow | ✅ |
| FastAPI inference | ✅ |
| Health endpoint | ✅ |
| Readiness endpoint | ✅ |
| Prometheus metrics | ✅ |
| Docker | ✅ |
| CI | ✅ |
| CD | ✅ |
| Amazon ECR | ✅ |
| AWS EC2 | ✅ |
| AWS API Gateway | ✅ |
| React frontend | ✅ |
| Vercel deployment | ✅ |
| Cross-machine access | ✅ |
| Live prediction | ✅ |

---

# 34. Final Architecture

```text
                             ┌─────────────────────┐
                             │        USER         │
                             └──────────┬──────────┘
                                        │
                                        ▼
                             ┌─────────────────────┐
                             │       VERCEL        │
                             │   React + Vite      │
                             └──────────┬──────────┘
                                        │ HTTPS
                                        ▼
                             ┌─────────────────────┐
                             │   AWS API GATEWAY   │
                             │      HTTP API       │
                             └──────────┬──────────┘
                                        │
                                        ▼
                 ┌────────────────────────────────────────┐
                 │                 AWS EC2                 │
                 │                                        │
                 │             Docker Container           │
                 │                                        │
                 │   FastAPI                              │
                 │      │                                 │
                 │      ├── Validation                    │
                 │      ├── Feature Builder               │
                 │      ├── Production Model              │
                 │      ├── Prediction Logger              │
                 │      └── Prometheus Metrics             │
                 └──────────────────┬─────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Monitoring / Drift  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Retraining      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Evaluation / Gate   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ MLflow / Promotion │
                         └──────────┬──────────┘
                                    │
                                    ▼
                              New Model
                                    │
                                    └──► Production


CI/CD CONTROL PLANE

Developer
   ↓
GitHub
   ↓
GitHub Actions
   ├── CI → Tests
   └── CD → Docker Build
              ↓
           Amazon ECR
              ↓
           AWS EC2
              ↓
        Health Verification
```

---

# 35. Author

## Talari Jayakiran

**AI Engineer | MLOps Engineer | Machine Learning Engineer**

Focus:

- Production ML systems
- MLOps
- AI Engineering
- Cloud deployment
- ML inference systems
- Generative AI / LLMOps

---

## Final Principle

> **If a model cannot be validated, deployed, monitored, recovered, and continuously improved, the ML lifecycle is incomplete.**
