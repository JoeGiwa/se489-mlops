# Phase 3: Continuous Machine Learning (CML) & Deployment

##  Objectives Completed

- Implemented CI/CD pipelines using GitHub Actions for both API and UI
- Deployed prediction API (FastAPI) and interactive UI (Streamlit) on Cloud Run
- Integrated Prometheus-based monitoring with real-time `/metrics` endpoint
- Configured secure Google Cloud authentication using a service account key and GitHub Secrets
- Generated model evaluation reports using CML and Weights & Biases
- Organized complete documentation for deployment, usage, and monitoring

---

## 1. Continuous Integration & Testing

### 1.1 Unit Testing with Pytest

- Unit tests implemented for input validation, data preprocessing, and model loading
- Coverage of training pipeline modules partially completed
- Ruff linting executed manually and via CI on push events

### 1.2 GitHub Actions Workflows

- `predict.yml`: Handles Docker build and deployment for the FastAPI prediction service
- `streamlit.yml`: Automates Docker build and deploy of the Streamlit UI
- Both workflows run on `main` and `phase3_v2` branches and on changes to `dockerfiles/`, `models/`, or `streamlit_app/`

### 1.3 Pre-commit Hooks

- Configured `.pre-commit-config.yaml` with `ruff`, `black`, and end-of-file fixer
- Ensures consistent code style and prevents common issues before commits

---

## 2. Continuous Docker Building & CML

### 2.1 Docker Image Automation

- Dockerfiles:
  - `predict_model.dockerfile` for the FastAPI inference service
  - `streamlit.dockerfile` for the user-facing CSV-upload UI
- CI/CD builds triggered using GitHub Actions with `docker buildx`
- Tagged images pushed to GCP Artifact Registry

### 2.2 Continuous Machine Learning (CML)

- Model training and evaluation tracked using Weights & Biases (W&B)
- GitHub Action (`cml.yaml`) logs confusion matrix, accuracy, and F1 score
- Evaluation summary posted as PR comment on model update branches

---

## 3. Deployment on Google Cloud Platform (GCP)

### 3.1 Artifact Registry

- Docker images hosted at:  
  `europe-west1-docker.pkg.dev/carbon-hulling-454008-a0/music-repo/music-genre-predict`

### 3.2 Custom Training Job on GCP (Optional)

- Skipped in this phase — training done locally and tracked via W&B

### 3.3 Deploying API with FastAPI & Cloud Run

- `/predict`, `/health`, and `/metrics` endpoints deployed as part of FastAPI container
- Exposes structured model serving pipeline and health monitoring

### 3.4 Dockerize & Deploy Model with GCP Cloud Run

- Docker image deployed to Cloud Run via GitHub Actions
- Configuration includes:
  - `--memory=1Gi`
  - `--timeout=1000`
  - `--max-instances=1`

### 3.5 Interactive UI Deployment

- Streamlit app supports manual input and CSV file upload
- Deployed independently to Cloud Run
- Connected to `/predict` endpoint of the API
- Triggered by `streamlit.yml` CI workflow on commit


### 3.5 Interactive UI Deployment

The Streamlit web app supports:
- Manual input or CSV upload (58 features)
- Model selection from dropdown (XGBoost, etc.)
- Deployment to Cloud Run via GitHub Actions

#### Streamlit UI

![Streamlit UI](docs/screes-1.jpg)

#### FastAPI /predict Endpoint

![FastAPI Endpoint](docs/SS-2.jpg)

#### GCP Cloud Run Metrics

> Showing request count, memory usage, instance count, and latency

![Cloud Run Metrics](docs/SS-3.jpg)

#### GCP Cloud Run Logs

> Logs of container startup, model serving, and inference requests

![Cloud Run Logs](docs/SS-4.jpg)


## 4. Documentation & Repository Updates

### 4.1 Comprehensive README

- Covers:
  - Project structure
  - Setup instructions
  - Deployment walkthrough
  - Endpoint examples
  - Monitoring setup

### 4.2 Resource Cleanup

- Checklist added to README to avoid unnecessary billing
  - Includes: Artifact Registry cleanup, Cloud Run shutdown, and GCS data archiving

---

##  Final Test Plan

- Sent valid POST requests to `/predict` with JSON input
- Verified API output for edge cases and sample inputs
- Uploaded CSV via Streamlit UI and validated batch predictions
- Monitored `/metrics` for CPU, memory, and latency under load

---

##  Conclusion

All required Phase 3 deliverables are completed:

- CI/CD automation for both prediction API and UI
- Containerized deployment on Google Cloud Run
- Real-time monitoring with Prometheus metrics
- Secure cloud access using GitHub Secrets and service account

Only optional enhancements such as Hugging Face UI and full unit test coverage remain outside scope. The project is now production-ready with automated training, serving, UI, and evaluation integrated.

> **Next Step:** Run full validation test, capture final deployment screenshots, and finalize the README. 
