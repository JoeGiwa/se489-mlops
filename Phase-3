# Phase 3: CI/CD and Deployment


## Objectives completed


- Implemented CI/CD pipelines using GitHub Actions

- Deployed prediction API and Streamlit UI on Cloud Run

- Integrated Prometheus monitoring middleware

- Configured secure GCP authentication via service account


## CI/CD Workflow


### `predict.yml`

- Builds and pushes Docker image of `predict_api`

- Deploys to Cloud Run with platform `linux/amd64`


### `streamlit.yml`

- Builds and deploys Streamlit app

- Triggers on file changes or push to `phase3_v2`/`main`


## Secrets & Authentication


- Used GitHub Secrets for secure GCP credentials

- Service account key in one-line JSON


## Monitoring


- Integrated Prometheus middleware

- Exposed `/metrics` with CPU, memory, GC, and latency stats


## Final Test Plan


- Send valid requests to `/predict`

- Upload CSV through Streamlit

- Validate output consistency

- Confirm metrics reflect incoming traffic


Phase 3: Continuous Machine Learning (CML) & Deployment

1. Continuous Integration & Testing

1.1 Unit Testing with pytest

· Basic unit tests implemented for model loading and input validation.

· Full test coverage of data processing and training modules is pending.

· Ruff linting checks run manually and in GitHub Actions.

1.2 GitHub Actions Workflows

· predict.yml and streamlit.yml implemented for Docker CI/CD.

· GitHub Actions run on changes to key folders and branches (main, phase3_v2).

1.3 Pre-commit Hooks

· Pre-commit hooks configured with ruff and black.

· .pre-commit-config.yaml file included.


2. Continuous Docker Building & CML

2.1 Docker Image Automation

· predict.dockerfile and streamlit.dockerfile created.

· Automated Docker builds via GitHub Actions using buildx.

· Images pushed to GCP Artifact Registry.

2.2 Continuous Machine Learning (CML)

· Model training results logged using Weights & Biases (W&B).

· CML report generated with confusion matrix, F1 score, accuracy.

· cml.yaml GitHub Actions workflow outputs evaluation report on PRs.


3. Deployment on Google Cloud Platform (GCP)

3.1 GCP Artifact Registry

· Docker images stored at: us-central1-docker.pkg.dev/utility-cumulus-462615-h4/mlops-repo/
  
  3.2 Custom Training Job on GCP

· submit_vertex_jobs.py script submits training jobs for XGBoost, MLP, and CNN to Vertex AI.

· Artifacts saved to GCS bucket.

3.3 Deploying API with FastAPI & GCP Cloud Functions

· FastAPI deployed as container to Cloud Run (instead of Cloud Functions).

· /predict, /health, and /metrics endpoints available.

3.4 Dockerize & Deploy Model with GCP Cloud Run

· Docker container deployed using GitHub Actions.

· Accessible public API and logs validated.

3.5 Interactive UI Deployment

· Streamlit app created with CSV upload support.

· UI deployed to Cloud Run.

· CI/CD pipeline integrated via streamlit.yml.

· Manual input and CSV tested successfully.


4. Documentation & Repository Updates

4.1 Comprehensive README

· README includes structure, usage, deployment, and monitoring.

· CI/CD workflows and endpoints are documented.

· Prometheus /metrics endpoint described.

4.2 Resource Cleanup Reminder

· GCP cleanup checklist included in README.md to prevent cost overrun.


Conclusion

All required Phase 3 deliverables are implemented, including:
  
  · CI/CD automation for training and deployment

· Model serving and Streamlit UI

· Monitoring via Prometheus

· Google Cloud Platform integration

Only optional enhancements like full unit test coverage and Hugging Face UI were omitted.

Next step: run end-to-end validation to capture screenshots and finalize README.
