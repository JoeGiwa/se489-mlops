# Music Genre Classification using Machine Learning and MLOps

## 1. Team Information

* **Team Name:** Team Random
* **Team Members:**

  * Lajja Desai ([ldesai2@depaul.edu](mailto:ldesai2@depaul.edu))
  * Mitanshi Kapadiya ([mkapadiy@depaul.edu](mailto:mkapadiy@depaul.edu))
  * Joseph Giwa ([jgiwa@depaul.edu](mailto:jgiwa@depaul.edu))
* **Course & Section:** SE489 Machine Learning Engineering for Production

---

## 2. Project Overview

### Summary

This project implements an automated music genre classification pipeline using traditional machine learning and deep learning models. It classifies audio tracks from the GTZAN dataset into 10 distinct genres, integrated with full MLOps practices for scalable deployment.

### Problem Statement & Motivation

Digital music consumption has surged, making manual curation inefficient. Our goal is to automate genre classification to improve recommendation systems on platforms like Spotify and Apple Music.

### Main Objectives

* Feature extraction using Librosa
* Training and evaluation using DNN, CNN, and XGBoost
* Full MLOps pipeline: DVC, CI/CD, cloud deployment
* Serving models with API and Streamlit UI

---

## 3. Architecture Diagram

![alt text](https://github.com/JoeGiwa/se489-mlops/blob/main/img_1.jpeg)

---

## 4. Phase Deliverables

* [PHASE1.md](https://github.com/JoeGiwa/se489-mlops/blob/main/PHASE1.md): Project Design & Initial Model Development
* [PHASE2.md](https://github.com/JoeGiwa/se489-mlops/blob/main/PHASE2.md): Containerization & Monitoring
* [PHASE3.md](https://github.com/JoeGiwa/se489-mlops/blob/main/PHASE3.md): Continuous Machine Learning & Deployment

---

## 5. Setup Instructions

### Environment Setup

```bash
conda create -n se489-mlops python=3.9 -y
conda activate se489-mlops
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/JoeGiwa/se489-mlops.git
cd se489-mlops
```

### Configure Google Cloud

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/dvc.json"
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
gcloud config set project carbon-hulling-454008-a0
```

---

## 6. Running the Code

### Model Training

```bash
python mlops_randproject/model_training.py model.name=cnn
```

### Model Evaluation

```bash
python mlops_randproject/evaluate.py model.name=cnn
```

### FastAPI (Prediction API)

```bash
uvicorn mlops_randproject.api.predict_api:app --reload
```

### Streamlit UI

```bash
cd streamlit
streamlit run app.py
```

---

## 7. CI/CD & Deployment (Phase 3)

### CI/CD Overview

* GitHub Actions: `predict.yml` and `streamlit.yml`
* Triggers: main & phase3\_v2 branches
* Secrets: Base64-encoded GCP service account credentials

### Docker Build & Push

```bash
docker buildx build \
  --platform linux/amd64 \
  -f dockerfiles/predict_model.dockerfile \
  -t music-genre-predict:latest \
  --load .

docker tag music-genre-predict:latest \
  europe-west1-docker.pkg.dev/carbon-hulling-454008-a0/music-repo/music-genre-predict

docker push \
  europe-west1-docker.pkg.dev/carbon-hulling-454008-a0/music-repo/music-genre-predict
```

### Cloud Run Deployment (FastAPI)

```bash
gcloud run deploy music-genre-service \
  --image=europe-west1-docker.pkg.dev/carbon-hulling-454008-a0/music-repo/music-genre-predict \
  --platform=managed \
  --region=europe-west1 \
  --allow-unauthenticated \
  --memory=1Gi \
  --timeout=1000 \
  --max-instances=1
```

### Endpoints

* `/predict` – genre prediction (POST)
* `/health` – API health check (GET)
* `/metrics` – Prometheus metrics (GET)

---

## 8. Monitoring, Profiling & Tracking

### Monitoring

* Prometheus-style metrics exposed via `/metrics`
* Tracks latency, memory, CPU, and GC stats

### Profiling

* Scripts under `profiling/` directory using `cProfile`

### Experiment Tracking

* Weights & Biases (W\&B): automatic logging of training metrics

### Logging

* Python’s `logging` + `rich` for formatted system and data logs

### Configuration Management

* Hydra for flexible CLI-driven config control
* Configs stored under `conf/`

---

## 9. Contribution Summary

* **Lajja Desai:** Data preprocessing, analysis, training pipeline setup, unit tests
* **Mitanshi Kapadiya:** DNN implementation, model evaluation, CI workflows, Documentations
* **Joseph Giwa:** Hydra configs, DVC integration, monitoring, GCP deployment

---

## 10. References

### Dataset

[GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

### Libraries & Tools

* TensorFlow/Keras
* Scikit-learn
* Librosa
* Matplotlib/Seaborn
* Hydra
* DVC
* Ruff
* MyPy
* FastAPI, Streamlit
* Prometheus
* Weights & Biases (W\&B)

---

## 11. Cleanup Checklist

```bash
gcloud run services delete music-genre-service
gcloud run services delete streamlit-ui
gcloud artifacts repositories delete music-repo --location=europe-west1
```

---

## 12. Final Notes

This project demonstrates a complete MLOps lifecycle: from training to deployment and monitoring. It is CI/CD-ready, cloud-scalable, and supports continuous evaluation. Future work may include model drift detection, Hugging Face hosting, and retraining pipelines.

---
**Authors**: SE-489 Team

**License**: MIT
