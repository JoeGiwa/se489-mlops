# Music Genre Classification using Machine Learning and MLOps
## 1. Team Information
- Team Name: Team Random
- Team Members: 
  - Lajja Desai (ldesai2@depaul.edu)
  - Mitanshi Kapadiya (mkapadiy@depaul.edu)
  - Joseph Giwa (jgiwa@depaul.edu)
- Course & Section: SE489 Machine Learning Engineering for Production              

## 2. Project Overview
## Summary:  
  This project aims to build an automated music genre classification pipeline using machine learning and deep learning techniques. The system classifies audio tracks into one of ten genres from the GTZAN dataset.

Problem Statement & Motivation:  
  With the exponential rise of digital music, manual organization of music libraries is inefficient. Automating genre classification enhances user experience and recommendation systems in platforms like Spotify, Apple Music, and YouTube Music.

## Main Objectives:  
  - Extract and analyze audio features using Librosa  
  - Train and evaluate both traditional ML and deep learning models (DNN, CNN)  
  - Incorporate MLOps best practices (DVC, Hydra, code linting, reproducibility)  
  - Compare model performance and explore deployment options

## 3. Project Architecture Diagram

![alt text](https://github.com/JoeGiwa/se489-mlops/blob/main/img_1.jpeg)

## 4. Phase Deliverables

[PHASE1.md](https://github.com/JoeGiwa/se489-mlops/blob/main/PHASE1.md): Project Design & Initial Model Development

[PHASE2.md](https://github.com/JoeGiwa/se489-mlops/blob/main/PHASE2.md): Enhancing ML Operations with Containerization & Monitoring

## 5. Setup Instructions

# Environment Setup
Using venv (recommended)

1.⁠ ⁠Clone the repository
git clone https://github.com/JoeGiwa/se489-mlops.git
cd se489-mlops

2.⁠ ⁠Create a virtual environment
python -m venv env
source env/bin/activate      # On Windows: env\Scripts\activate

3.⁠ ⁠Install dependencies
pip install -r requirements.txt

# Monitoring, Profiling & Tracking:

Monitoring:
System resource usage (CPU and memory) is logged during execution using a background monitoring thread.

Profiling:
Profiling scripts under the profiling/ directory generate .prof files for performance analysis using cProfile.

Tracking:
Training and prediction metrics are automatically logged to Weights & Biases (W&B). You can view experiment dashboards on the W&B platform.

Logging:
Logging is handled through Python’s built-in logging module, enhanced with rich for visually formatted outputs. Logs include system metrics, data flow steps, and W&B sync status.

Configuration Management:
Hydra is used for managing experiment configurations. All config files are located in the conf/ directory and support flexible CLI overrides for model selection and hyperparameter tuning.

# Running the Code
# Train the Model

python model_training.py

# Evaluate Model Performance & Visualize

python model_performance.py

# Data Versioning with DVC

dvc pull

# Linting & Type Checking

ruff check .
mypy src/

## 6. Contribution Summary
- Lajja Desai: Worked on data preprocessing and exploring data analyis, set up training pipeline and callbacks.
- Mitanshi Kapadiya: Implemented DNN architecture in model training, handled cross-validation in model_performance.
- Joseph Giwa: Set up the Github repository structure and environment, integrated Hydra, and initialized DVC datset tracking.

## 7. References
•⁠  ⁠GTZAN Genre Collection: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

A benchmark dataset consisting of 1,000 audio tracks across 10 music genres, used for genre classification research.

•⁠  ⁠Frameworks & Libraries

TensorFlow/Keras – Deep learning framework used for building and training DNN models

Scikit-learn – Traditional machine learning and evaluation metrics

Matplotlib & Seaborn – Data visualization libraries

Hydra – Configurable experiment management system

DVC (Data Version Control) – Tracks datasets and model artifacts alongside Git

Ruff – Fast Python linter

MyPy – Static type checker for Python
