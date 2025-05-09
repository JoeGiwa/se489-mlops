# Music Genre Classification using Machine Learning and MLOps

## 1. Team Information
Team Name: Team Random
Team Members: Lajja Desai (ldesai2@depaul.edu)
              Mitanshi Kapadiya (mkapadiy@depaul.edu)
              Joseph Giwa (jgiwa@depaul.edu)
Course & Section: SE489 Machine Learning Engineering for Production              

## 2. Project Overview
Summary:  
  This project aims to build an automated music genre classification pipeline using machine learning and deep learning techniques. The system classifies audio tracks into one of ten genres from the GTZAN dataset.

Problem Statement & Motivation:  
  With the exponential rise of digital music, manual organization of music libraries is inefficient. Automating genre classification enhances user experience and recommendation systems in platforms like Spotify, Apple Music, and YouTube Music.

Main Objectives:  
  - Extract and analyze audio features using Librosa  
  - Train and evaluate both traditional ML and deep learning models (DNN, CNN)  
  - Incorporate MLOps best practices (DVC, Hydra, code linting, reproducibility)  
  - Compare model performance and explore deployment options

## 3. Project Architecture Diagram

![alt text](image-1.png)

## 4. Phase Deliverables
PHASE1.md(./PHASE1.md): Project Design & Initial Model Development

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
Lajja Desai: Worked on data preprocessing and exploring data analyis, set up training pipeline and callbacks.
Mitanshi Kapadiya: Implemented DNN architecture in model training, handled cross-validation in model_performance.
Joseph Giwa: Set up the Github repository structure and environment, integrated Hydra, and initialized DVC datset tracking.

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
