
# PHASE 1: Project Design & Model Development

This document presents the system planning, design choices, data engineering workflow, and model development activities completed in Phase 1 of the Music Genre Classification project, conducted as part of SE489 — Machine Learning Engineering for Production. This phase establishes a reproducible and modular ML pipeline foundation, incorporating modern MLOps tools.

---


## 1. Project Proposal

### 1.1 Project Scope and Objectives

**Problem Statement:**  
Music genre classification enables intelligent music indexing, recommendations, and playlist generation. Given the complexity of musical features and genre overlap, building a robust classifier for music genres is a challenging and impactful task.

**Objective:**  
- Build a scalable ML pipeline to classify music into 10 genres.
- Compare traditional ML models with deep learning approaches.
- Apply MLOps practices for versioning, automation, reproducibility, and modularity.
- Automate training and evaluation workflows using tools like DVC, Hydra, and Make.

**Success Metrics:**  
Traditional ML (e.g., SVM, RF): ≥ 80% accuracy on feature vectors.
CNN-based classifier: ≥ 85% accuracy on mel-spectrogram images.
Fully reproducible pipeline using Git, DVC, and Docker.

**Tools:**  
- Feature extraction: Librosa  
- Modeling: Scikit-learn, Keras/TensorFlow  
- Experiment tracking: MLflow  
- Config management: Hydra  
- Data tracking: DVC  
- Code quality: ruff, mypy  
- Project structure: Cookiecutter  
- Automation: Makefile

---

### 1.2 Selection of Data

- Dataset: [GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- 1,000 audio tracks (30s each) across 10 genres
- Format: WAV, mono, 22.05 kHz
- Preprocessing includes:
  - MFCC/chroma/tonnetz features
  - Mel-spectrogram conversion for CNN input
- DVC will be used to version the raw and preprocessed data

---

### 1.3 Model Considerations

- Traditional Models (baseline):
  - Logistic Regression
  - SVM
  - Random Forest
- Deep Learning Models:
  - CNN trained on mel-spectrogram images
  - Layers: Conv2D → BatchNorm → MaxPool → Dropout → Dense → Softmax
  - Regularization: Early stopping, learning rate scheduling, dropout

---

### 1.4 Open-source Tools

| Tool         | Purpose                            |
|--------------|------------------------------------|
| Librosa      | Audio feature extraction (MFCCs, etc.) |
| TensorFlow   | Deep learning framework for CNNs   |
| Scikit-learn | Baseline models, metrics           |
| DVC          | Dataset versioning and pipeline tracking |
| Hydra        | Configuration management            |
| MLflow       | Experiment tracking                 |
| ruff / mypy  | Code linting and type checking      |
| Cookiecutter | Project structure scaffold          |

---

## 2. Code Organization & Setup

### 2.1 Repository Setup

- Structured using [Cookiecutter Data Science template]
- Source files in flat layout 
- Git + GitHub used for version control
- DVC initialized and `.dvc` files tracked in Git

### 2.2 Environment Setup

- Python 3.11 with virtual environment
- Generated and tracked requirements.txt 
- Docker setup planned for training and inference environments
- Colab used for GPU-based training/testing

---

## 3. Version Control & Team Collaboration

### 3.1 Git Usage

- Git for code versioning with feature branches
- Remote collaboration via GitHub

### 3.2 DVC Setup

- DVC initialized
- `dvc add` for raw and processed data
- `.dvc` files committed to Git
- Remote storage (GDrive) configured

### 3.3 Team Collaboration
- Lajja Desai: Data exploration, traditional model development
- Mitanshi Kapadiya: CNN modeling and performance tuning
- Joseph Siwa: Environment setup, DVC integration, config automation

---

## 4. Data Handling

### 4.1 Data Preparation

- Handled by data_split.py and extract_features.py
- Extracted MFCCs, ZCR, chroma, spectral features
- Data stored in NumPy arrays and CSV for training

### 4.2 Data Documentation

- Data prep steps saved in notebooks and scripts
- Documented and versioned with DVC

---

## 5. Model Training

### 5.1 Training Infrastructure

- Initial training in Jupyter or Colab
- CNN training in Python scripts (with maybe GPU)

### 5.2 Initial Training & Evaluation

- Baseline models: SVM, RF
- Evaluation metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
- Deep learning: CNN on spectrograms, validated using train/val split

---

## 6. Documentation & Reporting

### 6.1 Project README

- Clear sections: Objectives, Dataset, Setup, Results, Contribution

### 6.2 Code Documentation

- Docstrings and type hints
- Style checked with `ruff`
- Type checked with `mypy`
- Automations defined in `Makefile`

---

### Phase 1 Outcome:
- Phase 1 delivers a robust genre classification pipeline with reproducible setup, modular design, and baseline models. With configurations tracked via Hydra, data and models versioned with DVC, and automated evaluation steps in place, the project is well-positioned for Phase 2 advancements including experiment tracking, deployment, and CI/CD.

