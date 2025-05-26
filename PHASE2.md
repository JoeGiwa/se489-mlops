
# PHASE 02: Enhancing ML Operations with Containerization & Monitoring

## 1. Containerization

- Dockerfiles created for both training train_model.dockerfile and prediction predict_model.dockerfile.
- Each container includes environment setup via requirements.txt and pyproject.toml.

**Containers tested using:**

`docker build -f dockerfiles/train_model.dockerfile -t mlops-train .`

`docker build -f dockerfiles/predict_model.dockerfile -t mlops-predict .`

`docker run -v $(pwd)/artifacts:/app/artifacts mlops-train`

## 2. Monitoring & Debugging

- Runtime system resource monitoring (CPU/RAM) integrated via custom logging thread.
- 
  Example: `[MONITOR] CPU usage: 32.5%, Memory usage: 74.3%`
  
- Debugging tools used include pdb and rich.logging for enhanced tracebacks and log clarity.

## 3. Profiling & Optimization

- Implemented performance profiling with `cProfile`:
  
  `python profiling/profile_train.py model=cnn`
  
  `python -m pstats profiling/train_profile.prof`

- `.prof` results help identify bottlenecks in training/inference.

## 4. Experiment Management & Tracking

- Integrated Weights & Biases (W&B) for:
    - Tracking training metrics (loss, accuracy, epochs)
    - Monitoring inference stats and durations
    - W&B initialized via:
      
      `wandb login`

## 5. Logging

- Logging set up across training and prediction using:
    - Python’s logging module
    - `rich` integration for readable outputs
      
- Logs include:
    - Data loading status
    - Model predictions
    - W&B sync and system diagnostics

## 6. Configuration Management

- Project configurations managed with Hydra under `conf/.`
    - Allows modular, command-line configurable runs:
      
      `python mlops_randproject/models/train_model.py model=xgboost train.epochs=5`

## 7. CI/CD & Documentation

- GitHub Actions configured to:
    - Auto-build and push Docker images to Docker Hub
    - Use DOCKERHUB_USERNAME and DOCKERHUB_TOKEN secrets
      
- README.md and Phase 2 documentation updated to reflect:
    - Docker usage
    - Logging/monitoring setup
    - Model training & prediction instructions


**PHASE 2 Outcome:**
 All of the requirements have been met. Our project now uses Docker for reproducibility, MLflow for experiment tracking, and structured logging to support monitoring and debugging. Hydra and config files support modular experimentation.
