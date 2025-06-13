from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List
import numpy as np
from functools import lru_cache
import joblib
import os
from enum import Enum
from omegaconf import OmegaConf
from mlops_randproject.model_zoo import build_mlp, build_cnn
import logging

model_cache = {}

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
ARTIFACT_DIR = "artifacts"
NUM_FEATURES = 58

# --- FastAPI Init ---
app = FastAPI()


# --- Enums ---
class ModelType(str, Enum):
    xgboost = "xgboost"
    mlp = "mlp"
    cnn = "cnn"


# --- Request Schema ---
class PredictRequest(BaseModel):
    features: List[float]
    model_name: ModelType

    @field_validator("features")
    def validate_length(cls, v):
        if len(v) != 58:
            raise ValueError("Feature vector must have exactly 58 elements.")
        return v


# --- Load Common Artifacts ---
@lru_cache(maxsize=1)
def load_common_artifacts():
    label_encoder = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    return label_encoder, scaler


# --- Load Model ---
def load_model(model_name: ModelType):
    if model_name in model_cache:
        return model_cache[model_name]

    _, scaler = load_common_artifacts()

    if model_name == ModelType.xgboost:
        model = joblib.load(os.path.join(ARTIFACT_DIR, "xgboost_model.pkl"))

    elif model_name in [ModelType.mlp, ModelType.cnn]:
        cfg_path = os.path.join(ARTIFACT_DIR, "used_config.yaml")
        cfg = OmegaConf.load(cfg_path)

        dummy_input = np.zeros(NUM_FEATURES).reshape(1, -1)
        input_shape = scaler.transform(dummy_input).shape[1]

        if model_name == ModelType.mlp:
            model = build_mlp(cfg.model, input_shape=input_shape)
        else:
            model = build_cnn(cfg.model, input_shape=input_shape)

        model.load_weights(os.path.join(ARTIFACT_DIR, "model.weights.h5"))

    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model_cache[model_name] = model
    return model


# --- Routes ---
@app.get("/")
def read_root():
    return {"message": "ML Inference API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "models_supported": [e.value for e in ModelType],
        "input_features": NUM_FEATURES,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        label_encoder, scaler = load_common_artifacts()
        features = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        model = load_model(request.model_name)

        if request.model_name == ModelType.xgboost:
            preds = model.predict(features_scaled)
        elif request.model_name == ModelType.cnn:
            features_scaled = features_scaled.reshape(
                features_scaled.shape[0], features_scaled.shape[1], 1
            )
            preds = model.predict(features_scaled)
            preds = np.argmax(preds, axis=1)
        elif request.model_name == ModelType.mlp:
            preds = model.predict(features_scaled)
            preds = np.argmax(preds, axis=1)
        else:
            raise HTTPException(status_code=400, detail="Unsupported model")

        class_name = label_encoder.inverse_transform(preds)[0]
        logger.info(f"Model: {request.model_name}, Prediction: {class_name}")
        return {"prediction": class_name}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# @app.get("/")
# def read_root():
#     return {"message": "ML Inference API is running!"}
