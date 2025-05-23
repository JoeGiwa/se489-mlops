# model_training.py

import os
import hydra
import numpy as np
import logging
from omegaconf import OmegaConf, DictConfig
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from mlops_randproject.model_zoo import build_cnn, build_mlp, build_xgboost
import joblib

from mlops_randproject.data.data_split import load_and_split_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def build_model_2(input_shape, cfg):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(cfg.model.hidden1, activation='relu', kernel_regularizer=l2(cfg.model.l2)),
        BatchNormalization(),
        Dropout(cfg.model.dropout),
        Dense(cfg.model.hidden2, activation='relu', kernel_regularizer=l2(cfg.model.l2)),
        BatchNormalization(),
        Dropout(cfg.model.dropout),
        Dense(cfg.model.output, activation='softmax')
    ])
    model.compile(optimizer=cfg.model.optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    logger.info(f" Starting training for model: {cfg.model.name.upper()}")
    X1_train_scaled, X1_test_scaled, y1_train, y1_test, label_encoder, scaler = load_and_split_data(version="30_sec")
    input_shape = X1_train_scaled.shape[1]
    
    if cfg.model.name == "cnn":
        X1_train_scaled = X1_train_scaled.reshape(-1, X1_train_scaled.shape[1], 1, 1)
        X1_test_scaled = X1_test_scaled.reshape(-1, X1_test_scaled.shape[1], 1, 1)
        model = build_cnn(cfg.model, input_shape=(X1_train_scaled.shape[1], 1, 1))
    elif cfg.model.name == "mlp":
        model = build_mlp(cfg.model, input_shape=input_shape)
    elif cfg.model.name == "xgboost":
        model = build_xgboost(cfg.model)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")

    if cfg.model.name in ["cnn", "mlp"]:
        early_stop = EarlyStopping(
            monitor=cfg.train.early_stopping.monitor,
            patience=cfg.train.early_stopping.patience,
            restore_best_weights=cfg.train.early_stopping.restore_best_weights,
)

        reduce_lr = ReduceLROnPlateau(
            monitor=cfg.train.reduce_lr_on_plateau.monitor,
            factor=cfg.train.reduce_lr_on_plateau.factor,
            patience=cfg.train.reduce_lr_on_plateau.patience,
            min_lr=cfg.train.reduce_lr_on_plateau.min_lr,
)
        logger.info(" Training model...")
        history = model.fit(
            X1_train_scaled, y1_train,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            validation_split=cfg.train.validation_split,
            callbacks=[early_stop, reduce_lr]
    )
    

        # Save training artifacts
        artifact_dir = "artifacts"
        os.makedirs("artifacts", exist_ok=True)
        model.save_weights(os.path.join(artifact_dir, "model.weights.h5"))
        joblib.dump(history.history, os.path.join(artifact_dir, "history.pkl"))
        logger.info(" Model training complete and weights saved.")

    else:
        # Train and save XGBoost model
        logger.info(" Training XGBoost model...")
        os.makedirs(artifact_dir, exist_ok=True)
        model.fit(X1_train_scaled, y1_train)
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(model, os.path.join(artifact_dir, "xgboost_model.pkl"))
        joblib.dump(label_encoder, os.path.join(artifact_dir, "label_encoder.pkl"))
        joblib.dump(scaler, os.path.join(artifact_dir, "scaler.pkl"))
        logger.info(" XGBoost model and preprocessors saved.")


    # Save config used
    with open("artifacts/used_config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info("Config saved with run artifacts.")

    np.save("artifacts/test_features.npy", X1_test_scaled)
    logger.info("Training complete!")
    
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(" Error occurred during model training.")
        raise
