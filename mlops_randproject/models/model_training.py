# model_training.py
import os
import hydra
import time
import wandb
import joblib
import logging
import numpy as np
import subprocess
from datetime import datetime  # Add this import at the top with others
from rich.console import Console
from rich.logging import RichHandler
from omegaconf import OmegaConf, DictConfig
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from mlops_randproject.model_zoo import build_cnn, build_mlp, build_xgboost
from mlops_randproject.data.data_split import load_and_split_data
from mlops_randproject.utils.monitor import SystemMonitor

OmegaConf.register_new_resolver("env", lambda key, default=".": os.getenv(key, default))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("mlops")
console = Console()


class WandbLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            wandb.log(logs)


# def build_model_2(input_shape, cfg):
#     model = Sequential([
#         Input(shape=(input_shape,)),
#         Dense(cfg.model.hidden1, activation='relu', kernel_regularizer=l2(cfg.model.l2)),
#         BatchNormalization(),
#         Dropout(cfg.model.dropout),
#         Dense(cfg.model.hidden2, activation='relu', kernel_regularizer=l2(cfg.model.l2)),
#         BatchNormalization(),
#         Dropout(cfg.model.dropout),
#         Dense(cfg.model.output, activation='softmax')
#     ])
#     model.compile(optimizer=cfg.model.optimizer,
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    with SystemMonitor(interval=5):
        console.rule("[bold blue]Model Training Started")
        console.log("Using configuration:", cfg)

        logger.info(f" Starting training for model: {cfg.model.name.upper()}")
        X1_train_scaled, X1_test_scaled, y1_train, y1_test, label_encoder, scaler = (
            load_and_split_data(version="30_sec")
        )
        input_shape = X1_train_scaled.shape[1]

        run_name = f"{cfg.model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="mlops-randproject-train",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
        )

        if cfg.model.name == "cnn":
            X1_train_scaled = X1_train_scaled.reshape(-1, X1_train_scaled.shape[1], 1)
            X1_test_scaled = X1_test_scaled.reshape(-1, X1_test_scaled.shape[1], 1)
            model = build_cnn(cfg.model, input_shape=X1_train_scaled.shape[1])

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
            start_time = time.time()  # Start timer
            history = model.fit(
                X1_train_scaled,
                y1_train,
                epochs=cfg.train.epochs,
                batch_size=cfg.train.batch_size,
                validation_split=cfg.train.validation_split,
                callbacks=[early_stop, reduce_lr, WandbLogger()],
                verbose=1,
            )
            elapsed = time.time() - start_time  # Stop timer
            logger.info(f"⏱️ Training took {elapsed:.2f} seconds.")

            # Save training artifacts
            artifact_dir = "artifacts"
            os.makedirs("artifacts", exist_ok=True)
            weights_path = os.path.join(artifact_dir, "model.weights.h5")
            model.save_weights(weights_path)
            # model.save_weights(os.path.join(artifact_dir, "model.weights.h5"))
            joblib.dump(history.history, os.path.join(artifact_dir, "history.pkl"))
            logger.info(" Logging metrics and saving to Weights & Biases...")
            val_acc = history.history.get("val_accuracy", [None])[-1]
            logger.info(f" Final validation accuracy: {val_acc:.4f}")
            wandb.save(weights_path)
            wandb.log({"final_val_accuracy": history.history["val_accuracy"][-1]})
        else:
            # Train and save XGBoost model
            logger.info(" Training XGBoost model...")
            artifact_dir = "artifacts"
            os.makedirs(artifact_dir, exist_ok=True)
            model.fit(X1_train_scaled, y1_train)
            joblib.dump(model, os.path.join(artifact_dir, "xgboost_model.pkl"))
            joblib.dump(label_encoder, os.path.join(artifact_dir, "label_encoder.pkl"))
            joblib.dump(scaler, os.path.join(artifact_dir, "scaler.pkl"))
            logger.info(" XGBoost model and preprocessors saved.")

        wandb.finish()

        # Save config used
        with open("artifacts/used_config.yaml", "w") as f:
            OmegaConf.save(config=cfg, f=f)
        logger.info("Config saved with run artifacts.")
        np.save("artifacts/test_features.npy", X1_test_scaled)
        np.save("artifacts/test_labels.npy", y1_test)
        logger.info("Training complete!")
        console.rule("[bold green]Training Pipeline Complete ")

        def upload_to_gcs(local_path, bucket_name):
            if os.path.exists(local_path):
                remote_path = f"gs://{bucket_name}/{local_path}"
                subprocess.run(
                    ["gsutil", "-m", "cp", "-r", local_path, remote_path], check=True
                )
                logger.info(f"✅ Uploaded {local_path} to {remote_path}")
            else:
                logger.warning(f"⚠️ {local_path} not found. Skipping GCS upload.")

        # After training completes
        if cfg.get("gcs") and cfg.gcs.upload_artifacts:
            bucket = cfg.gcs.bucket_name
            upload_to_gcs("artifacts/", bucket)
            upload_to_gcs(cfg.hydra.run.dir, bucket)  # Upload full Hydra output folder


if __name__ == "__main__":
    try:
        main()
    except Exception:
        console.print_exception()
        logger.exception(" Error occurred during model training.")
        raise
