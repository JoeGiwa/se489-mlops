import os
import time
import wandb
import hydra
import joblib
import logging
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from omegaconf import OmegaConf, DictConfig
from tensorflow.keras.models import Sequential
from mlops_randproject.utils.monitor import SystemMonitor
from mlops_randproject.model_zoo import build_mlp, build_cnn


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("mlops")
console = Console()

OmegaConf.register_new_resolver("env", lambda key, default="." : os.getenv(key, default))


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    with SystemMonitor(interval=5):

        try:
            console.rule("[bold blue]Starting Prediction Pipeline")

            logger.info(" Loading test features...")
            test_features = np.load(cfg.predict.test_features_path)
            logger.info(f" Loaded test features: shape={test_features.shape}")

            # Initialize model
            model = Sequential()
            input_shape = (test_features.shape[1], 1, 1)  # becomes (58, 1, 1)
            test_features = np.reshape(test_features, (-1, 58, 1, 1))

            # Load model architecture based on config
            if cfg.model.name == "mlp":
                model = build_mlp(cfg.model, input_shape)
            elif cfg.model.name == "cnn":
                model = build_cnn(cfg.model, input_shape)
            else:
                raise ValueError(f"Unsupported model: {cfg.model.name}")

            # Load weights
            logger.info(" Loading model weights...")
            model.load_weights(cfg.predict.weights_path)
            logger.info(" Model weights loaded.")


            # Make predictions
            logger.info(" Making predictions...")
            start_time = time.time()
            preds = model.predict(test_features)
            elapsed = time.time() - start_time
            logger.info(f"⏱ Prediction completed in {elapsed:.2f} seconds.")

            predicted_classes = np.argmax(preds, axis=1)

            # Save predictions
            os.makedirs("artifacts", exist_ok=True)
            output_path = os.path.join("artifacts", cfg.predict.output_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savetxt(output_path, predicted_classes, fmt='%d')
            logger.info(f" Predictions saved to {output_path}")
            
            # Log to Weights & Biases
            run_name = f"{cfg.model.name}_predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project="mlops-randproject-predict",
                config=OmegaConf.to_container(cfg, resolve=True),  #Make config JSON-serializable
                name=run_name
            )          
            wandb.log({
                "num_predictions": len(predicted_classes),
                "prediction_duration_sec": elapsed
            })
            wandb.save(output_path)
            wandb.finish()
            logger.info(" Logged predictions to Weights & Biases")    
            console.rule("[bold green]Prediction Complete ")


        except Exception as e:
            console.print_exception()
            logger.exception(" Error occurred during prediction.")
            raise

    
if __name__ == "__main__":
    main()
