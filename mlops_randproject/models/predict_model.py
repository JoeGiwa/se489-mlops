import os
import time
import wandb
import hydra
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from omegaconf import OmegaConf, DictConfig
from tensorflow.keras.models import Sequential
from mlops_randproject.utils.monitor import SystemMonitor
from mlops_randproject.model_zoo import build_mlp, build_cnn

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("mlops")
console = Console()

OmegaConf.register_new_resolver("env", lambda key, default=".": os.getenv(key, default))


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    model_name = cfg.model.name
    with SystemMonitor(interval=5):
        try:
            console.rule("[bold blue]Starting Prediction Pipeline")

            logger.info(" Loading test features and labels...")
            test_features = np.load(cfg.predict.test_features_path)
            true_labels = np.load(cfg.predict.test_labels_path)
            logger.info(f" Test features shape: {test_features.shape}")

            model = Sequential()
            input_shape = test_features.shape[1:]  # (58, 1)
            if cfg.model.name == "cnn":
                test_features = test_features.reshape(-1, test_features.shape[1], 1)

                input_shape = test_features.shape[1]  # just the 58
                logger.info(f" Reshaped test features for CNN: {test_features.shape}")
            else:
                input_shape = test_features.shape[1:]

            if cfg.model.name == "mlp":
                model = build_mlp(cfg.model, input_shape)
            elif cfg.model.name == "cnn":
                model = build_cnn(cfg.model, input_shape=input_shape)
            else:
                raise ValueError(f"Unsupported model: {cfg.model.name}")

            logger.info(" Loading model weights...")
            model.load_weights(cfg.predict.weights_path)
            logger.info(" Model weights loaded.")

            logger.info(" Making predictions...")
            start_time = time.time()
            preds = model.predict(test_features)
            elapsed = time.time() - start_time
            logger.info(f"⏱ Prediction completed in {elapsed:.2f} seconds.")

            predicted_classes = np.argmax(preds, axis=1)

            os.makedirs("artifacts", exist_ok=True)
            output_path = f"artifacts/predictions_{model_name}.txt"
            np.savetxt(output_path, predicted_classes, fmt="%d")
            logger.info(f" Predictions saved to {output_path}")

            #  Confusion Matrix Plot
            logger.info(" Generating confusion matrix...")
            cm = confusion_matrix(true_labels, predicted_classes)
            cm_path = f"artifacts/confusion_matrix_{model_name}.png"
            logger.info(f" Saving confusion matrix to {cm_path}")
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(cm_path)
            logger.info(f" Confusion matrix saved to {cm_path}")

            # ✅ Save confusion matrix as markdown for CML
            md_path = f"artifacts/confusion_matrix_{model_name}.md"
            with open(md_path, "w") as f:
                f.write(f"### Confusion Matrix for `{model_name}`\n\n")
                f.write("![](" + cm_path + ")")
            logger.info(f" Markdown confusion matrix saved to {md_path}")

            #  W&B Logging
            run_name = (
                f"{cfg.model.name}_predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            wandb.init(
                project="mlops-randproject-predict",
                config=OmegaConf.to_container(cfg, resolve=True),
                name=run_name,
            )
            wandb.log(
                {
                    "num_predictions": len(predicted_classes),
                    "prediction_duration_sec": elapsed,
                    "confusion_matrix": wandb.Image(cm_path),
                }
            )
            wandb.save(output_path)
            wandb.finish()

            logger.info(" Logged predictions and confusion matrix to Weights & Biases")
            console.rule("[bold green]Prediction Complete")

        except Exception:
            console.print_exception()
            logger.exception(" Error occurred during prediction.")
            raise


if __name__ == "__main__":
    main()
