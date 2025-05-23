
import joblib
import numpy as np
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from mlops_randproject.model_zoo import build_mlp, build_cnn, build_xgboost
import hydra
from omegaconf import DictConfig
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Load preprocessed test data
    try:
        logger.info("📦 Loading test features...")
        test_features = np.load(cfg.predict.test_features_path)
        logger.info(f" Loaded test features: shape={test_features.shape}")

        # Initialize model
        model = Sequential()
        input_shape = test_features.shape[1]

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
        preds = model.predict(test_features)
        predicted_classes = np.argmax(preds, axis=1)

        # Save predictions
        os.makedirs("artifacts", exist_ok=True)
        output_path = os.path.join("artifacts", cfg.predict.output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savetxt(output_path, predicted_classes, fmt='%d')
        logger.info(f" Predictions saved to {output_path}")
        
    except Exception as e:
        logger.exception(" Error occurred during prediction.")
        raise

    
if __name__ == "__main__":
    main()
