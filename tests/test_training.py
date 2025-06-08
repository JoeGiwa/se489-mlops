import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlops_randproject.train_model import train_model

def test_training_runs_and_returns_model():
    model, history = train_model()

    # Ensure model object is returned
    assert model is not None
    assert hasattr(model, "predict")

    # Validate training history
    assert history is not None
    assert "loss" in history
    assert "val_accuracy" in history
    assert isinstance(history["loss"], list)
    assert len(history["loss"]) > 0
