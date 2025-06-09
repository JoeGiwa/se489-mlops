import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlops_randproject.train_model import train_model
from mlops_randproject.predict_api import predict


def test_predict_on_valid_input():
    # Step 1: Create or load a model
    model, _ = train_model()

    # Step 2: Create a dummy input simulating a mel-spectrogram
    input_sample = np.random.rand(1, 128, 660).astype(
        "float32"
    )  # Adjust shape if needed

    # Step 3: Predict using the model
    prediction = predict(model, input_sample)

    # Step 4: Assertions
    assert prediction is not None, "Prediction result should not be None"
    assert isinstance(prediction, np.ndarray), "Prediction must return a NumPy array"
    assert prediction.shape == (
        1,
        10,
    ), "Expected prediction shape of (1, 10) for 10 genres"
    assert np.all(prediction >= 0) and np.all(
        prediction <= 1
    ), "Probabilities must be in [0, 1]"
