import numpy as np


def predict(model, input_data):
    """
    Runs prediction on given input data using the trained model.

    Args:
        model: A compiled and loaded Keras model.
        input_data (np.ndarray): Input features, expected shape (N, features).

    Returns:
        np.ndarray: Predicted class probabilities, shape (N, C).
    """
    if input_data.ndim == 2:
        input_data = np.expand_dims(input_data, axis=-1)
        input_data = np.expand_dims(input_data, axis=-1)  # shape becomes (N, F, 1, 1)

    preds = model.predict(input_data)
    return preds
