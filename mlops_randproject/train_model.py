import numpy as np

class DummyModel:
    def predict(self, x):
        return np.random.rand(x.shape[0], 10)  # simulate probabilities for 10 genres

def train_model():
    model = DummyModel()
    history = {"loss": [0.9, 0.6, 0.3], "val_accuracy": [0.5, 0.6, 0.7]}
    return model, history
