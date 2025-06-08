import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlops_randproject.evaluate import evaluate_model

def test_evaluation_metrics_are_valid():
    # Simulate predictions and labels
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 0, 0])

    results = evaluate_model(y_pred, y_true)

    assert "accuracy" in results
    assert "f1_score" in results
    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["f1_score"] <= 1.0
