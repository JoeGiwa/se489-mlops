import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlops_randproject.preprocessing import load_and_preprocess_audio


def test_audio_preprocessing_returns_valid_shape():
    sample_path = "tests/sample.wav"
    if not os.path.exists(sample_path):
        pytest.skip("sample.wav not found in tests/ directory.")

    features = load_and_preprocess_audio(sample_path)

    assert isinstance(features, np.ndarray)
    assert len(features.shape) == 2
    assert features.shape[0] > 0 and features.shape[1] > 0
