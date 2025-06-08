# mlops_randproject/preprocessing.py

import librosa
import numpy as np

def load_and_preprocess_audio(path, sr=22050, n_mels=128):
    y, sr = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype("float32")