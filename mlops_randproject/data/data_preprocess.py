# data/data_preprocess.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Construct safe relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path_3sec = os.path.join(
    BASE_DIR, "..", "..", "data", "music-dataset", "features_3_sec.csv"
)
data_path_30sec = os.path.join(
    BASE_DIR, "..", "..", "data", "music-dataset", "features_30_sec.csv"
)

# Load datasets
df = pd.read_csv(data_path_3sec)
df2 = pd.read_csv(data_path_30sec)

# Drop non-feature columns
df.drop(labels="filename", axis=1, inplace=True)
df2.drop(labels="filename", axis=1, inplace=True)

# Split into features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X1 = df2.iloc[:, :-1]
y1 = df2.iloc[:, -1]

# Encode genre labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y1_encoded = label_encoder.transform(y1)  # use same encoder

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X1_train_scaled = scaler.transform(X1)

# Exported variables for training
# Use these in model_training.py
X1_train_scaled = X1_train_scaled
y1_train = y1_encoded
label_encoder = label_encoder
scaler = scaler

# os.makedirs("artifacts", exist_ok=True)
# np.save("artifacts/test_labels.npy", y1_train)
# Save test features and labels for prediction
os.makedirs(os.path.join(BASE_DIR, "..", "..", "artifacts"), exist_ok=True)
np.save(
    os.path.join(BASE_DIR, "..", "..", "artifacts", "test_features.npy"),
    X1_train_scaled,
)
np.save(os.path.join(BASE_DIR, "..", "..", "artifacts", "test_labels.npy"), y1_train)
