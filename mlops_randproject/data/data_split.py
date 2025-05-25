import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_split_data(version="30_sec"):
    if version == "30_sec":
        df = pd.read_csv("data/music-dataset/features_30_sec.csv")
    else:
        df = pd.read_csv("data/music-dataset/features_3_sec.csv")

    # Drop 'filename' if it exists, then separate features and label
    if 'filename' in df.columns:
        df = df.drop(columns=['filename'])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, label_encoder, scaler
