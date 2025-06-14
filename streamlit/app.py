import streamlit as st
import pandas as pd
import requests

# Cloud Run URL
API_URL = "https://predict-api-296893387432.us-central1.run.app/predict"

st.title("🎵 Music Genre Prediction (Cloud Deployed)")

# Model selection
model_name = st.selectbox("Choose a model:", ["xgboost", "mlp", "cnn"])

st.markdown("### Option 1: Paste 58 comma-separated features")
features_input = st.text_area("Input Features", placeholder="0.1, 0.2, ..., 0.9")

st.markdown("### Option 2: Upload a CSV file with one row of 58 features")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])


def parse_features_from_text(text):
    try:
        features = list(map(float, text.strip().split(",")))
        if len(features) != 58:
            st.warning("⚠️ Please enter exactly 58 features.")
            return None
        return features
    except Exception as e:
        st.error(f"❌ Invalid input. Error: {str(e)}")
        return None


def parse_features_from_csv(file):
    try:
        df = pd.read_csv(file, header=None)
        if df.shape != (1, 58):
            st.warning("⚠️ CSV must contain exactly 1 row and 58 columns.")
            return None
        return df.iloc[0].tolist()
    except Exception as e:
        st.error(f"❌ Failed to read CSV. Error: {str(e)}")
        return None


# Predict button
if st.button("Predict"):
    features = None

    if uploaded_file:
        features = parse_features_from_csv(uploaded_file)
    elif features_input:
        features = parse_features_from_text(features_input)

    if features:
        payload = {"features": features, "model_name": model_name}
        with st.spinner("Predicting..."):
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                st.success(f"🎧 Prediction: {response.json()['prediction']}")
            else:
                st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
    else:
        st.warning("⚠️ Please provide input via text area or CSV upload.")
