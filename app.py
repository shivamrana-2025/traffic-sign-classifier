"""Streamlit demo:  streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras

from src.labels import CLASS_NAMES

MODELS_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦")
st.title(" Traffic Sign Classifier")
st.caption("CNN trained on GTSRB (43 classes). Upload a cropped photo of a road sign.")

available = sorted(p.stem for p in MODELS_DIR.glob("*.keras"))
if not available:
    st.error("No trained model found in models/. Run `python -m src.train --model baseline` first.")
    st.stop()

name = st.sidebar.selectbox("Model", available)


@st.cache_resource
def get_model(model_name):
    return keras.models.load_model(MODELS_DIR / f"{model_name}.keras")


model = get_model(name)
size = model.input_shape[1]

file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", width=250)

    x = np.array(img.resize((size, size), Image.LANCZOS), dtype=np.float32)[None]
    probs = model.predict(x, verbose=0)[0]
    top = probs.argsort()[::-1][:5]

    st.subheader(f"Prediction: {CLASS_NAMES[top[0]]}")
    st.metric("Confidence", f"{probs[top[0]] * 100:.1f}%")
    if probs[top[0]] < 0.6:
        st.warning("Low confidence. Try a tighter crop where the sign fills most of the frame.")

    df = pd.DataFrame({"Class": [CLASS_NAMES[i] for i in top], "Probability": probs[top]})
    st.bar_chart(df.set_index("Class"))
