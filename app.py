import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Face Recognition System",
    layout="centered"
)

# -----------------------------
# Load model (robust)
# -----------------------------
MODEL_PATHS = ["face_model.keras", "face_model.h5"]
model = None

for path in MODEL_PATHS:
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
            st.info(f"✅ Loaded model: {path}")
            break
        except Exception as e:
            st.warning(f"⚠️ Found {path} but failed to load: {e}")

if model is None:
    st.error("❌ No trained model found. Please run `train_model.py` first.")
    st.stop()

# -----------------------------
# Load class names (folder names)
# -----------------------------
if not os.path.exists("class_names.json"):
    st.error("❌ class_names.json not found. Please re-run training.")
    st.stop()

with open("class_names.json") as f:
    class_names = json.load(f)

# -----------------------------
# Load real name mapping (folder → actual name)
# -----------------------------
name_mapping = {}
if os.path.exists("name_mapping.json"):
    with open("name_mapping.json") as f:
        name_mapping = json.load(f)
else:
    st.warning("⚠️ name_mapping.json not found — showing folder names.")

# -----------------------------
# Image settings
# -----------------------------
IMG_SIZE = (224, 224)

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Face Recognition System")
st.write("Upload a photo to identify the person")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    confidence = np.max(predictions) * 100

    raw_label = class_names[np.argmax(predictions)]
    predicted_name = name_mapping.get(raw_label, raw_label)

    # Display result
    st.success(f"✅ Predicted Person: **{predicted_name}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

    # Optional: unknown face detection
    THRESHOLD = 70
    if confidence < THRESHOLD:
        st.warning("⚠️ Low confidence — this face may be unknown.")
