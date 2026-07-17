import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Malaria Detector", page_icon="🦟", layout="centered")

st.title("🦟 AI Malaria Detector")
st.caption("Research Prototype by Maimouna Tougoutcho Coulibaly")
st.markdown("Upload a microscopic image of a blood cell to check for malaria parasites.")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_malaria_model():
    model_path = "malaria_detector.keras"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Please run train_malaria_model.py first.")
        return None
    model = load_model(model_path)
    return model

model = load_malaria_model()

if model is None:
    st.stop()

# -------------------------------
# DISCLAIMER
# -------------------------------
with st.expander("📋 Disclaimer"):
    st.warning("""
    **For Research Purposes Only.** This tool is a proof-of-concept prototype.
    It is **NOT** a certified medical device. All predictions must be verified by a qualified healthcare professional.
    """)

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess for model
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

    # Show result
    st.divider()
    st.subheader("Screening Result")

    # confidence > 0.5 = Uninfected (Class 1), confidence < 0.5 = Parasitized (Class 0)
    if confidence > 0.5:
        # Uninfected (Healthy)
        st.success(f"✅ **Result: Uninfected**")
        st.info("**Recommendation:** No parasites detected. Continue routine monitoring.")
        confidence_display = confidence * 100  # Raw confidence for Uninfected
    else:
        # Parasitized (Infected)
        st.error(f"⚠️ **Result: Parasitized (Infected)**")
        st.warning("**Recommendation:** Refer the patient for immediate clinical evaluation and confirmatory testing.")
        confidence_display = (1 - confidence) * 100  # Invert for Parasitized

    st.metric("Confidence", f"{confidence_display:.2f}%")

# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.caption("Built with ❤️ by Maimouna Tougoutcho Coulibaly in Bamako, Mali | NIH Malaria Dataset")