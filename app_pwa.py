import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Sahel MalaVision",
    page_icon="🦟",
    layout="centered"
)

# PWA Meta Tags (for installation)
st.markdown("""
<link rel="manifest" href="manifest.json">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="Sahel MalaVision">
""", unsafe_allow_html=True)

# -------------------------------
# BRANDING
# -------------------------------
st.title("🦟 Sahel MalaVision")
st.caption("AI-Powered Malaria Detector | Research Prototype")

with st.expander("📋 Disclaimer"):
    st.warning("""
    **For Research Purposes Only.** This tool is a proof-of-concept prototype.
    It is **NOT** a certified medical device. All predictions must be verified by a qualified healthcare professional.
    """)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("malaria_detector.keras")
    return model

model = load_model()

# -------------------------------
# IMAGE CAPTURE
# -------------------------------
st.subheader("📸 Capture Blood Smear Image")

# Option 1: Camera input (for mobile)
img = st.camera_input("Take a photo of the blood smear")

# Option 2: File upload (fallback)
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

# Use whichever is provided
if img is not None:
    image = Image.open(img)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)
else:
    image = None

# -------------------------------
# PROCESSING
# -------------------------------
if image is not None:
    # Display the image
    st.image(image, caption="Captured Image", use_container_width=True)
    
    # Preprocess
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    with st.spinner("Analyzing..."):
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
    
    # Result
    st.divider()
    st.subheader("🩺 Screening Result")
    
    if confidence > 0.5:
        st.error(f"⚠️ **Result: Parasitized (Infected)**")
        st.warning("Refer the patient for immediate clinical evaluation and confirmatory testing.")
        confidence_display = confidence * 100
    else:
        st.success(f"✅ **Result: Uninfected**")
        st.info("No parasites detected. Continue routine monitoring.")
        confidence_display = (1 - confidence) * 100
    
    st.metric("Confidence", f"{confidence_display:.2f}%")
    
    # License Info
    st.divider()
    st.caption("🔑 Licensed to: [Clinic Name] | Expires: [Date]")

# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.caption("Built with ❤️ by Maimouna Tougoutcho Coulibaly | Sahel BioMed Solutions")