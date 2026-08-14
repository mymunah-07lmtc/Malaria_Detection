import streamlit as st
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

# PWA Meta Tags
st.markdown("""
<link rel="manifest" href="manifest.json">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="Sahel MalaVision">
""", unsafe_allow_html=True)

# -------------------------------
# LANGUAGE SELECTOR
# -------------------------------
lang = st.sidebar.radio("Language / Langue", ["English", "Français"])

# Language dictionary
T = {
    "English": {
        "title": "🦟 Sahel MalaVision",
        "subtitle": "AI-Powered Malaria Detector | Research Prototype",
        "disclaimer_title": "📋 Disclaimer",
        "disclaimer": "**For Research Purposes Only.** This tool is a proof-of-concept prototype. It is **NOT** a certified medical device. All predictions must be verified by a qualified healthcare professional.",
        "capture": "📸 Capture Blood Smear Image",
        "camera": "Take a photo of the blood smear",
        "upload": "Or upload an image",
        "analyzing": "Analyzing...",
        "result": "🩺 Screening Result",
        "infected": "⚠️ **Result: Parasitized (Infected)**",
        "infected_msg": "Refer the patient for immediate clinical evaluation and confirmatory testing.",
        "uninfected": "✅ **Result: Uninfected**",
        "uninfected_msg": "No parasites detected. Continue routine monitoring.",
        "confidence": "Confidence",
        "license": "🔑 Licensed to: [Clinic Name] | Expires: [Date]",
        "footer": "Built with ❤️ by Maimouna Tougoutcho Coulibaly | Sahel BioMed Solutions",
        "mode": "Mode",
        "demo": "Demo",
        "live": "Live"
    },
    "Français": {
        "title": "🦟 Sahel MalaVision",
        "subtitle": "Détecteur de Paludisme par IA | Prototype de Recherche",
        "disclaimer_title": "📋 Avis de non-responsabilité",
        "disclaimer": "**À des fins de recherche uniquement.** Cet outil est un prototype de preuve de concept. Ce n'est **PAS** un dispositif médical certifié. Toutes les prédictions doivent être vérifiées par un professionnel de santé qualifié.",
        "capture": "📸 Capturer l'image du frottis sanguin",
        "camera": "Prendre une photo du frottis sanguin",
        "upload": "Ou télécharger une image",
        "analyzing": "Analyse en cours...",
        "result": "🩺 Résultat du dépistage",
        "infected": "⚠️ **Résultat : Parasité (Infecté)**",
        "infected_msg": "Orientez le patient pour une évaluation clinique immédiate et un test de confirmation.",
        "uninfected": "✅ **Résultat : Non infecté**",
        "uninfected_msg": "Aucun parasite détecté. Poursuivre la surveillance de routine.",
        "confidence": "Confiance",
        "license": "🔑 Licencié à : [Nom de la clinique] | Expire le : [Date]",
        "footer": "Construit avec ❤️ par Maimouna Tougoutcho Coulibaly | Sahel BioMed Solutions",
        "mode": "Mode",
        "demo": "Démo",
        "live": "En direct"
    }
}

text = T[lang]

# -------------------------------
# BRANDING
# -------------------------------
st.title(text["title"])
st.caption(text["subtitle"])

with st.expander(text["disclaimer_title"], expanded=True):
    st.warning(text["disclaimer"])

# LOAD KERAS MODEL
@st.cache_resource
def load_keras_model():
    model_path = "malaria_detector.keras"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please ensure 'malaria_detector.keras' is in the app directory.")
        return None
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    return model

model = load_keras_model()

if model is None:
    st.stop()

# -------------------------------
# IMAGE CAPTURE
# -------------------------------
st.subheader(text["capture"])

# Option 1: Camera input
img = st.camera_input(text["camera"])

# Option 2: File upload
uploaded_file = st.file_uploader(text["upload"], type=["jpg", "jpeg", "png"])

# Determine input source
image = None
if img is not None:
    image = Image.open(img)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)

# -------------------------------
# PROCESSING
# -------------------------------
if image is not None:
    # Display the image
    st.image(image, caption="Captured Image", use_container_width=True)
    
    # Preprocess
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    # Run inference
    with st.spinner(text["analyzing"]):
        prediction = model.predict(img_array)[0][0]
    
    # Result
    st.divider()
    st.subheader(text["result"])
    
    if prediction > 0.5:
        st.error(text["infected"])
        st.warning(text["infected_msg"])
        confidence_display = prediction * 100
    else:
        st.success(text["uninfected"])
        st.info(text["uninfected_msg"])
        confidence_display = (1 - prediction) * 100
    
    st.metric(text["confidence"], f"{confidence_display:.2f}%")
    
    # License info
    st.divider()
    st.caption(text["license"])

# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.caption(text["footer"])