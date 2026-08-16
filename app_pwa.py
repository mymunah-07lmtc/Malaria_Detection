import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
import gdown

# -------------------------------
# MODEL DOWNLOAD (First run only)
# -------------------------------
MODEL_PATH = "malaria_detector.keras"

def download_model():
    """Download the model from Google Drive if not present."""
    if os.path.exists(MODEL_PATH):
        return
    
    st.info("📥 Downloading AI model... This may take a few minutes.")
    
    # Google Drive file ID (replace with your actual ID)
    FILE_ID = "14Cz8IPuspJ8CoVCNFrhWhriWuXTlJ4IU"  # <-- REPLACE THIS
    
    try:
        # Method 1: Using gdown (most reliable)
        import gdown
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    except ImportError:
        # Method 2: Using requests (fallback)
        import requests
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        response = requests.get(url, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    if os.path.exists(MODEL_PATH):
        st.success("✅ Model downloaded successfully!")
    else:
        st.error("❌ Model download failed. Please check your internet connection.")

# Download model on first run
download_model()

# -------------------------------
# LOAD KERAS MODEL
# -------------------------------
import tensorflow as tf

@st.cache_resource
def load_keras_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_keras_model()

if model is None:
    st.stop()

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

# -------------------------------
# IMAGE CAPTURE
# -------------------------------
st.subheader(text["capture"])

img = st.camera_input(text["camera"])
uploaded_file = st.file_uploader(text["upload"], type=["jpg", "jpeg", "png"])

image = None
if img is not None:
    image = Image.open(img)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)

# -------------------------------
# PROCESSING
# -------------------------------
if image is not None:
    st.image(image, caption="Captured Image", use_container_width=True)
    
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner(text["analyzing"]):
        prediction = model.predict(img_array)[0][0]
    
    st.divider()
    st.subheader(text["result"])
    
    if prediction > 0.5:
        # Uninfected (Healthy)
        st.success(text["uninfected"])
        st.info(text["uninfected_msg"])
        confidence_display = prediction * 100
    else:
        # Parasitized (Infected)
        st.error(text["infected"])
        st.warning(text["infected_msg"])
        confidence_display = (1 - prediction) * 100
    
    st.metric(text["confidence"], f"{confidence_display:.2f}%")
    
    st.divider()
    st.caption(text["license"])

st.divider()
st.caption(text["footer"])