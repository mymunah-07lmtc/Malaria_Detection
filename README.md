# 🦟 AI Malaria Detector

**Research Prototype | Biomedical Engineering, USTTB**

An AI-powered malaria detection tool using deep learning (Convolutional Neural Networks) to classify blood cell images as "Parasitized" (infected) or "Uninfected" (healthy). Achieves **96% accuracy** on the NIH Malaria dataset.

---

## 📦 Project Structure
Malaria_Detection/

├── app.py                   # Streamlit web application

├── train_malaria_model.py   # CNN training script

├── requirements.txt         # Python dependencies

├── .gitignore               # Files to exclude from Git

└── README.md                # This file

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Malaria_Detection.git
cd Malaria_Detection
```
### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```
### 3. Install Dependencies
The trained model files are too large for GitHub. Download them from [Google Drive](https://drive.google.com/file/d/14Cz8IPuspJ8CoVCNFrhWhriWuXTlJ4IU/view?usp=sharing)
```bash
pip install -r requirements.txt
```
### 4. Download the Model
The trained model (malaria_detector.keras) is too large for GitHub. Download it from Google Drive and place it in the project root.

### 5. Run the App
```bash
streamlit run app.py
```
---

## 📊 Dataset
- **NIH Malaria Dataset**
- **27,558 blood cell images** (13,779 Parasitized, 13,779 Uninfected)
- **Source:** National Institutes of Health (NIH)
- **Publicly available** for research purposes

---

## 🧠 Model Performance
| Metric | Value |
| :--- | :--- |
| **Architecture** | CNN (Convolutional Network) |
| **Framework** | TensorFlow / Keras |
| **Test Accuracy** | 96.17% |
| **Confidence Range** | 97–99% |
| **Training Data** | 22,047 images |
| **Validation Data** | 5,511 images |
| **Epochs** | 15 |

---

## 📄 Disclaimer
⚠️ **For Research Purposes Only.** This tool is a proof-of-concept prototype. It is **NOT** a certified medical device. All predictions must be verified by a qualified healthcare professional.

---

## 📬 Contact

**Author:** Maimouna Tougoutcho Coulibaly

**Email:** maimounatcoul@gmail.com

**GitHub:** [github.com/mymunah-07lmtc](https://www.github.com/mymunah-07lmtc)

**LinkedIn:** [linkedin.com/in/maimouna-tougoutcho-coulibaly](https://www.linkedin.com/in/maimouna-tougoutcho-coulibaly)

---

**Built with ❤️ in Bamako, Mali | NIH Malaria Dataset**
