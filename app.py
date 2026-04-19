import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import requests
import os

# =============================
# CONFIG
# =============================
ONNX_GDRIVE_ID = "1wNRAdJOrmjaewqu9YRpqMqKS0RcihzDA"
MODEL_PATH = "model.onnx"

st.set_page_config(page_title="Skin Cancer AI", layout="centered")

# =============================
# DOWNLOAD MODEL FROM DRIVE
# =============================
def download_model():
    if os.path.exists(MODEL_PATH):
        return

    url = f"https://drive.google.com/uc?export=download&id={ONNX_GDRIVE_ID}"
    response = requests.get(url)

    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    download_model()
    session = ort.InferenceSession(MODEL_PATH)
    return session

# =============================
# PREDICTION
# =============================
def predict(image, session):
    img = image.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})[0]

    if output.shape[-1] == 1:
        mal = float(output[0][0])
        ben = 1 - mal
    else:
        ben = float(output[0][0])
        mal = float(output[0][1])

    label = "Malignant" if mal > 0.5 else "Benign"
    confidence = round(max(mal, ben) * 100, 2)

    return label, confidence

# =============================
# UI
# =============================
st.title("🧬 Skin Cancer Detection AI")

session = load_model()

uploaded = st.file_uploader("Upload Skin Image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image")

    if st.button("Predict"):
        label, conf = predict(img, session)

        if label == "Malignant":
            st.error(f"⚠️ {label} ({conf}%)")
        else:
            st.success(f"✅ {label} ({conf}%)")
