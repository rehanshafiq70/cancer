import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import gdown
import os

# ======================
# CONFIG
# ======================
ONNX_GDRIVE_ID = "1wNRAdJOrmjaewqu9YRpqMqKS0RcihzDA"
MODEL_PATH = "model.onnx"

st.set_page_config(page_title="Skin Cancer AI", layout="centered")

# ======================
# DOWNLOAD MODEL SAFE
# ======================
def download_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000000:
        return

    url = f"https://drive.google.com/uc?id={ONNX_GDRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    download_model()
    return ort.InferenceSession(MODEL_PATH)

session = load_model()

# ======================
# PREDICT
# ======================
def predict(img):
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: img})[0]

    if out.shape[-1] == 1:
        mal = float(out[0][0])
        ben = 1 - mal
    else:
        ben = float(out[0][0])
        mal = float(out[0][1])

    label = "Malignant" if mal > 0.5 else "Benign"
    conf = round(max(mal, ben) * 100, 2)

    return label, conf

# ======================
# UI
# ======================
st.title("🧬 Skin Cancer Detection AI")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img)

    if st.button("Predict"):
        label, conf = predict(img)

        if label == "Malignant":
            st.error(f"⚠️ {label} ({conf}%)")
        else:
            st.success(f"✅ {label} ({conf}%)")
