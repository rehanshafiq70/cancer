import streamlit as st
import numpy as np
import os
import gdown
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "model.onnx"
FILE_ID = "PASTE_YOUR_FILE_ID_HERE"

# =========================
# SAFE DOWNLOAD MODEL
# =========================
def download_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100000:
        return True

    try:
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

        # CHECK FILE SIZE (VERY IMPORTANT FIX)
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
            st.error("❌ Model file corrupted or incomplete download")
            return False

        return True

    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        return False


# =========================
# LOAD MODEL SAFELY
# =========================
@st.cache_resource
def load_model():
    ok = download_model()
    if not ok:
        return None

    try:
        session = ort.InferenceSession(MODEL_PATH)
        return session
    except Exception as e:
        st.error("❌ ONNX Model corrupted (InvalidProtobuf)")
        st.error("Reconvert model in Colab again")
        return None


session = load_model()

if session is None:
    st.stop()

# =========================
# UI
# =========================
st.title("Skin Cancer AI")

img_file = st.file_uploader("Upload Image")

if img_file:
    img = Image.open(img_file).resize((224,224))
    arr = np.array(img).astype(np.float32)/255.0
    arr = np.expand_dims(arr,0)

    inp = session.get_inputs()[0].name
    out = session.run(None, {inp: arr})

    st.write("Prediction:", out[0])
