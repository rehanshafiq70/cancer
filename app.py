import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

# ======================
# CONFIG
# ======================
MODEL_PATH = "model.onnx"

st.set_page_config(page_title="SkinScan AI", layout="centered")

# ======================
# LOAD MODEL (LOCAL FILE)
# ======================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ model.onnx file missing in repo!")
        st.stop()

    return ort.InferenceSession(MODEL_PATH)

session = load_model()

# ======================
# PREDICT FUNCTION
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

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image")

    if st.button("Predict"):
        label, conf = predict(img)

        if label == "Malignant":
            st.error(f"⚠️ {label} ({conf}%)")
        else:
            st.success(f"✅ {label} ({conf}%)")
