# ═════════════════════════════════════════════
# Skin Cancer Model → ONNX (STABLE VERSION)
# ═════════════════════════════════════════════

!pip install tf2onnx gdown -q

import gdown
import tensorflow as tf
import tf2onnx
import numpy as np
import os
import shutil

# =========================
# STEP 1: DOWNLOAD MODEL
# =========================
FILE_ID = "18VE_D81425cZVYwAXjOn0gWti8_lZSML"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

print("📥 Downloading model...")
gdown.download(URL, "model.h5", quiet=False)

# =========================
# STEP 2: LOAD MODEL SAFELY
# =========================
print("🔄 Loading model...")
model = tf.keras.models.load_model("model.h5", compile=False)

# =========================
# STEP 3: FIX INPUT SHAPE ISSUE
# =========================
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

# =========================
# STEP 4: CONVERT TO ONNX
# =========================
print("🔄 Converting to ONNX...")

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path="skin_cancer_model.onnx"
)

print("✅ ONNX created")

# =========================
# STEP 5: VERIFY MODEL
# =========================
import onnxruntime as ort

session = ort.InferenceSession("skin_cancer_model.onnx")

dummy = np.random.rand(1,224,224,3).astype(np.float32)
inp = session.get_inputs()[0].name
out = session.run(None, {inp: dummy})

print("✅ ONNX WORKING:", out[0].shape)

# =========================
# STEP 6: SAVE TO DRIVE
# =========================
from google.colab import drive
drive.mount("/content/drive")

dest = "/content/drive/MyDrive/skin_cancer_model.onnx"
shutil.copy("skin_cancer_model.onnx", dest)

print("✅ SAVED TO DRIVE:", dest)

print("\n👉 IMPORTANT:")
print("Drive file share → Anyone with link → COPY FILE ID")
