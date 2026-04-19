# ══════════════════════════════════════════════════════════
# SkinScan AI — Model Converter (H5 → ONNX)
# Google Colab Ready | Streamlined Version
# ══════════════════════════════════════════════════════════

# =========================
# STEP 1 — Install Packages
# =========================
!pip install tf2onnx gdown tensorflow -q

# =========================
# STEP 2 — Download Model
# =========================
import gdown
import os

FILE_ID = "18VE_D81425cZVYwAXjOn0gWti8_lZSML"
H5_FILE  = "skin_cancer_cnn.h5"

print("📥 Downloading model from Google Drive...")

url = f"https://drive.google.com/uc?id={FILE_ID}"
gdown.download(url, H5_FILE, quiet=False)

print(f"✅ Downloaded size: {os.path.getsize(H5_FILE)/1e6:.2f} MB")

# =========================
# STEP 3 — Load Keras Model
# =========================
import tensorflow as tf

print("\n🔄 Loading Keras model...")
model = tf.keras.models.load_model(H5_FILE)

print("✅ Model Loaded Successfully")

# =========================
# STEP 4 — Convert to ONNX
# =========================
import tf2onnx

print("\n🔄 Converting to ONNX...")

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path="skin_cancer_model.onnx"
)

print("✅ ONNX conversion complete!")

# =========================
# STEP 5 — Verify ONNX Model
# =========================
import onnxruntime as ort
import numpy as np

print("\n🧪 Testing ONNX model...")

session = ort.InferenceSession("skin_cancer_model.onnx")

dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)

input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: dummy_input})

print("✅ ONNX working fine!")
print("Output shape:", output[0].shape)

# =========================
# STEP 6 — Save to Google Drive
# =========================
from google.colab import drive
import shutil

drive.mount('/content/drive')

DEST = "/content/drive/MyDrive/skin_cancer_model.onnx"

shutil.copy("skin_cancer_model.onnx", DEST)

print("\n✅ Uploaded to Google Drive:")
print(DEST)

# =========================
# FINAL INSTRUCTIONS
# =========================
print("\n📌 NEXT STEP:")
print("1. Open Google Drive")
print("2. Find skin_cancer_model.onnx")
print("3. Right click → Share → Anyone with link")
print("4. Copy File ID")
print("5. Paste in app.py → ONNX_GDRIVE_ID")
