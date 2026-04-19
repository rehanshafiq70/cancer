# ═══════════════════════════════════════
# SkinScan AI — H5 → ONNX Converter
# Run in Google Colab only
# ═══════════════════════════════════════

!pip install tf2onnx gdown tensorflow onnx -q

import gdown
import tensorflow as tf
import tf2onnx
import os
import shutil
import numpy as np
from google.colab import drive

# ======================
# STEP 1: Download H5
# ======================
FILE_ID = "18VE_D81425cZVYwAXjOn0gWti8_lZSML"
H5_FILE = "skin_cancer_cnn.h5"

print("📥 Downloading model...")

gdown.download(
    f"https://drive.google.com/uc?id={FILE_ID}",
    H5_FILE,
    quiet=False
)

print("✅ Downloaded")

# ======================
# STEP 2: Load Model
# ======================
print("🔄 Loading model...")
model = tf.keras.models.load_model(H5_FILE)
print("✅ Model loaded")

# ======================
# STEP 3: Convert ONNX
# ======================
print("🔄 Converting to ONNX...")

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

onnx_path = "skin_cancer_model.onnx"

tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=onnx_path
)

print("✅ ONNX created")

# ======================
# STEP 4: Save to Drive
# ======================
drive.mount('/content/drive')

dest = "/content/drive/MyDrive/skin_cancer_model.onnx"
shutil.copy(onnx_path, dest)

print("✅ Saved to Drive:", dest)
