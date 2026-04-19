# ══════════════════════════════════════════════════════════
# SkinScan AI — Model Converter (H5 → ONNX)
# Run this in Google Colab to convert your model
# ══════════════════════════════════════════════════════════

# Step 1: Install required packages
!pip install tf2onnx gdown -q

# Step 2: Download your model from Google Drive
import gdown
import os

FILE_ID = "18VE_D81425cZVYwAXjOn0gWti8_lZSML"
OUTPUT  = "skin_cancer_model.h5"

print("📥 Downloading model from Google Drive...")
gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", OUTPUT, quiet=False)
print(f"✅ Downloaded: {os.path.getsize(OUTPUT)/1e6:.1f} MB")

# Step 3: Load the Keras model
import tensorflow as tf
print("\n🔄 Loading Keras model...")
model = tf.keras.models.load_model(OUTPUT)
model.summary()

# Step 4: Convert to ONNX
import tf2onnx
import numpy as np

print("\n🔄 Converting to ONNX...")
input_signature = [
    tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name="input")
]

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path="skin_cancer_model.onnx"
)
print(f"✅ ONNX model saved! Size: {os.path.getsize('skin_cancer_model.onnx')/1e6:.1f} MB")

# Step 5: Verify ONNX model works
import onnxruntime as ort
print("\n🧪 Verifying ONNX model...")
sess = ort.InferenceSession("skin_cancer_model.onnx")
dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
out   = sess.run(None, {"input": dummy})
print(f"✅ ONNX inference OK! Output shape: {out[0].shape}, Values: {out[0]}")

# Step 6: Upload ONNX model to Google Drive
from google.colab import drive
drive.mount('/content/drive')

import shutil
dest = "/content/drive/MyDrive/skin_cancer_model.onnx"
shutil.copy("skin_cancer_model.onnx", dest)
print(f"\n✅ Uploaded to Google Drive: {dest}")
print("\n📋 NEXT STEP:")
print("1. Go to Google Drive")
print("2. Right-click on skin_cancer_model.onnx")
print("3. Click 'Share' → 'Anyone with link'")
print("4. Copy the File ID from the link")
print("   Example: https://drive.google.com/file/d/FILE_ID_HERE/view")
print("5. Paste the FILE_ID in the app.py ONNX_FILE_ID variable")
