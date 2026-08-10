"""
convert_to_tflite.py
---------------------
One-time script: converts sign_model.keras into a smaller,
memory-efficient sign_model.tflite for deployment.

Run once locally:
    python convert_to_tflite.py
"""

import tensorflow as tf

MODEL_PATH = "sign_model.keras"
OUTPUT_PATH = "sign_model.tflite"

print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: reduces size/memory further via quantization.
# Safe default - doesn't hurt accuracy meaningfully for most CNNs.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Done! Saved to {OUTPUT_PATH}")