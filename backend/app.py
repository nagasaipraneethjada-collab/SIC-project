"""
app.py
------
Flask backend for PraneethSigns - serves predictions from the
TFLite-converted sign_model.tflite CNN to the React frontend.
"""

import io
import json

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

MODEL_PATH = "sign_model.tflite"
LABELS_PATH = "class_labels.json"
IMG_HEIGHT = 128
IMG_WIDTH = 128
TOP_K = 3

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

print("Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH) as f:
    class_names = json.load(f)
print(f"Model loaded. {len(class_names)} classes: {class_names}")


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype="float32")
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def run_inference(img_array: np.ndarray) -> np.ndarray:
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output[0]


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file field found in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    try:
        img_array = preprocess_image(file.read())
    except Exception:
        return jsonify({"error": "Could not read image - is it a valid image file?"}), 400

    predictions = run_inference(img_array)

    top_indices = np.argsort(predictions)[-TOP_K:][::-1]

    top3 = [
        {"label": class_names[i], "confidence": float(predictions[i])}
        for i in top_indices
    ]

    best_idx = int(top_indices[0])
    response = {
        "label": class_names[best_idx],
        "confidence": float(predictions[best_idx]),
        "top3": top3,
    }

    return jsonify(response)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "num_classes": len(class_names)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)