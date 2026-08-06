"""
app.py
------
Flask backend for PraneethSigns - serves predictions from the trained
sign_model.keras CNN to the React frontend.

API CONTRACT (must match the frontend exactly):

    POST /api/predict
    Request:  multipart/form-data, field name "image" (a file/blob)
    Response: {
        "label": "A",
        "confidence": 0.997,
        "top3": [
            {"label": "A", "confidence": 0.997},
            {"label": "E", "confidence": 0.002},
            {"label": "I", "confidence": 0.001}
        ]
    }

    Note: "confidence" is a FRACTION between 0 and 1, not a percentage -
    the frontend multiplies by 100 itself.

FILES NEEDED IN THIS FOLDER:
    sign_model.keras     (copy from your training project)
    class_labels.json    (copy from your training project)

Run:
    python app.py
"""

import io
import json

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

MODEL_PATH = "sign_model.keras"
LABELS_PATH = "class_labels.json"
IMG_HEIGHT = 128
IMG_WIDTH = 128
TOP_K = 3

app = Flask(__name__)

# The React dev server (Vite) runs on a different port than Flask,
# so the browser blocks the request by default (CORS). This allows
# the frontend's origin to actually reach this API during development.
CORS(app, resources={r"/api/*": {"origins": "*"}})

# -----------------------------------------------------------------
# Load the model and class labels ONCE, when the server starts - not
# on every request. Loading a model takes a couple seconds; running
# a prediction on an already-loaded model takes milliseconds.
# -----------------------------------------------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    class_names = json.load(f)
print(f"Model loaded. {len(class_names)} classes: {class_names}")


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Turns raw uploaded image bytes into the exact array shape/format
    the model expects - same preprocessing used in training and in
    predict.py / webcam_predict.py:
      - RGB color order
      - resized to IMG_HEIGHT x IMG_WIDTH
      - wrapped in a batch dimension of size 1
    (Pixel values are left as raw 0-255 floats - the model has its own
    Rescaling(1./255) layer built in, so we don't normalize here.)
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # The frontend's guide box isn't perfectly square (it's a rectangle
    # sized relative to the video element), so naively resizing straight
    # to 128x128 would stretch/distort the hand's proportions in a way
    # the model never saw during training. Instead, we center-crop to
    # the largest possible square first, THEN resize - this preserves
    # proportions correctly, matching how training images were framed.
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype="float32")
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, H, W, 3)
    return img_array


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

    predictions = model.predict(img_array, verbose=0)[0]  # shape: (num_classes,)

    # Get the indices of the top 3 highest-probability classes,
    # sorted from highest to lowest confidence.
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
    """Simple check to confirm the server + model are up and running."""
    return jsonify({"status": "ok", "num_classes": len(class_names)})


if __name__ == "__main__":
    # debug=True auto-reloads on code changes - handy during development.
    # Flask's default port is 5000, matching API_BASE in the frontend.
    app.run(debug=True, port=5000)