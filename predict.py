"""
predict.py
----------
Loads the trained CNN and predicts the sign shown in a single image.

Uses model.predict() - per your Ch.8 Unit 03 slides, Keras provides
this method "well-prepared for training and evaluation" alongside
fit() and evaluate(). It automatically switches off Dropout (which is
only meant to be active during training) and runs a clean forward
pass through the network.

Run:
    python predict.py path/to/your/image.jpg
"""

import json
import sys

import numpy as np
import tensorflow as tf

MODEL_PATH = "sign_model.keras"
LABELS_PATH = "class_labels.json"
IMG_HEIGHT = 128
IMG_WIDTH = 128


def load_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


def predict_image(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_labels()

    # Load at full original size first (not resized yet).
    img = tf.keras.utils.load_img(image_path)

    # If the image isn't square, resizing it straight to 128x128 would
    # stretch/distort the hand's proportions in a way the model never
    # saw during training. So we center-crop to the largest possible
    # square first, THEN resize - matching backend/app.py's logic.
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    img_array = tf.keras.utils.img_to_array(img)

    # The model expects a BATCH of images, shape [batch_size, H, W, 3].
    # We only have one image, so we add a batch dimension of size 1.
    img_array = tf.expand_dims(img_array, 0)

    # model.predict() runs a forward pass and returns the softmax
    # output: one probability per class (they sum to 1 across all 29).
    predictions = model.predict(img_array, verbose=0)[0]

    best_idx = int(np.argmax(predictions))          # index of the highest probability
    confidence = float(predictions[best_idx]) * 100  # as a percentage

    print(f"\nPredicted sign: '{class_names[best_idx]}'")
    print(f"Confidence: {confidence:.2f}%")

    # Also show the next two most likely guesses - useful when the
    # model is unsure between visually similar signs.
    top3_idx = np.argsort(predictions)[-3:][::-1]
    print("\nTop 3 guesses:")
    for idx in top3_idx:
        print(f"  {class_names[idx]}: {predictions[idx] * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    predict_image(sys.argv[1])