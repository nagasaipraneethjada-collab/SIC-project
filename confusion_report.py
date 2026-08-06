"""
confusion_report.py
--------------------
Runs the trained model against every image in asl_alphabet_test/ (one
sample per class) and builds a report showing exactly which letters
the model gets right, which it gets wrong, and what it confuses them
with. Useful for a "Limitations" section in your project write-up.

Uses the same center-crop-to-square preprocessing as predict.py and
backend/app.py, so results are consistent with what you've been
testing manually.

Run:
    python confusion_report.py
"""

import json
import os

import numpy as np
import tensorflow as tf

MODEL_PATH = "sign_model.keras"
LABELS_PATH = "class_labels.json"
TEST_DIR = "asl_alphabet_test/asl_alphabet_test"  # adjust if your path differs
IMG_HEIGHT = 128
IMG_WIDTH = 128


def preprocess(image_path):
    img = tf.keras.utils.load_img(image_path)
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    return tf.expand_dims(img_array, 0)


def true_label_from_filename(filename):
    # e.g. "A_test.jpg" -> "A", "del_test.jpg" -> "del"
    return filename.split("_")[0]


def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_names = json.load(f)

    files = sorted(f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        print(f"No image files found in {TEST_DIR} - check the path.")
        return

    results = []
    for filename in files:
        true_label = true_label_from_filename(filename)
        img_array = preprocess(os.path.join(TEST_DIR, filename))
        predictions = model.predict(img_array, verbose=0)[0]

        best_idx = int(np.argmax(predictions))
        predicted_label = class_names[best_idx]
        confidence = float(predictions[best_idx]) * 100

        # confidence the model gave to the ACTUAL correct answer,
        # even if it wasn't the top guess - useful to see "how close" it was
        true_idx = class_names.index(true_label) if true_label in class_names else None
        true_confidence = float(predictions[true_idx]) * 100 if true_idx is not None else None

        correct = predicted_label == true_label
        results.append({
            "true": true_label,
            "predicted": predicted_label,
            "confidence": confidence,
            "true_confidence": true_confidence,
            "correct": correct,
        })

    # ---- print the table ----
    print(f"\n{'True':<8}{'Predicted':<12}{'Confidence':<13}{'Correct?':<10}{'True label conf.'}")
    print("-" * 65)
    for r in results:
        mark = "YES" if r["correct"] else "NO"
        tc = f"{r['true_confidence']:.1f}%" if r["true_confidence"] is not None else "n/a"
        print(f"{r['true']:<8}{r['predicted']:<12}{r['confidence']:<12.1f}%{mark:<10}{tc}")

    # ---- summary ----
    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    print("\n" + "=" * 65)
    print(f"Overall: {correct_count}/{total} correct ({correct_count / total * 100:.1f}%)")

    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print("\nMisclassified letters (True -> Predicted):")
        for r in wrong:
            print(f"  {r['true']} -> {r['predicted']}  (predicted with {r['confidence']:.1f}% confidence)")
    else:
        print("\nAll classes correctly identified!")


if __name__ == "__main__":
    main()