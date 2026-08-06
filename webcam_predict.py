"""
webcam_predict.py
------------------
Live sign-to-text using your webcam. Same model.predict() forward
pass as predict.py, just run continuously on live video frames.

Press 'q' to quit.

Run:
    python webcam_predict.py
"""

import json

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "sign_model.keras"
LABELS_PATH = "class_labels.json"
IMG_HEIGHT = 128
IMG_WIDTH = 128
CONFIDENCE_THRESHOLD = 60  # only display a prediction above this %

# Region of the frame the user should put their hand in (x1, y1, x2, y2)
ROI = (100, 100, 400, 400)


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_names = json.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror, feels more natural
        x1, y1, x2, y2 = ROI
        roi = frame[y1:y2, x1:x2]

        # Same preprocessing as training: resize to 128x128, RGB order.
        # (Rescaling 0-1 is handled inside the model itself, since we
        # included a Rescaling layer when we built it in train.py.)
        img = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img.astype("float32"), axis=0)

        predictions = model.predict(img_array, verbose=0)[0]
        best_idx = int(np.argmax(predictions))
        confidence = float(predictions[best_idx]) * 100
        label = class_names[best_idx]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{label} ({confidence:.0f}%)" if confidence >= CONFIDENCE_THRESHOLD else "..."
        cv2.putText(
            frame, text, (x1, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
        )

        cv2.imshow("Sign to Text", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
