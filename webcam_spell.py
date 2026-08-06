"""
webcam_spell.py
----------------
Live sign-to-text that builds up a full sentence, letter by letter -
like real fingerspelling.

How it works:
- Hold a sign steady in the green box for about 1 second -> that
  letter gets "typed" into the sentence at the bottom of the screen.
- Show the 'space' sign to add a space between words.
- Show the 'del' sign to backspace (delete the last letter).
- Move your hand away / show 'nothing' between letters so the same
  letter doesn't get typed over and over while your hand is still
  in frame.

Controls:
- 'c' : clear the whole sentence
- 'q' : quit

Run:
    python webcam_spell.py
"""

import json
import time

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "sign_model.keras"
LABELS_PATH = "class_labels.json"
IMG_HEIGHT = 128
IMG_WIDTH = 128

# how confident a prediction must be to count at all
CONFIDENCE_THRESHOLD = 70

# how many seconds the SAME sign must be held before it gets "typed"
HOLD_SECONDS = 1.0

# region of the frame the user should put their hand in (x1, y1, x2, y2)
ROI = (100, 100, 400, 400)


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_names = json.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    sentence = ""
    current_hold_label = None      # the sign we're currently "holding"
    hold_start_time = None         # when we started holding it
    already_typed = False          # stops one hold from typing repeatedly

    print("Hold a sign steady to type it. 'space' = space, 'del' = backspace.")
    print("Press 'c' to clear, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        x1, y1, x2, y2 = ROI
        roi = frame[y1:y2, x1:x2]

        # preprocess the same way as training
        img = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img.astype("float32"), axis=0)

        predictions = model.predict(img_array, verbose=0)[0]
        best_idx = int(np.argmax(predictions))
        confidence = float(predictions[best_idx]) * 100
        label = class_names[best_idx]

        # ---- "hold to type" logic ----
        if confidence >= CONFIDENCE_THRESHOLD and label != "nothing":
            if label == current_hold_label:
                # still holding the same sign - check how long
                held_for = time.time() - hold_start_time
                if held_for >= HOLD_SECONDS and not already_typed:
                    if label == "space":
                        sentence += " "
                    elif label == "del":
                        sentence = sentence[:-1]
                    else:
                        sentence += label
                    already_typed = True  # don't type again until sign changes
            else:
                # new sign started - reset the hold timer
                current_hold_label = label
                hold_start_time = time.time()
                already_typed = False
        else:
            # low confidence, or 'nothing' (hand moved away / reset)
            current_hold_label = None
            hold_start_time = None
            already_typed = False

        # ---- drawing ----
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # live prediction + progress toward "typing" it
        if current_hold_label and not already_typed:
            held_for = time.time() - hold_start_time
            progress = min(held_for / HOLD_SECONDS, 1.0)
            bar_width = int(progress * (x2 - x1))
            cv2.rectangle(frame, (x1, y2 + 10), (x1 + bar_width, y2 + 25), (0, 255, 0), -1)

        status_text = f"{label} ({confidence:.0f}%)" if confidence >= CONFIDENCE_THRESHOLD else "..."
        cv2.putText(
            frame, status_text, (x1, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
        )

        # the sentence built so far, along the bottom of the window
        cv2.rectangle(frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(
            frame, sentence, (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
        )

        cv2.imshow("Sign to Text - Spelling Mode", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence = ""

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal sentence: {sentence}")


if __name__ == "__main__":
    main()