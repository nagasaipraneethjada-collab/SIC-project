# Sign Language to Text Translator

A CNN-based ASL fingerspelling recognizer, built to match the concepts
taught in SIC AI Chapters 8 (Neural Networks) and 9 (CNNs).

## 1. Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 2. Data

Place your dataset folders here, so the structure looks like:

```
sign_language_translator/
├── asl_alphabet_train/
│   └── asl_alphabet_train/
│       ├── A/
│       ├── B/
│       ├── ...
│       ├── del/
│       ├── space/
│       └── nothing/
├── asl_alphabet_test/
├── train.py
├── predict.py
├── webcam_predict.py
└── requirements.txt
```

## 3. Train

```bash
python train.py
```

This will:
1. Load images from `asl_alphabet_train/asl_alphabet_train`, holding back 20% as a validation set
2. Build and train a CNN (3 convolution+pooling blocks, dense layer, dropout, softmax output)
3. Print progress per epoch (loss, accuracy, val_loss, val_accuracy)
4. Save a training curve chart to `training_history.png`
5. Print final validation accuracy
6. Save `sign_model.keras` and `class_labels.json`

## 4. Predict on a single image

```bash
python predict.py path/to/image.jpg
```

## 5. Live webcam demo

```bash
python webcam_predict.py
```

Put your hand in the green box. Press `q` to quit.

## Notes

- This is a static-image classifier - great for still hand poses (the alphabet).
- Trained on a clean-background dataset, so live webcam accuracy is best with a
  plain background and good lighting behind your hand.
- Every line of `train.py` is commented with references back to the specific
  SIC AI chapter/unit it corresponds to.
