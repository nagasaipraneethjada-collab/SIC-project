
import json

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# =====================================================================
# GPU MEMORY SETUP
# ---------------------------------------------------------------------
# By default, TensorFlow tries to reserve a large chunk of GPU memory
# upfront. On a smaller GPU (like a 4GB laptop card), this can cause an
# out-of-memory error before training even really starts. Setting
# "memory growth" makes it allocate only what it actually needs,
# growing gradually instead of grabbing everything at once.
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass  # must be set before GPUs are initialized - safe to ignore if already set
if gpus:
    print(f"GPU memory growth enabled for: {[g.name for g in gpus]}")
else:
    print("No GPU detected - running on CPU.")

# =====================================================================
# 1. HYPERPARAMETERS
# =====================================================================
TRAIN_DIR = "asl_alphabet_train/asl_alphabet_train"  # one sub-folder per class

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16      # lowered from 32 - your GPU has only 4GB VRAM
EPOCHS = 15          # number of full passes through the training data
MODEL_OUT = "sign_model.keras"
LABELS_OUT = "class_labels.json"

# =====================================================================
# 2. LOAD AND SPLIT THE DATA
# =====================================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,                          # same seed on both calls -> no overlap between splits
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="int",                  # integer labels (0,1,2,...) -> matches "sparse" loss below
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# Save the class names so predict.py can turn "class 0" back into "A", etc.
with open(LABELS_OUT, "w") as f:
    json.dump(class_names, f)

# Let TensorFlow prefetch batches in the background while training runs,
# so the CPU isn't sitting idle waiting on disk reads.
#
# NOTE: .cache() with no filename caches in MEMORY. With 113,000 images
# at 128x128x3, that's roughly 22 GB - fine on plain CPU training (which
# happened to just barely work), but it caused GPU/WSL2 training to
# crash trying to fit that much into "pinned" memory. Caching to a file
# on disk instead gives the same speed benefit (skips re-decoding JPEGs
# every epoch) without needing to hold it all in memory at once.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache("train_cache").shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache("val_cache").prefetch(buffer_size=AUTOTUNE)

# =====================================================================
# 3. BUILD THE CNN ("Building CNN using TensorFlow")

# =====================================================================
model = models.Sequential([

    # ---- Input layer ----
    # Every image is 128 (height) x 128 (width) x 3 (Red, Green, Blue channels)
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # Pixel values normally range 0-255. Rescaling to 0-1 helps
    # gradient descent (Ch.8 Unit 01) converge faster and more reliably.
    layers.Rescaling(1.0 / 255),

    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.15),
    # IMPORTANT: value_range=(0, 1) must be set explicitly here, because
    # RandomBrightness defaults to assuming 0-255 input. Since we already
    # rescaled to 0-1 above, leaving the default caused every image to
    # get corrupted into near-blank noise - which is why the model was
    # stuck at exactly chance-level accuracy (loss = ln(29) = 3.367).
    layers.RandomBrightness(0.25, value_range=(0, 1)),
    layers.RandomContrast(0.25),
    layers.RandomTranslation(0.1, 0.1),

    # ---- Convolution block 1 ----
  
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),

    # MaxPooling2D: sub-samples the feature map (Ch.9 Unit 01, 1.2),
    # keeping the strongest signal in each 2x2 block and halving the
    # width/height. Reduces computation and adds some position tolerance.
    layers.MaxPooling2D(pool_size=(2, 2)),

    # ---- Convolution block 2 ----
    # More filters (64) here because after pooling, each unit represents
    # a larger, more complex region of the original image.
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # ---- Convolution block 3 ----
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    
    layers.Flatten(),

   
    layers.Dense(128, activation="relu"),

  
    layers.Dropout(0.4),

 
    layers.Dense(num_classes, activation="softmax"),
])

# =====================================================================
# 4. DEFINE THE OPTIMIZER, THEN COMPILE
# ---------------------------------------------------------------------

optimizer = optimizers.Adam()

# Loss function: measures how wrong a prediction is.

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# =====================================================================
# 5. TRAIN THE MODEL
# ---------------------------------------------------------------------
# verbose=1 shows progress for every epoch (per your slides: "verbose=0
# means no output, verbose=1 to view the epochs").
# =====================================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)

# =====================================================================
# 6. PLOT THE TRAINING HISTORY 
# ---------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history["loss"], label="Train loss")
ax1.plot(history.history["val_loss"], label="Validation loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training History - Loss")
ax1.legend()

ax2.plot(history.history["accuracy"], label="Train acc.")
ax2.plot(history.history["val_accuracy"], label="Validation acc.")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training History - Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("training_history.png")
print("Saved training curves to training_history.png")

# =====================================================================
# 7. EVALUATE ON THE VALIDATION SET AND SAVE THE MODEL
# ---------------------------------------------------------------------

test_results = model.evaluate(val_ds, verbose=1)
print("\n test accuracy {:.2f}%".format(test_results[1] * 100))

model.save(MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")
print(f"Class labels saved to {LABELS_OUT}")