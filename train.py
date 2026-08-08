

import json

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2

# =====================================================================
# GPU MEMORY SETUP
# =====================================================================
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass
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
BATCH_SIZE = 16
EPOCHS = 15
MODEL_OUT = "sign_model.keras"
LABELS_OUT = "class_labels.json"

# =====================================================================
# 2. LOAD AND SPLIT THE DATA
# =====================================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="int",
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

with open(LABELS_OUT, "w") as f:
    json.dump(class_names, f)

# NOTE: images stay as raw 0-255 pixels here - MobileNetV2 has its own
# specific preprocessing (below), different from the simple /255 we
# used in the from-scratch CNN. We do NOT rescale here.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache("train_cache").shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache("val_cache").prefetch(buffer_size=AUTOTUNE)

# =====================================================================
# 3. BUILD THE MODEL USING TRANSFER LEARNING
# =====================================================================

# ---- Data augmentation (same idea as before, same reasoning) ----
# Operates on raw 0-255 images here, so RandomBrightness's default
# value_range=(0,255) is actually correct this time - no explicit fix
# needed, unlike the earlier from-scratch version.
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.25),
    layers.RandomContrast(0.25),
    layers.RandomTranslation(0.1, 0.1),
])

# ---- Load the pretrained base model ----

base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights="imagenet",
)


base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    data_augmentation,

    layers.Rescaling(scale=1.0 / 127.5, offset=-1),

    # The pretrained feature extractor (frozen, not being retrained).
    base_model,

    # GlobalAveragePooling2D: condenses the base model's output feature
    # maps down to a single vector per image, ready for classification -
    # the standard way to connect a pretrained base to a new head.
    layers.GlobalAveragePooling2D(),

    # Our new, small classifier head - this is the part that actually
    # learns to tell ASL signs apart, built on top of the pretrained
    # general features.
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax"),
])

# =====================================================================
# 4. DEFINE THE OPTIMIZER, THEN COMPILE
# =====================================================================
optimizer = optimizers.Adam()

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# =====================================================================
# 5. TRAIN THE MODEL - PHASE 1 (frozen base, "feature extraction")
# ---------------------------------------------------------------------
# Since only our small classifier head is actually being trained (the
# base model is frozen), this should train noticeably faster per epoch
# than the from-scratch CNN, despite MobileNetV2 being a much bigger
# model overall.
# =====================================================================
print("\n=== PHASE 1: Training the new classifier head (base frozen) ===\n")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)

# =====================================================================
# 5b. PHASE 2: FINE-TUNING
# Phase 1 taught the new classifier head to use MobileNetV2's existing,
# generic features. Phase 2 goes a step further: unfreeze the TOP
# layers of the pretrained base (the more specialized, higher-level
# ones) and keep training them too, alongside the head - letting the
# model adjust its highest-level features specifically for ASL hand
# shapes, rather than the generic ImageNet objects they were originally
# trained on.
#
# Two important precautions, standard practice for fine-tuning:
#   1. Only unfreeze the TOP layers, not all of them - the early layers
#      detect very generic patterns (edges, colors, textures) that are
#      useful for almost any image task and don't need to change.
#   2. Use a MUCH lower learning rate than Phase 1. The pretrained
#      weights are already good - large updates here risk "catastrophic
#      forgetting" (wrecking the useful pretrained knowledge). A small
#      learning rate makes small, careful adjustments instead.
print("\n=== PHASE 2: Fine-tuning the top layers of MobileNetV2 ===\n")

base_model.trainable = True

# MobileNetV2 has ~154 layers total. Freeze everything except roughly
# the last 30 - keeping the early, generic feature detectors locked,
# only fine-tuning the later, more specialized ones.
FINE_TUNE_AT = len(base_model.layers) - 30
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

print(f"Fine-tuning the top {len(base_model.layers) - FINE_TUNE_AT} of "
      f"{len(base_model.layers)} base model layers.")

# Recompile is required any time trainable status changes - and we use
# a much smaller learning rate here (1e-5, vs Adam's default ~1e-3).
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

FINE_TUNE_EPOCHS = 10
total_epochs = EPOCHS + FINE_TUNE_EPOCHS

# initial_epoch continues the epoch count from where Phase 1 left off,
# so the history plot below reads as one continuous timeline.
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history1.epoch[-1] + 1,
    verbose=1,
)

# Combine both phases' histories into one, for a single continuous plot.
history = {}
for key in history1.history:
    history[key] = history1.history[key] + history2.history[key]

# =====================================================================
# 6. PLOT THE TRAINING HISTORY (both phases, with a marker between them)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["loss"], label="Train loss")
ax1.plot(history["val_loss"], label="Validation loss")
ax1.axvline(x=EPOCHS - 1, color="gray", linestyle="--", label="Fine-tuning starts")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training History - Loss")
ax1.legend()

ax2.plot(history["accuracy"], label="Train acc.")
ax2.plot(history["val_accuracy"], label="Validation acc.")
ax2.axvline(x=EPOCHS - 1, color="gray", linestyle="--", label="Fine-tuning starts")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training History - Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("training_history.png")
print("Saved training curves to training_history.png")

# =====================================================================
# 7. EVALUATE ON THE VALIDATION SET AND SAVE THE MODEL
# =====================================================================
test_results = model.evaluate(val_ds, verbose=1)
print("\n test accuracy {:.2f}%".format(test_results[1] * 100))

model.save(MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")
print(f"Class labels saved to {LABELS_OUT}")