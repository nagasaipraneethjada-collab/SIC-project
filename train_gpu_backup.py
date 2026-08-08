"""
train.py
--------
Trains a Convolutional Neural Network (CNN) to classify sign-language
images into classes (A-Z, del, space, nothing).

"""

import json

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# =====================================================================
# 1. HYPERPARAMETERS
# ---------------------------------------------------------------------
# Your course (Ch.8, Unit 03) calls these "hyperparameters" - the
# settings WE choose before training starts, as opposed to the weights
# the network learns on its own during training.
# =====================================================================
TRAIN_DIR = "asl_alphabet_train/asl_alphabet_train"  # one sub-folder per class

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32     # number of images shown to the network before each weight update
EPOCHS = 20          # number of full passes through the training data
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
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =====================================================================
# 3. BUILD THE CNN (Ch.9, Unit 01: "Building CNN using TensorFlow")
# =====================================================================
model = models.Sequential([

    # ---- Input layer ----
    # Every image is 128 (height) x 128 (width) x 3 (Red, Green, Blue channels)
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # Pixel values normally range 0-255. Rescaling to 0-1 helps
    # gradient descent (Ch.8 Unit 01) converge faster and more reliably.
    layers.Rescaling(1.0 / 255),

    # ---- Data augmentation ----
 
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
    # Conv2D: slides small learnable kernels (filters) across the image
    # to produce "feature maps" - this is the convolution operation
    # covered in Ch.9 Unit 01 (1.2 Components of CNN).
    # 32 filters = 32 different feature maps learned at this layer.
    # activation="relu" applies ReLU(x) = max(0, x), the activation
    # function from Ch.8 Unit 01 (1.5 Activation Function).
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

    # ---- Fully connected (Dense) layers ----
    # Flatten: the slides note "the input of the dense layer is rank 2,
    # i.e. [batch_size x number of input units]" - Flatten reshapes our
    # 3D feature maps into that required 2D shape.
    layers.Flatten(),

    # A standard fully-connected (dense) layer, same as a DMLP layer
    # from Ch.8. ReLU activation again.
    layers.Dense(128, activation="relu"),

    # Dropout (Ch.9 Unit 01, 1.3): randomly disables 40% of neurons
    # during training only, to reduce overfitting - the model can't
    # rely too heavily on any single neuron.
    layers.Dropout(0.4),

    # Output layer: one neuron per class (29 total), with softmax
    # activation (Ch.8 Unit 01, 1.5) converting raw scores into a
    # probability distribution that sums to 1 across all classes.
    layers.Dense(num_classes, activation="softmax"),
])

# =====================================================================
# 4. DEFINE THE OPTIMIZER, THEN COMPILE (Ch.8, Unit 03: "AI with Keras")
# =====================================================================
optimizer = optimizers.Adam()

# Loss function: measures how wrong a prediction is.
# Ch.9 Unit 01 explains: "Multi-class classification using integer
# (sparse) label (not one-hot encoded label) uses
# SparseCategoricalCrossentropy" - which matches label_mode="int" above.
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
# 6. PLOT THE TRAINING HISTORY (Ch.8, Unit 03 style chart)
# ---------------------------------------------------------------------
# Your slides show exactly this: Train vs Validation loss, and Train
# vs Validation accuracy, both plotted against Epoch. `history` (saved
# from model.fit above) holds these numbers for every epoch.
# =====================================================================
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
# model.evaluate() runs the fully-trained model against val_ds one
# more time, cleanly. Per your slides: "Returns the 0 = loss value and
# 1 = metrics value" - that's why we unpack it as (loss, acc) below.
# =====================================================================
test_results = model.evaluate(val_ds, verbose=1)
print("\n test accuracy {:.2f}%".format(test_results[1] * 100))

model.save(MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")
print(f"Class labels saved to {LABELS_OUT}")