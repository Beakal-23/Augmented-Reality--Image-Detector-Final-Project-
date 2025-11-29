#Importing the necessasry libraries -- Training a CNN from scrach on the pre-processed QickDraw dataset from Github
import os
import urllib.request
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#classes used: House cat, dog, car
CLASS_NAMES = [
    "cat", "dog", "bird", "fish", "bear", "butterfly", "bee", "spider",
    "house", "castle", "barn", "bridge", "lighthouse", "church",
    "car", "airplane", "bicycle", "train", "truck", "bus",
    "tree", "flower", "sun", "moon", "cloud",
    "apple", "banana", "book", "chair", "table", "cup", "umbrella",
    "face", "eye", "hand", "foot",
    "circle", "triangle", "square", "star"
]
NUM_CLASSES = len(CLASS_NAMES)

# Limit samples per class (dataset is huge; tune as needed)
MAX_ITEMS_PER_CLASS = 2000   # try 2000 for faster experiments

# Where to cache the downloaded .npy files and saved model
DATA_DIR = "quickdraw_npy"
MODEL_DIR = "saved_models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("TensorFlow version:", tf.__version__)
print("Classes:", CLASS_NAMES)

#Download helper -- uses pre processed numpy_bitmap dataset
def download_quickdraw_class(class_name: str) -> np.ndarray:
    """
    Download the preprocessed numpy bitmap (.npy) for one QuickDraw class
    from Google's public bucket, if not already cached.

    Each file contains an array of shape (N, 784) where 784 = 28*28.
    """
    # From the QuickDraw docs:
    # https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/<class>.npy
    url_class_name = class_name.replace(" ", "%20")
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{url_class_name}.npy"
    local_path = os.path.join(DATA_DIR, f"{class_name}.npy")

    if not os.path.exists(local_path):
        print(f"\nDownloading {class_name} data from:\n  {url}")
        urllib.request.urlretrieve(url, local_path)
        print("Download complete:", local_path)
    else:
        print(f"\nUsing cached file for {class_name}: {local_path}")

    data = np.load(local_path)
    print(f"{class_name}: {data.shape[0]} total samples available")
    return data

#Loading the data for all classes
all_images = []
all_labels = []

for label_idx, cls in enumerate(CLASS_NAMES):
    data = download_quickdraw_class(cls)

    # Optionally subsample for speed
    if MAX_ITEMS_PER_CLASS is not None:
        data = data[:MAX_ITEMS_PER_CLASS]

    # data shape: (N, 784)
    all_images.append(data)
    all_labels.append(np.full(data.shape[0], label_idx, dtype=np.int64))

# Stack into big arrays
X = np.concatenate(all_images, axis=0)   # (N_total, 784)
y = np.concatenate(all_labels, axis=0)   # (N_total,)

print("\nTotal samples:", X.shape[0])

#Preprocess: reshape & normalize
# (N, 784) -> (N, 28, 28, 1)
X = X.reshape((-1, 28, 28, 1)).astype("float32")

# Normalize pixel values from [0, 255] to [0, 1]
X = X / 255.0

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape, y_train.shape)
print("Val shape:  ", X_val.shape, y_val.shape)

#Visualize a few samples
def show_examples(X_batch, y_batch, class_names, n=9):
    plt.figure(figsize=(4, 4))
    idxs = np.random.choice(len(X_batch), n, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_batch[idx].squeeze(), cmap="gray")
        plt.title(class_names[y_batch[idx]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()
show_examples(X_train, y_train, CLASS_NAMES)

#Building a CNN model from scratch
def build_quickdraw_cnn(num_classes: int) -> tf.keras.Model:
    """
    Simple CNN built from scratch for QuickDraw 28x28 grayscale sketches.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


model = build_quickdraw_cnn(NUM_CLASSES)
model.summary()
# 🔹 Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 🔹 TRAINING SECTION (ADDED)
EPOCHS = 10        # you can increase to 20+ later
BATCH_SIZE = 64    # adjust based on GPU/CPU memory

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

#Plot training curves
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
# 🔹 Save the trained model
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

keras_path = os.path.join(MODEL_DIR, "quickdraw_house_cat_dog_car.keras")
h5_path    = os.path.join(MODEL_DIR, "quickdraw_house_cat_dog_car.h5")

model.save(keras_path)
model.save(h5_path)

print("\nModel saved to:")
print("  Keras format:", keras_path)
print("  H5 format:   ", h5_path)

import tf2onnx
import onnx
import onnxruntime

import tensorflow as tf
import tf2onnx
import os

# 1. Load model
model = tf.keras.models.load_model("saved_models/quickdraw_house_cat_dog_car.keras")

# 2. Monkey-patch missing attribute for tf2onnx (Keras 3 issue)
model.output_names = [t.name.split(":")[0] for t in model.outputs]

# 3. ONNX export -- Exporting the model for VR use
onnx_path = "saved_models/quickdraw_house_cat_dog_car.onnx"

spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=onnx_path,
)

print("ONNX model saved to:", onnx_path)
