"""
Hand Gesture Recognition - Model Training
==========================================
Project : Hand Gesture Recognition for Hearing Impaired and Aphonic People
Method  : Binarization → Contour Detection → SIFT → CNN Classification
Guide   : Mrs S. Deepa (Assistant Prof-II)
Team    : B. Srirekha, S.P. Shanthinii, S. Sruthi — Final Year CSE-C

Mac Note: Uses tensorflow-macos + tensorflow-metal for Apple Silicon GPU.
          On Intel Mac, standard tensorflow works fine.

Usage:
    python train_model.py
    python train_model.py my_custom_dataset_folder
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (safe on Mac)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, # type: ignore
                                     Flatten, Dropout, BatchNormalization)
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, # type: ignore
                                        ReduceLROnPlateau)
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# ── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE   = 64
BATCH_SIZE = 32
EPOCHS     = 30
MODEL_PATH = "gesture_cnn_model.h5"


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 – BINARIZATION USING THRESHOLDING
# ─────────────────────────────────────────────────────────────────────────────
def binarize_image(image):
    """BGR → grayscale → binary using Otsu's thresholding."""
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    return binary


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 – CONTOUR DETECTION  (findContours)
# ─────────────────────────────────────────────────────────────────────────────
def detect_and_crop_contour(binary, original):
    """Find the largest hand contour and crop its bounding rectangle."""
    contours, _ = cv2.findContours(binary,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return original

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(original.shape[1] - x, w + 2 * pad)
    h = min(original.shape[0] - y, h + 2 * pad)

    return original[y:y + h, x:x + w]


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 – SIFT FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_sift_features(gray_image, n_keypoints=128):
    """Scale-Invariant Feature Transform — returns fixed-length descriptor."""
    sift = cv2.SIFT_create(nfeatures=n_keypoints)
    _, descriptors = sift.detectAndCompute(gray_image, None)

    if descriptors is None:
        return np.zeros((n_keypoints, 128), dtype=np.float32)

    if len(descriptors) < n_keypoints:
        pad = np.zeros((n_keypoints - len(descriptors), 128), dtype=np.float32)
        descriptors = np.vstack([descriptors, pad])
    else:
        descriptors = descriptors[:n_keypoints]

    return descriptors.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(image):
    """BGR image → binary → contour crop → resize → normalize → (64,64,1)"""
    binary  = binarize_image(image)
    cropped = detect_and_crop_contour(binary, image)
    gray    = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return (resized / 255.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(dataset_dir):
    """
    Expects folder layout:
        dataset/
            PEACE/      img1.jpg ...
            GOOD_LUCK/  img1.jpg ...
            ...
    Returns X (N,64,64,1), y_encoded, LabelEncoder.
    """
    X, y = [], []

    for label in sorted(os.listdir(dataset_dir)):
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            continue
        print(f"  Loading → {label}")
        for fname in os.listdir(label_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            img = cv2.imread(os.path.join(label_path, fname))
            if img is None:
                continue
            X.append(preprocess_image(img))
            y.append(label)

    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\n  Total samples : {len(X)}")
    print(f"  Classes       : {le.classes_.tolist()}")
    return X, y_enc, le


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 – CNN MODEL  (Conv → Pool → Dense)
# ─────────────────────────────────────────────────────────────────────────────
def build_cnn(num_classes):
    model = Sequential([
        # Block 1 — Convolutional + Pooling
        Conv2D(32, (3,3), activation='relu', padding='same',
               input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.4),

        # Dense classification layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train(dataset_dir="dataset"):
    print("=" * 55)
    print("  Hand Gesture Recognition — Training (Mac)")
    print("=" * 55)

    print("\n[1] Loading dataset ...")
    X, y, le = load_dataset(dataset_dir)
    num_classes = len(le.classes_)
    y_cat = to_categorical(y, num_classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}")

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(X_train)

    print("\n[2] Building CNN ...")
    model = build_cnn(num_classes)
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True,
                        monitor='val_accuracy', verbose=1),
        EarlyStopping(patience=8, restore_best_weights=True,
                      monitor='val_accuracy', verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=4,
                          monitor='val_loss', verbose=1),
    ]

    print("\n[3] Training ...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    np.save("gesture_classes.npy", le.classes_)
    print(f"\n  Model saved  → {MODEL_PATH}")
    print(f"  Classes saved → gesture_classes.npy")

    _plot_history(history)
    return model, le


def _plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'],     label='Train Acc')
    ax1.plot(history.history['val_accuracy'], label='Val Acc')
    ax1.set_title('Model Accuracy'); ax1.legend()
    ax2.plot(history.history['loss'],     label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss'); ax2.legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("  Plot saved → training_history.png")


if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    if not os.path.isdir(dataset_dir):
        print(f"\nERROR: '{dataset_dir}' folder not found.")
        print("Run collect_dataset.py first to build your dataset.\n")
        sys.exit(1)
    train(dataset_dir)
