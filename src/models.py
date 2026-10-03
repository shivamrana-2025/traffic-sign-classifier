"""Two models: a baseline CNN and MobileNetV2 transfer learning.

Both take raw RGB pixels in [0, 255], so preprocessing lives inside the model
and inference code (e.g. the Streamlit app) stays simple.
Augmentation layers are only active during training. No horizontal flips,
because signs like "turn left" and "turn right" would swap meaning.
"""
from tensorflow import keras
from tensorflow.keras import layers

from .config import NUM_CLASSES


def augmentation():
    return keras.Sequential(
        [
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.2),
        ],
        name="augmentation",
    )


def build_baseline(size):
    inputs = keras.Input((size, size, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = augmentation()(x)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_mobilenet(size):
    base = keras.applications.MobileNetV2(
        input_shape=(size, size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = keras.Input((size, size, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)  # MobileNetV2 expects [-1, 1]
    x = augmentation()(x)
    x = base(x, training=False)  # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2"), base


def unfreeze_top(base, n_layers=30):
    """Fine-tuning: unfreeze the last n layers, but keep BatchNorm frozen."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
