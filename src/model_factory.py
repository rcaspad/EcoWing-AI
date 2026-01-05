"""Model factory for EcoWing-AI.

Provides `build_econet_dual(num_classes)` that builds a dual-branch model
using MobileNetV2 and EfficientNetB0 (both ImageNet-pretrained, frozen),
fuses features, and returns a compiled Keras model.
"""
from __future__ import annotations

from typing import Optional

import tensorflow as tf

try:
    from src.config import IMG_SIZE
except Exception:
    try:
        from .config import IMG_SIZE
    except Exception:
        import importlib.util
        from pathlib import Path

        cfg_path = Path(__file__).resolve().parents[1] / "src" / "config.py"
        if not cfg_path.exists():
            cfg_path = Path(__file__).resolve().parents[1] / "config.py"
        spec = importlib.util.spec_from_file_location("config", str(cfg_path))
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)  # type: ignore
        IMG_SIZE = config.IMG_SIZE


def build_econet_dual(num_classes: int) -> tf.keras.Model:
    """Builds and compiles the hybrid EcoNet dual-branch model.

    Architecture:
    - Input -> shared input
    - Branch A: MobileNetV2 (include_top=False, imagenet, frozen) -> GAP
    - Branch B: EfficientNetB0 (include_top=False, imagenet, frozen) -> GAP
    - Concatenate features -> Dense -> BatchNorm -> Dropout(0.5) -> Softmax output

    Args:
        num_classes: number of output classes.

    Returns:
        Compiled `tf.keras.Model`.
    """
    input_shape = (*IMG_SIZE, 3)
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # Backbone A: MobileNetV2
    mobilenet = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    mobilenet.trainable = False
    x_a = mobilenet(inputs, training=False)
    x_a = tf.keras.layers.GlobalAveragePooling2D(name="gap_mobilenetv2")(x_a)

    # Backbone B: EfficientNetB0
    efficient = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    efficient.trainable = False
    x_b = efficient(inputs, training=False)
    x_b = tf.keras.layers.GlobalAveragePooling2D(name="gap_efficientnetb0")(x_b)

    # Fusion
    x = tf.keras.layers.Concatenate(name="concat_features")([x_a, x_b])

    # Final dense block
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_final")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout_final")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EcoNetDual")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


__all__ = ["build_econet_dual"]
