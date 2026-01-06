"""Crea una cuadrícula 3x3 mostrando la imagen original y 8 variantes
aplicando las augmentaciones definidas en `src.data_loader`.

El resultado se guarda en `docs/augmentation_evidence.png`.
"""
from __future__ import annotations

from pathlib import Path
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Add parent directory to path for imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset, get_augmentation_layer


def motion_blur(image: tf.Tensor, kernel_size: int = 9, direction: str = "horizontal") -> tf.Tensor:
    """Simula un MotionBlur simple aplicando una convolución lineal horizontal o vertical.

    image: tensor HWC float32 [0,1]
    """
    img = tf.expand_dims(image, axis=0)  # 1,H,W,C
    c = img.shape[-1]
    if direction == "horizontal":
        k = tf.ones((kernel_size, 1, c, 1), dtype=tf.float32) / tf.cast(kernel_size, tf.float32)
    else:
        k = tf.ones((1, kernel_size, c, 1), dtype=tf.float32) / tf.cast(kernel_size, tf.float32)

    # depthwise conv: filter shape [H, W, in_channels, channel_multiplier]
    blurred = tf.nn.depthwise_conv2d(img, k, strides=[1, 1, 1, 1], padding="SAME")
    return tf.squeeze(blurred, axis=0)


def main() -> None:
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    data_dir = ROOT / "data" / "raw"
    if not data_dir.exists():
        print("data/raw/ no existe. Ejecuta src/download_data.py primero.")
        return

    # Load datasets (batched)
    train_ds, val_ds, test_ds = load_dataset(data_dir)

    # Take one image from train_ds
    for batch in train_ds.take(1):
        images, labels = batch
        img = images[0]  # tensor HWC float32
        break

    aug_layer = get_augmentation_layer()

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.ravel()

    # First subplot: original
    orig_np = img.numpy()
    axes[0].imshow(np.clip(orig_np, 0.0, 1.0))
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Create 8 augmented versions
    for i in range(1, 9):
        augmented = aug_layer(tf.expand_dims(img, axis=0), training=True)
        aug_img = tf.squeeze(augmented, axis=0)

        # Apply simulated motion blur randomly horizontal or vertical
        direction = random.choice(["horizontal", "vertical"])
        mb = motion_blur(aug_img, kernel_size=7, direction=direction)
        mb_np = mb.numpy()

        axes[i].imshow(np.clip(mb_np, 0.0, 1.0))
        axes[i].axis("off")

    fig.suptitle("Simulación de Captura desde Dron (Augmentation)", fontsize=16)
    plt.tight_layout()
    out_path = docs_dir / "augmentation_evidence.png"
    plt.savefig(out_path, dpi=150)
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
