"""Cargador de datos para EcoWing-AI.

Provee `load_dataset(data_dir)` que usa `tf.keras.utils.image_dataset_from_directory`
para cargar imágenes desde `data/raw/`, aplica un pipeline de aumento con capas
de Keras (`RandomFlip`, `RandomRotation`, `RandomContrast`) y devuelve tres
`tf.data.Dataset`: `train_ds`, `val_ds`, `test_ds` con la división 70/20/10.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf

try:
    from src.config import IMG_SIZE, BATCH_SIZE
except Exception:
    from .config import IMG_SIZE, BATCH_SIZE


AUTOTUNE = tf.data.AUTOTUNE


def _count_images(data_dir: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for p in data_dir.rglob("*") if p.suffix.lower() in exts and p.is_file())


def _preprocess_image_batch(img, label):
    # img: batch_size=1 image already resized by image_dataset_from_directory
    img = tf.cast(img, tf.float32) / 255.0
    # remove the batch dim (image_dataset_from_directory with batch_size=1 returns shape (1,H,W,C))
    img = tf.squeeze(img, axis=0)
    label = tf.squeeze(label, axis=0)
    return img, label


def _final_batch(ds: tf.data.Dataset, batch_size: int, training: bool = False) -> tf.data.Dataset:
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ],
        name="augmentation",
    )


def load_dataset(data_dir: str | Path) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Carga imágenes desde `data_dir` (espera subcarpetas por clase) y devuelve
    `train_ds`, `val_ds`, `test_ds` con particiones 70/20/10.

    El pipeline usa `image_dataset_from_directory` con `batch_size=1` para
    poder dividir por número de ejemplos y luego reagrupar por `BATCH_SIZE`.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir no existe: {data_dir}")

    total = _count_images(data_dir)
    if total == 0:
        raise ValueError(f"No se encontraron imágenes en {data_dir}")

    train_n = int(total * 0.7)
    val_n = int(total * 0.2)
    test_n = total - train_n - val_n

    # Load all examples with batch_size=1 to split deterministically
    ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        labels="inferred",
        label_mode="int",
        batch_size=1,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=42,
    )

    # Split datasets
    train_ds = ds.take(train_n)
    rest = ds.skip(train_n)
    val_ds = rest.take(val_n)
    test_ds = rest.skip(val_n)

    # Preprocess (normalize) and remove extra batch dim
    train_ds = train_ds.map(_preprocess_image_batch, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(_preprocess_image_batch, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(_preprocess_image_batch, num_parallel_calls=AUTOTUNE)

    # Augmentation only for training
    augmentation = get_augmentation_layer()
    def _augment(x, y):
        return augmentation(x), y

    train_ds = train_ds.map(_augment, num_parallel_calls=AUTOTUNE)

    # Batch and prefetch
    train_ds = _final_batch(train_ds, BATCH_SIZE, training=True)
    val_ds = _final_batch(val_ds, BATCH_SIZE, training=False)
    test_ds = _final_batch(test_ds, BATCH_SIZE, training=False)

    return train_ds, val_ds, test_ds


__all__ = ["load_dataset", "get_augmentation_layer"]
