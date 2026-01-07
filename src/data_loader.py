"""Cargador de datos para EcoWing-AI.

Provee `load_dataset()` que carga imágenes desde `data/train` y `data/val`,
aplica un pipeline de aumento con capas de Keras (`RandomFlip`, `RandomRotation`, 
`RandomContrast`) y devuelve `train_ds`, `val_ds`, `test_ds` y `num_classes`.

También genera un archivo `models/labels.txt` con los nombres de clases en orden
alfabético para usar en inferencia.
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


def get_class_names(data_dir: str | Path) -> list[str]:
    """Obtiene los nombres de las clases en orden alfabético.
    
    Args:
        data_dir: Directorio con subcarpetas de clases
        
    Returns:
        Lista de nombres de clases ordenados alfabéticamente
    """
    data_dir = Path(data_dir)
    class_dirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    return class_dirs


def save_labels(class_names: list[str], output_dir: str = 'models') -> Path:
    """Guarda los nombres de clases en models/labels.txt.
    
    Args:
        class_names: Lista de nombres de clases
        output_dir: Directorio donde guardar labels.txt
        
    Returns:
        Path al archivo labels.txt creado
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels_path = output_dir / 'labels.txt'
    
    with open(labels_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"Labels guardados en: {labels_path}")
    return labels_path


def load_dataset(
    train_dir: str | Path = 'data/train',
    val_dir: str | Path = 'data/val'
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Carga imágenes desde data/train y data/val.
    
    Args:
        train_dir: Directorio de entrenamiento (default: 'data/train')
        val_dir: Directorio de validación (default: 'data/val')
        
    Returns:
        Tupla (train_ds, val_ds, num_classes)
    """
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    
    # Validar directorios
    if not train_dir.exists():
        raise FileNotFoundError(f"data/train no existe: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"data/val no existe: {val_dir}")
    
    # Obtener nombres de clases desde data/train
    class_names = get_class_names(train_dir)
    num_classes = len(class_names)
    
    if num_classes == 0:
        raise ValueError(f"No se encontraron clases en {train_dir}")
    
    print(f"Clases encontradas ({num_classes}): {', '.join(class_names)}")
    
    # Guardar labels para inferencia
    save_labels(class_names)
    
    # Cargar dataset de entrenamiento
    print(f"Cargando dataset de entrenamiento desde {train_dir}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        labels="inferred",
        label_mode="int",
        batch_size=1,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=42,
    )
    
    # Cargar dataset de validación
    print(f"Cargando dataset de validación desde {val_dir}...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(val_dir),
        labels="inferred",
        label_mode="int",
        batch_size=1,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=42,
    )
    
    # Preprocess (normalize) y remover batch dim extra
    train_ds = train_ds.map(_preprocess_image_batch, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(_preprocess_image_batch, num_parallel_calls=AUTOTUNE)
    
    # Augmentation solo para entrenamiento
    augmentation = get_augmentation_layer()
    def _augment(x, y):
        return augmentation(x), y
    
    train_ds = train_ds.map(_augment, num_parallel_calls=AUTOTUNE)
    
    # Batch y prefetch
    train_ds = _final_batch(train_ds, BATCH_SIZE, training=True)
    val_ds = _final_batch(val_ds, BATCH_SIZE, training=False)
    
    return train_ds, val_ds, num_classes


__all__ = ["load_dataset", "get_augmentation_layer", "get_class_names", "save_labels"]
