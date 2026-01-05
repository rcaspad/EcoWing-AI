"""Descarga y organiza un dataset de ejemplo en `data/raw/`.

Intento 1: usar `tf.keras.utils.get_file` para descargar una versión
de PlantVillage si se proporciona `PLANTVILLAGE_URL`.

Fallback: descargar `tf_flowers` desde `tensorflow_datasets` y
extraer las imágenes a `data/raw/<class_name>/`.

Al final imprime el número de imágenes por clase.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import tensorflow as tf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Opcional: si tienes una URL fiable para PlantVillage, colócala aquí.
PLANTVILLAGE_URL = ""  # p.ej. "https://example.com/plantvillage_subset.zip"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_via_get_file(dest_dir: Path) -> bool:
    if not PLANTVILLAGE_URL:
        logger.info("No PLANTVILLAGE_URL configurada; saltando get_file intento.")
        return False

    try:
        logger.info("Intentando descargar PlantVillage desde URL con get_file...")
        zip_path = tf.keras.utils.get_file(
            fname="plantvillage.zip",
            origin=PLANTVILLAGE_URL,
            extract=True,
            cache_dir=str(dest_dir.parent),
        )
        logger.info("Descargado y extraído en: %s", zip_path)
        return True
    except Exception as e:
        logger.warning("Fallo al descargar con get_file: %s", e)
        return False


def download_tf_flowers(dest_dir: Path) -> Dict[str, int]:
    try:
        import tensorflow_datasets as tfds
    except Exception as e:
        raise RuntimeError(
            "tensorflow_datasets no está instalado. Ejecuta pip install tensorflow-datasets"
        ) from e

    logger.info("Descargando 'tf_flowers' desde tensorflow_datasets (fallback)...")
    ds, info = tfds.load("tf_flowers", split="train", with_info=True, as_supervised=True)

    # Try to get label names if available; otherwise use numeric labels
    label_feature = info.features["label"]
    label_names = getattr(label_feature, "names", None)

    counts: Dict[str, int] = {}
    ensure_dir(dest_dir)

    for img, label in tfds.as_numpy(ds):
        lbl = int(label)
        name = label_names[lbl] if label_names else str(lbl)
        cls_dir = dest_dir / name
        ensure_dir(cls_dir)
        idx = counts.get(name, 0)
        out_path = cls_dir / f"img_{idx:06d}.jpg"
        # img is a numpy array HWC uint8
        try:
            jpeg = tf.io.encode_jpeg(tf.convert_to_tensor(img))
            out_path.write_bytes(jpeg.numpy())
        except Exception:
            # fallback: use PIL
            from PIL import Image

            im = Image.fromarray(img)
            im.save(out_path)

        counts[name] = idx + 1

    return counts


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    raw_dir = base / "data" / "raw"
    ensure_dir(raw_dir)

    # Primero intento con get_file si hay URL configurada
    ok = download_via_get_file(raw_dir)
    counts: Dict[str, int] = {}
    if not ok:
        try:
            counts = download_tf_flowers(raw_dir)
        except RuntimeError as e:
            logger.error(str(e))
            logger.error("No se pudo descargar ningún dataset. Instala tensorflow-datasets o proporciona PLANTVILLAGE_URL.")
            return

    # Imprimir resumen
    total = sum(counts.values()) if counts else 0
    logger.info("Descarga organizada en: %s", raw_dir)
    logger.info("Total imágenes descargadas: %d", total)
    for cls, cnt in sorted(counts.items()):
        logger.info("Clase %s: %d imágenes", cls, cnt)


if __name__ == "__main__":
    main()
