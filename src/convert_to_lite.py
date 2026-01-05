from __future__ import annotations

import os
from pathlib import Path
import sys
import tensorflow as tf


def bytes_to_mb(n: int) -> float:
    return float(n) / (1024 ** 2)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    keras_path = models_dir / "best_model.keras"
    tflite_path = models_dir / "ecowing_quantized.tflite"

    if not keras_path.exists():
        print(f"Error: modelo Keras no encontrado en: {keras_path}")
        return 2

    print(f"Cargando modelo Keras desde: {keras_path}")
    model = tf.keras.models.load_model(str(keras_path))

    print("Inicializando TFLiteConverter desde el modelo Keras...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("Convirtiendo a TFLite (optimización por defecto / cuantización dinámica)...")
    tflite_model = converter.convert()

    os.makedirs(models_dir, exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    keras_size = os.path.getsize(keras_path)
    tflite_size = os.path.getsize(tflite_path)

    print("")
    print(f"Tamaño original (.keras): {bytes_to_mb(keras_size):.2f} MB")
    print(f"Tamaño optimizado (.tflite): {bytes_to_mb(tflite_size):.2f} MB")
    saved = keras_size - tflite_size
    print(f"Ahorro: {bytes_to_mb(saved):.2f} MB ({saved} bytes)")

    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
