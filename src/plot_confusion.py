from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from src.data_loader import load_dataset
except Exception:
    from .data_loader import load_dataset

try:
    from src.config import DATA_DIR, MODELS_DIR
except Exception:
    from .config import DATA_DIR, MODELS_DIR


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(MODELS_DIR) / "best_model.keras"
    if not model_path.exists():
        print(f"Error: modelo no encontrado en {model_path}")
        return 2

    print(f"Cargando modelo desde: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    print(f"Cargando conjunto de prueba desde: {DATA_DIR}")
    _, _, test_ds = load_dataset(DATA_DIR)

    y_true_parts = []
    for _, y in test_ds:
        y_true_parts.append(y.numpy())
    if len(y_true_parts) == 0:
        print("No se encontraron etiquetas en el dataset de prueba.")
        return 3
    y_true = np.concatenate(y_true_parts, axis=0)
    print("Generando predicciones sobre el conjunto de prueba...")
    preds = model.predict(test_ds)
    y_pred = np.argmax(preds, axis=1)
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.1)
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    out_path = docs_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Matriz de confusión guardada en: {out_path}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
