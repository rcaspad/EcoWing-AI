"""Genera la matriz de confusión con orden de etiquetas consistente.

Lógica estricta para evitar 'Label Mismatch':
- Usa ImageDataGenerator con shuffle=False.
- y_true = generator.classes (orden consistente con filenames).
- class_indices = generator.class_indices; invertir para obtener labels por índice.
- Predicciones del modelo sobre el generator (mismo orden).
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    val_dir = project_root / "data" / "val"
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best_model_v2.keras"
    if not model_path.exists():
        print(f"Error: modelo no encontrado en {model_path}")
        return 2

    print(f"Cargando modelo desde: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    if not val_dir.exists():
        print(f"Error: data/val no existe en {val_dir}")
        return 3

    # Generador de imágenes del conjunto de validación (orden determinista)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
    test_generator = datagen.flow_from_directory(
        str(val_dir),
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    # Etiquetas verdaderas con el mismo orden que las imágenes
    y_true = test_generator.classes

    # Mapeo correcto de clases (nombre -> índice), invertido a (índice -> nombre)
    class_indices = test_generator.class_indices
    labels = [None] * len(class_indices)
    for name, idx in class_indices.items():
        labels[idx] = name

    # Predicciones en el mismo orden del generator
    print("Generando predicciones (orden estable)...")
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Matriz de confusión y accuracy
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"Accuracy (validación): {accuracy:.2%}")

    # Visualización
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.0)
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Confusion Matrix (Fixed Labels) - Val Acc: {accuracy:.2%}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    out_path = docs_dir / "confusion_matrix_v3_fixed.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Matriz de confusión guardada en: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
