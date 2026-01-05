"""Script para generar matriz de confusión del modelo EcoNetDual.

Evalúa el modelo entrenado sobre el conjunto de test y visualiza
la matriz de confusión con heatmap usando seaborn. Útil para
identificar clases con mayor confusión y ajustar el modelo.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Añadir el directorio raíz al path para imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset
from src.config import DATA_DIR, MODELS_DIR


def main() -> int:
    """Genera matriz de confusión sobre el conjunto de test."""
    
    project_root = Path(__file__).resolve().parents[1]
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar modelo entrenado
    model_path = Path(MODELS_DIR) / "best_model.keras"
    if not model_path.exists():
        print(f"❌ Error: modelo no encontrado en {model_path}")
        return 2

    print(f"📦 Cargando modelo desde: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    # 2. Cargar conjunto de prueba
    print(f"📂 Cargando conjunto de prueba desde: {DATA_DIR}")
    _, _, test_ds = load_dataset(DATA_DIR)

    # 3. Extraer etiquetas reales (y_true) del dataset
    print("🔍 Extrayendo etiquetas reales del test set...")
    y_true_parts = []
    for _, y in test_ds:
        y_true_parts.append(y.numpy())
    
    if len(y_true_parts) == 0:
        print("❌ No se encontraron etiquetas en el dataset de prueba.")
        return 3
    
    y_true = np.concatenate(y_true_parts, axis=0)
    
    # 4. Generar predicciones y convertir a etiquetas de clase
    print("🧠 Generando predicciones sobre el conjunto de prueba...")
    preds = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(preds, axis=1)  # Convertir probabilidades a clases
    
    print(f"✅ Predicciones completadas: {len(y_pred)} muestras")
    
    # 5. Calcular matriz de confusión
    print("📊 Calculando matriz de confusión...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Calcular accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"🎯 Accuracy en test set: {accuracy:.2%}")

    # 6. Visualizar con seaborn heatmap
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(
        cm, 
        annot=True,        # Mostrar números en cada celda
        fmt="d",           # Formato entero
        cmap="Blues",      # Paleta azul
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight='bold')
    ax.set_ylabel("True Class", fontsize=12, fontweight='bold')
    ax.set_title(f"Confusion Matrix - Test Accuracy: {accuracy:.2%}", 
                 fontsize=14, fontweight='bold', pad=20)

    # 7. Guardar imagen
    out_path = docs_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Matriz de confusión guardada en: {out_path}")
    
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
