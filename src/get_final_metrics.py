"""Script de auditoría para generar métricas finales del modelo EcoWing-AI.

Objetivo: Imprimir un reporte en formato Markdown listo para copiar al README.

Requisitos:
- Carga models/best_model.keras y lo evalúa contra el test_ds
- Mide el tamaño en MB de models/best_model.keras
- Mide el tamaño en MB de models/ecowing_quantized.tflite (si existe)
- Calcula el porcentaje de reducción de tamaño
- Imprime un reporte Markdown formateado
"""
from __future__ import annotations

import sys
from pathlib import Path
import os
import io
import tensorflow as tf

# Fix stdout encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add parent directory to path for imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, DATA_DIR
from src.data_loader import load_dataset


def bytes_to_mb(size_bytes: int) -> float:
    """Convierte bytes a megabytes."""
    return size_bytes / (1024 * 1024)


def main() -> int:
    """Genera reporte de métricas finales."""
    
    print("\n" + "=" * 70)
    print("🔍 AUDITORÍA DE MÉTRICAS FINALES - EcoWing-AI")
    print("=" * 70 + "\n")
    
    # 1. Cargar modelo
    model_path = MODELS_DIR / "best_model.keras"
    if not model_path.exists():
        print(f"❌ Error: modelo no encontrado en {model_path}")
        return 1
    
    print(f"📦 Cargando modelo desde: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    
    # 2. Cargar dataset de prueba
    print(f"📂 Cargando dataset desde: {DATA_DIR}")
    train_ds, val_ds, test_ds = load_dataset(DATA_DIR)
    
    # 3. Evaluar modelo en test set
    print("🧠 Evaluando modelo en test set...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    test_accuracy_percent = test_accuracy * 100
    
    print(f"✅ Evaluación completada")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy_percent:.2f}%")
    
    # 4. Obtener tamaño de archivo .keras
    keras_size_bytes = os.path.getsize(str(model_path))
    keras_size_mb = bytes_to_mb(keras_size_bytes)
    
    # 5. Obtener tamaño de archivo .tflite (si existe)
    tflite_path = MODELS_DIR / "ecowing_quantized.tflite"
    tflite_exists = tflite_path.exists()
    
    if tflite_exists:
        tflite_size_bytes = os.path.getsize(str(tflite_path))
        tflite_size_mb = bytes_to_mb(tflite_size_bytes)
        reduction_percent = ((keras_size_bytes - tflite_size_bytes) / keras_size_bytes) * 100
    else:
        tflite_size_mb = 0.0
        reduction_percent = 0.0
        print(f"⚠️  Archivo TFLite no encontrado en {tflite_path}")
    
    print(f"\n📊 Tamaños de archivo:")
    print(f"   Original (.keras):    {keras_size_mb:.2f} MB")
    if tflite_exists:
        print(f"   Edge TFLite (.tflite): {tflite_size_mb:.2f} MB")
        print(f"   Reducción:            {reduction_percent:.2f}%")
    
    # 6. Imprimir reporte en Markdown
    print("\n" + "=" * 70)
    print("📋 REPORTE MARKDOWN (copiar al README):")
    print("=" * 70 + "\n")
    
    markdown_report = f"""## 📊 Rendimiento Final
| Métrica | Valor |
| :--- | :--- |
| **Precisión (Test)** | {test_accuracy_percent:.2f}% |
| **Tamaño Original** | {keras_size_mb:.2f} MB |
| **Tamaño Edge (TFLite)** | {tflite_size_mb:.2f} MB |
| **Reducción** | {reduction_percent:.2f}% |
"""
    
    print(markdown_report)
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
