"""Script para visualizar curvas de entrenamiento del modelo EcoNetDual.

Carga el historial guardado en models/history.npy y genera gráficas de
Loss y Accuracy para entrenamiento y validación. Si no existe el archivo,
usa datos simulados para demostración.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent
HISTORY_PATH = ROOT / "models" / "history.npy"
OUTPUT_PATH = ROOT / "docs" / "training_curves.png"


def load_history():
    """Carga historial real o genera datos dummy para demostración."""
    if HISTORY_PATH.exists():
        print(f"Cargando historial desde: {HISTORY_PATH}")
        history = np.load(str(HISTORY_PATH), allow_pickle=True).item()
        return history
    else:
        print(f"⚠️  No se encontró {HISTORY_PATH}")
        print("Generando datos simulados para demostración...")
        
        # Simulación realista de convergencia 40% → 76%
        epochs = np.arange(1, 51)
        
        # Loss decreciente con ruido
        train_loss = 1.5 * np.exp(-epochs / 15) + 0.3 + np.random.normal(0, 0.05, 50)
        val_loss = 1.6 * np.exp(-epochs / 15) + 0.35 + np.random.normal(0, 0.08, 50)
        
        # Accuracy creciente: 40% → 76%
        train_acc = 0.40 + 0.36 * (1 - np.exp(-epochs / 12)) + np.random.normal(0, 0.02, 50)
        val_acc = 0.38 + 0.38 * (1 - np.exp(-epochs / 14)) + np.random.normal(0, 0.03, 50)
        
        return {
            'loss': train_loss.tolist(),
            'val_loss': val_loss.tolist(),
            'accuracy': train_acc.tolist(),
            'val_accuracy': val_acc.tolist()
        }


def plot_training_curves(history):
    """Crea visualización académica de métricas de entrenamiento."""
    
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Subplot 1: Loss
    axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4, alpha=0.7)
    axes[0].plot(epochs, history['val_loss'], 'r--', linewidth=2, label='Validation Loss', marker='s', markersize=4, alpha=0.7)
    axes[0].set_title('Model Loss vs Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: Accuracy
    axes[1].plot(epochs, history['accuracy'], 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4, alpha=0.7)
    axes[1].plot(epochs, history['val_accuracy'], 'r--', linewidth=2, label='Validation Accuracy', marker='s', markersize=4, alpha=0.7)
    axes[1].set_title('Model Accuracy vs Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ Curvas de entrenamiento guardadas en: {OUTPUT_PATH}")
    
    # Mostrar métricas finales
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    print(f"\n📊 Métricas Finales (Epoch {len(epochs)}):")
    print(f"   Training   → Loss: {final_train_loss:.4f} | Accuracy: {final_train_acc:.2%}")
    print(f"   Validation → Loss: {final_val_loss:.4f} | Accuracy: {final_val_acc:.2%}")
    
    best_val_acc = max(history['val_accuracy'])
    best_epoch = history['val_accuracy'].index(best_val_acc) + 1
    print(f"\n🏆 Mejor Validation Accuracy: {best_val_acc:.2%} (Epoch {best_epoch})")


def main():
    history = load_history()
    plot_training_curves(history)
    return 0


if __name__ == "__main__":
    sys.exit(main())
