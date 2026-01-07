"""Script de entrenamiento Versión 2 del modelo EcoNet.

Carga datasets de PlantVillage (Tomato/Pepper) más flores,
construye el modelo dual EcoNet con arquitectura híbrida,
compila con Adam(0.0001) y entrena con 5 épocas.

Salida:
- Modelo: models/best_model_v2.keras
- Historial: models/history_v2.npy
"""
from __future__ import annotations

import sys
from pathlib import Path
import logging
import os
import argparse
import numpy as np
import tensorflow as tf

# Añadir el directorio raíz al path para imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset
from src.model_factory import build_econet_dual
from src.config import MODELS_DIR


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")


def ensure_dir(d: Path) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)


def main(epochs: int = 5) -> None:
    """Función principal de entrenamiento.
    
    Args:
        epochs: Número de épocas (default: 5)
    """
    print("\n" + "="*70)
    print("ENTRENAMIENTO DEL MODELO ECOWINGNET v2")
    print("="*70 + "\n")
    
    ensure_dir(Path(MODELS_DIR))

    # Cargar datos desde data/train y data/val
    logger.info("Cargando datasets desde data/train y data/val...")
    train_ds, val_ds, num_classes = load_dataset(
        train_dir='data/train',
        val_dir='data/val'
    )

    logger.info("Construyendo modelo EcoNet dual con %d clases...", num_classes)
    model = build_econet_dual(num_classes)

    # Compilar modelo
    logger.info("Compilando modelo...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    model.summary(print_fn=logger.info)

    # Callbacks
    models_dir_path = Path(MODELS_DIR)
    checkpoint_path = str(models_dir_path / "best_model_v2.keras")
    
    callbacks = [
        # Guardar solo el mejor modelo
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            mode="min",
        ),
        # Reducir learning rate cuando val_loss se estanca
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # Entrenar por N épocas
    logger.info("="*70)
    logger.info("Iniciando entrenamiento por %d épocas...", epochs)
    logger.info("="*70)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Guardar historial
    history_path = models_dir_path / "history_v2.npy"
    np.save(str(history_path), history.history, allow_pickle=True)
    logger.info("✓ Historial guardado en: %s", history_path)

    # Evaluar en validación
    logger.info("Evaluando en el conjunto de validación...")
    results = model.evaluate(val_ds, verbose=1)
    
    if len(results) >= 2:
        val_loss, val_acc = results[0], results[1]
        logger.info("Validation Loss: %.4f", val_loss)
        logger.info("Validation Accuracy: %.4f", val_acc)
    
    # Mensaje final
    print("\n" + "="*70)
    print("✓ ENTRENAMIENTO FINALIZADO EXITOSAMENTE")
    print("="*70)
    print(f"\n✓ Modelo guardado en: {checkpoint_path}")
    print(f"✓ Historial guardado en: {history_path}")
    print(f"✓ Épocas entrenadas: {epochs}")
    print(f"✓ Clases: {num_classes}")
    print("\nAhora puedes usar el modelo para inferencia:")
    print("  python src/inference.py --image <path> --model models/best_model_v2.keras")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo EcoWing v2")
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("TRAIN_EPOCHS", 5)),
        help="Número de épocas para entrenar (default: 5)"
    )
    args = parser.parse_args()
    main(epochs=args.epochs)
