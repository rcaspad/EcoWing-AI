"""Script de entrenamiento maestro (versión rápida de prueba).

Carga datasets desde `src.data_loader`, construye el modelo dual
con `build_econet_dual`, compila con Adam(1e-4) y entrena 5 épocas.
Guarda el mejor modelo en `models/best_model.keras` y el historial
en `models/history.npy`.
"""
from __future__ import annotations

from pathlib import Path
import logging
import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.data_loader import load_dataset
from src.model_factory import build_econet_dual

try:
    from src.config import DATA_DIR, MODELS_DIR, BATCH_SIZE
except Exception:
    from .config import DATA_DIR, MODELS_DIR, BATCH_SIZE


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")


def ensure_dir(d: Path) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)


def extract_labels_from_dataset(ds: tf.data.Dataset) -> np.ndarray:
    labels = []
    for batch in ds.unbatch().as_numpy_iterator():
        # when unbatched, each element is (img, label)
        _, lbl = batch
        labels.append(int(lbl))
    return np.array(labels, dtype=np.int32)


def main(epochs: int = 2) -> None:
    ensure_dir(Path(MODELS_DIR))

    logger.info("Cargando datasets desde %s", DATA_DIR)
    train_ds, val_ds, test_ds = load_dataset(DATA_DIR)

    # Determine num_classes (tf_flowers has 5)
    num_classes = 5

    logger.info("Construyendo el modelo con %d clases...", num_classes)
    model = build_econet_dual(num_classes)

    # Re-compile according to requested hyperparameters
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
        run_eagerly=True,
    )

    model.summary(print_fn=logger.info)

    # Compute class weights from the unbatched train dataset
    logger.info("Calculando pesos de clase a partir del conjunto de entrenamiento...")
    y_train = extract_labels_from_dataset(train_ds)
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    logger.info("Class weights: %s", class_weight)

    # Callbacks
    models_dir_path = Path(MODELS_DIR)
    checkpoint_path = str(models_dir_path / "best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    # Train for specified epochs
    logger.info("Entrenando por %d épocas...", epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # Save history
    history_path = models_dir_path / "history.npy"
    np.save(str(history_path), history.history, allow_pickle=True)
    logger.info("Historial guardado en: %s", history_path)

    # Evaluate on test set
    logger.info("Evaluando en el conjunto de prueba...")
    results = model.evaluate(test_ds, verbose=2)
    # results: [loss, accuracy]
    if len(results) >= 2:
        test_acc = results[1]
    else:
        test_acc = None

    logger.info("Test Accuracy: %s", test_acc)
    print(f"Test Accuracy: {test_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("TRAIN_EPOCHS", 2)))
    args = parser.parse_args()
    main(epochs=args.epochs)
