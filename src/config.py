"""Configuración del proyecto usando pathlib.

Variables:
- BASE_DIR: raíz del proyecto
- DATA_DIR: data/raw
- MODELS_DIR: models/
- IMG_SIZE: (224, 224)
- BATCH_SIZE: 32
- EPOCHS: 20

Diseñado para funcionar en Windows y POSIX usando pathlib.Path.
"""
from pathlib import Path
from typing import Tuple


# Base del proyecto: subir dos niveles desde este archivo (src/ -> proyecto)
BASE_DIR: Path = Path(__file__).resolve().parents[1]

# Rutas de datos y modelos
DATA_DIR: Path = BASE_DIR / "data" / "raw"
MODELS_DIR: Path = BASE_DIR / "models"

# Parámetros de entrenamiento
IMG_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32
EPOCHS: int = 20


__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "IMG_SIZE",
    "BATCH_SIZE",
    "EPOCHS",
]
