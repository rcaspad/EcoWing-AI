"""Verificación rápida de arquitectura para EcoWing-AI.

Construye el modelo usando `build_econet_dual` y muestra `model.summary()`.
Usar para comprobar que las ramas y la concatenación se conectan correctamente.
"""
from __future__ import annotations

from src.model_factory import build_econet_dual


def main() -> None:
    # Usamos 3 clases como ejemplo; ajusta según tu dataset real.
    model = build_econet_dual(num_classes=3)
    model.summary()


if __name__ == "__main__":
    main()
