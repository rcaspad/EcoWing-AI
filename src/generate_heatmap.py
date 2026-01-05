from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2


def find_class_image(root: Path, class_name: str = "daisy") -> Path | None:
    target = root / class_name
    if target.exists() and target.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in target.rglob(ext):
                return p
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in root.rglob(ext):
            return p
    return None


def preprocess(img_bgr: np.ndarray, size=(224, 224)) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, size)
    img_norm = img_resized.astype("float32") / 255.0
    return img_norm


def to_colormap(norm_map: np.ndarray, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    m = np.uint8(255 * norm_map)
    colored = cv2.applyColorMap(m, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    data_raw = project_root / "data" / "raw"
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best_model.keras"
    if not model_path.exists():
        print(f"Error: modelo no encontrado en {model_path}")
        return 2

    print(f"Cargando modelo desde: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    sample = find_class_image(data_raw, "daisy")
    if sample is None:
        print(f"No se encontró imagen en {data_raw}")
        return 3

    print(f"Usando imagen: {sample}")
    img_bgr = cv2.imread(str(sample))
    if img_bgr is None:
        print("No se pudo leer la imagen.")
        return 4

    img_pre = preprocess(img_bgr)
    inp = np.expand_dims(img_pre, axis=0)

    try:
        eff_branch = model.get_layer("efficientnetb0")
        # Build a submodel using the efficientnet branch's own input -> top_activation output
        try:
            last_act = eff_branch.get_layer("top_activation")
            submodel = tf.keras.models.Model(inputs=eff_branch.input, outputs=last_act.output)
        except Exception:
            # Fallback: use the branch output tensor
            submodel = tf.keras.models.Model(inputs=eff_branch.input, outputs=eff_branch.output)
    except Exception as e:
        print("No se pudo localizar la rama 'efficientnetb0' en el modelo principal. Imprimiendo summary para depuración:")
        try:
            model.summary()
        except Exception:
            pass
        print("Excepción:", e)
        return 5

    features = submodel.predict(inp)
    if features.ndim != 4:
        print("Formato inesperado de features:", features.shape)
        return 6

    feat = features[0]
    heatmap = np.mean(feat, axis=-1)
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heat_resized = cv2.resize(heatmap, (224, 224))
    heat_col = to_colormap(heat_resized)

    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_resized = cv2.resize(orig_rgb, (224, 224))

    overlay = cv2.addWeighted(orig_resized, 0.6, heat_col, 0.4, 0)

    out_path = docs_dir / "visual_evidence.png"
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), overlay_bgr)

    print(f"Mapa de activación guardado en: {out_path}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
