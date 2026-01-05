from __future__ import annotations

import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


def find_sample_image(raw_dir: Path) -> Path | None:
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in raw_dir.rglob(ext):
            return p
    return None


def preprocess_image(img_bgr: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_norm = img_resized.astype(np.float32) / 255.0
    return img_norm


def make_heatmap(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    data_raw = project_root / "data" / "raw"
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    keras_path = models_dir / "best_model.keras"
    if not keras_path.exists():
        print(f"Error: modelo no encontrado en: {keras_path}")
        return 2

    print(f"Cargando modelo desde: {keras_path}")
    model = tf.keras.models.load_model(str(keras_path))

    sample = find_sample_image(data_raw)
    if sample is None:
        print(f"No se encontró ninguna imagen en: {data_raw}")
        return 3

    print(f"Usando imagen de ejemplo: {sample}")
    img_bgr = cv2.imread(str(sample))
    if img_bgr is None:
        print(f"Error al leer la imagen: {sample}")
        return 4

    img_pre = preprocess_image(img_bgr)
    input_tensor = np.expand_dims(img_pre, axis=0)

    try:
        eff = model.get_layer("efficientnetb0")
        last_conv = eff.get_layer("top_activation")
    except Exception as e:
        print("No se pudo localizar la capa 'efficientnetb0/top_activation'. Imprimiendo summary de la submodelo efficientnetb0 para depuración:")
        try:
            eff = model.get_layer("efficientnetb0")
            eff.summary()
        except Exception:
            print("No se encontró la subcapa 'efficientnetb0' en el modelo principal.")
        print("Excepción:", e)
        return 5

    target_layer = None
    for layer in model.layers:
        if "top_activation" in layer.name:
            target_layer = layer
            break
    if target_layer is None:
        for layer in model.layers:
            if layer.name.startswith("efficientnetb0"):
                target_layer = layer
                break

    if target_layer is None:
        print("No se pudo localizar una capa de activación objetivo dentro del modelo principal.")
        return 6

    grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])

    img_tensor = tf.convert_to_tensor(input_tensor)
    img_tensor = tf.cast(img_tensor, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        print("Gradients are None; aborting.")
        return 6

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_outputs = conv_outputs[0].numpy()

    # Weight the channels by corresponding gradients
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    cam = np.mean(conv_outputs, axis=-1)
    heatmap = make_heatmap(cam)

    heatmap_resized = cv2.resize(heatmap, (img_pre.shape[1], img_pre.shape[0]))

    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_resized = cv2.resize(orig_rgb, (img_pre.shape[1], img_pre.shape[0]))

    overlay = cv2.addWeighted(orig_resized, 0.6, heatmap_resized, 0.4, 0)

    out_path = docs_dir / "gradcam_result.png"
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), overlay_bgr)

    print(f"Grad-CAM guardado en: {out_path}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
