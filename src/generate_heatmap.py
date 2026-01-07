"""Generador de Mapas de Saliencia (Saliency Maps) para el modelo EcoWing AI.

Este script genera visualizaciones de interpretabilidad mediante mapas de saliencia,
una técnica que permite identificar qué regiones de una imagen tienen mayor influencia
en la predicción del modelo de clasificación.

Funcionamiento:
1. Carga el modelo entrenado (best_model_v2.keras)
2. Selecciona una imagen de validación (prioriza Late_blight o YellowLeaf)
3. Calcula gradientes de la clase predicha respecto a los píxeles de entrada
4. Genera un heatmap donde:
   - ROJO/AMARILLO: áreas de alta importancia para la predicción
   - AZUL/VERDE: áreas de baja importancia
5. Superpone el heatmap sobre la imagen original y guarda el resultado

Salida:
- Imagen generada: docs/visual_evidence_REAL.jpg

Uso:
    python src/generate_heatmap.py

Requiere:
- Modelo entrenado en models/best_model_v2.keras
- Imágenes en data/val/
"""
import numpy as np
import tensorflow as tf
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*70)
print("  🔬 GENERADOR DE MAPAS DE SALIENCIA (Saliency Maps)")
print("  Visualiza las regiones de la imagen más importantes para la predicción")
print("="*70)

print("\n[1/5] Cargando modelo...")
model = tf.keras.models.load_model('models/best_model_v2.keras')
print("      ✓ Modelo cargado")

print("\n[2/5] Buscando imagen de validación...")
val_dir = 'data/val'
found_img_path = None

# Priorizar Late_blight o YellowLeaf
for root, dirs, files in os.walk(val_dir):
    if 'Late_blight' in root or 'YellowLeaf' in root:
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                found_img_path = os.path.join(root, file)
                break
    if found_img_path:
        break

# Fallback: cualquier imagen
if not found_img_path:
    for root, dirs, files in os.walk(val_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                found_img_path = os.path.join(root, file)
                break
        if found_img_path:
            break

if not found_img_path:
    raise FileNotFoundError("No se encontró ninguna imagen en data/val")

print(f"      ✓ Imagen: {os.path.basename(found_img_path)}")

print("\n[3/5] Preprocesando imagen...")
img = tf.keras.preprocessing.image.load_img(found_img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0
print("      ✓ Normalizada a (224, 224)")

print("\n[4/5] Calculando Mapa de Saliencia...")
# Convertir a tensor y activar el seguimiento de gradientes
img_tensor = tf.Variable(img_array, dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(img_tensor)
    
    # Forward pass
    predictions = model(img_tensor, training=False)
    
    # Obtener la clase predicha y su score
    pred_class = tf.argmax(predictions[0])
    class_score = predictions[0, pred_class]

# Calcular gradientes respecto a la imagen de entrada
grads = tape.gradient(class_score, img_tensor)

# Procesar gradientes para crear el mapa de saliencia
# Tomar el valor absoluto y promediar a través de los canales RGB
grads_np = grads[0].numpy()
saliency_map = np.mean(np.abs(grads_np), axis=-1)

# Normalizar a [0, 1]
saliency_map = saliency_map / (np.max(saliency_map) + 1e-10)

print(f"      ✓ Clase predicha: {pred_class.numpy()}")
print(f"      ✓ Confianza: {class_score.numpy():.4f}")
print(f"      ✓ Mapa calculado")

print("\n[5/5] Generando visualización...")
# Cargar imagen original
img_cv2 = cv2.imread(found_img_path)
img_cv2 = cv2.resize(img_cv2, (224, 224))

# Convertir saliency map a heatmap con colores
saliency_uint8 = np.uint8(255 * saliency_map)
heatmap_color = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)

# Superponer heatmap sobre la imagen original
superimposed_img = cv2.addWeighted(img_cv2, 0.6, heatmap_color, 0.4, 0)

# Guardar
os.makedirs('docs', exist_ok=True)
output_path = 'docs/visual_evidence_REAL.jpg'
cv2.imwrite(output_path, superimposed_img)

print(f"      ✓ Imagen guardada: {output_path}")

print("\n" + "="*70)
print("  ✅ VISUALIZACIÓN COMPLETADA EXITOSAMENTE")
print("  ")
print("  El mapa muestra en ROJO/AMARILLO las áreas que más influyen en")
print("  la predicción del modelo, y en AZUL/VERDE las áreas menos relevantes.")
print("="*70)
