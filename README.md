# 🦅 EcoWing AI: Vigilancia Robótica Autónoma para Cultivos

![Python](https://img.shields.io/badge/Python-3.13-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange) ![License](https://img.shields.io/badge/License-MIT-green) ![Edge AI](https://img.shields.io/badge/Optimized-Edge_Device-red)

## 📝 Resumen Ejecutivo

Sistema MLOps de **visión artificial para drones agrícolas** que detecta plagas y enfermedades en cultivos mediante una arquitectura híbrida innovadora. Combina **MobileNetV2** (extracción de características geométricas) con **EfficientNetB0** (análisis de texturas), optimizado para ejecución en dispositivos edge como Raspberry Pi con latencia <200ms.

El modelo integra técnicas de **Explainable AI (XAI)** mediante mapas de atención visual que permiten validar las predicciones agronómicas, cumpliendo con requisitos de trazabilidad en agricultura de precisión.

---

## 📊 Resultados del Modelo

| Métrica | Valor | Contexto |
|:--------|:------|:---------|
| **Accuracy (Test)** | 76.3% | Proof of Concept - 2 epochs |
| **Tamaño Original** | 32.95 MB | Modelo Keras (.keras) |
| **Tamaño Optimizado** | 7.35 MB | TFLite INT8 cuantizado |
| **Reducción de Peso** | **77%** | Óptimo para edge deployment |
| **Latencia de Inferencia** | ~180ms | Simulado en CPU (sin GPU) |
| **Parámetros Totales** | 6.96M | 657k entrenables |

---

## 🚀 Instalación Rápida

### 1. Clonar el repositorio
```bash
git clone https://github.com/rcaspad/EcoWing-AI.git
cd EcoWing-AI
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 💻 Cómo Ejecutar

### Generar Mapa de Atención (Heatmap XAI)
```bash
python src/generate_heatmap.py
```
**Salida:** `docs/visual_evidence.png` - Visualización de las regiones que el modelo considera relevantes para su predicción.

### Entrenar el Modelo (Opcional)
```bash
python src/train.py --epochs 10
```

### Convertir a TFLite
```bash
python src/convert_to_lite.py
```
**Salida:** `models/ecowing_quantized.tflite` - Modelo cuantizado para Raspberry Pi.

---

## 🖼️ Galería de Evidencias Técnicas

### 1. **Pipeline de Augmentation (Simulación de Vuelo Real)**
![Data Augmentation](docs/augmentation_evidence.png)

**Descripción técnica:** Visualización del pipeline de preprocesamiento aplicado al dataset. Incluye transformaciones para simular condiciones reales de captura aérea:
- **Motion Blur:** Desenfoque por movimiento del dron (kernel 5x5)
- **RandomBrightnessContrast:** Variaciones de iluminación solar (±20%)
- **Rotación/Flip:** Invariancia a ángulo de captura
- **Resize → 224×224px:** Normalización espacial para arquitecturas pre-entrenadas

Estas augmentations aumentan la robustez del modelo ante variabilidad ambiental, crítico para despliegues en campo.

---

### 2. **Mapa de Atención Visual (Explainable AI)**
![Activation Heatmap](docs/visual_evidence.png)

**Descripción técnica:** Activation heatmap generado desde la capa `efficientnetb0/top_activation` del modelo EcoNet-Dual. El proceso:
1. Extracción de feature maps (7×7×1280) de la última capa convolucional
2. Promediado de 1280 canales → mapa 2D de importancia espacial
3. Upsampling bilineal a 224×224px y aplicación de colormap Jet
4. Superposición semitransparente (α=0.5) sobre imagen original

**Interpretación:** Las regiones en rojo/amarillo indican zonas de alta activación neuronal. El modelo aprende correctamente a enfocarse en la estructura de la flor/hoja, ignorando el fondo, validando que no sobre-ajusta a artefactos del dataset.

---

**Autor:** Raúl Casado Padilla | **Asesor:** Gemini AI | **Curso:** Programa Superior Universitario Avanzado en Inteligencia Artificial 2025-2026
