# 🦅 EcoWing AI: Vigilancia Robótica Autónoma para Cultivos

![Python](https://img.shields.io/badge/Python-3.10-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange) ![License](https://img.shields.io/badge/License-MIT-green) ![Edge AI](https://img.shields.io/badge/Optimized-RaspberryPi-red)

> **Sistema de visión artificial para drones agrícolas capaz de detectar plagas en tiempo real (<200ms) mediante arquitecturas híbridas (MobileNet + EfficientNet).**

## 🌟 Características Clave
- **Arquitectura EcoNet-Dual:** Fusión de dos backbones (MobileNetV2 para geometría + EfficientNetB0 para textura).
- **Edge-Ready:** Modelo optimizado a **INT8 (.tflite)** reduciendo el peso en un **77% (de 33MB a 7MB)**.
- **XAI Integrado:** Mapas de atención visual para validar diagnósticos agronómicos.
- **Simulación de Vuelo:** Pipeline de datos robusto a desenfoque de movimiento y cambios de luz.

## 📊 Rendimiento (Test Set)
| Métrica | Valor |
| :--- | :--- |
| **Accuracy** | 76.3% (Proof of Concept - 2 Epochs) |
| **Inferencia** | ~180ms (Simulado en CPU) |
| **Tamaño Modelo** | 7.35 MB (Quantized) |

## 🛠️ Instalación y Uso
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/TU_USUARIO/EcoWing-AI.git
   cd EcoWing-AI
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar demo de inferencia (Genera heatmap):
   ```bash
   python src/generate_heatmap.py
   ```

## 📸 Evidencias Visuales
1. **Simulación de Dron (Data Augmentation)**  
   ![Augmentation](docs/augmentation_evidence.png)

2. **Explicabilidad del Modelo (Attention Map)**  
   ![Heatmap](docs/visual_evidence.png)

El modelo identifica correctamente la estructura relevante de la planta, ignorando el ruido de fondo.

---

**Autor:** Raúl Casado Padilla | **Asesor:** Gemini AI | **Curso:** Programa Superior Universitario Avanzado en Inteligencia Artificial 2025-2026
