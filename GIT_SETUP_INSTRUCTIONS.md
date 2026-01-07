# 📦 Instrucciones para Subir EcoWing AI a GitHub

## ✅ Pre-requisitos

### 1. Verificar que Git está instalado
```bash
git --version
```

Si no está instalado, descárgalo de: https://git-scm.com/downloads

### 2. Configurar tu identidad en Git (solo primera vez)
```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"
```

### 3. Crear el repositorio en GitHub
1. Ve a https://github.com/new
2. **Nombre del repositorio:** `EcoWing-AI`
3. **Descripción:** "Sistema de detección de plagas en cultivos mediante Edge AI"
4. **Visibilidad:** Public o Private (según prefieras)
5. ⚠️ **NO marques:** "Add a README file" (ya tienes uno)
6. ⚠️ **NO marques:** "Add .gitignore" (ya tienes uno)
7. Click en **"Create repository"**

---

## 🚀 Comandos para Inicializar Git (Ejecutar en orden)

### Paso 1: Navegar a la carpeta del proyecto
```bash
cd E:\EcoWing-AI
```

### Paso 2: Inicializar repositorio Git
```bash
git init
```
✅ Esto crea el repositorio local Git

---

### Paso 3: Añadir todos los archivos
```bash
git add .
```
✅ Añade todos los archivos **respetando el .gitignore**

---

### Paso 4: Verificar qué archivos se añadirán (OPCIONAL pero recomendado)
```bash
git status
```
✅ Verifica que:
- ✓ Se incluyen: `src/`, `models/*.tflite`, `docs/`, `README.md`, `requirements.txt`
- ✗ Se excluyen: `data/`, `models/*.keras`, `venv/`, `__pycache__/`

---

### Paso 5: Hacer el primer commit
```bash
git commit -m "Initial commit: EcoWing AI - Plant Disease Detection System"
```
✅ Guarda los cambios en el repositorio local

---

### Paso 6: Renombrar la rama a 'main'
```bash
git branch -M main
```
✅ Cambia el nombre de la rama de `master` a `main` (estándar actual)

---

### Paso 7: Conectar con el repositorio remoto de GitHub
```bash
git remote add origin https://github.com/<TU-USUARIO>/EcoWing-AI.git
```
⚠️ **IMPORTANTE:** Reemplaza `<TU-USUARIO>` con tu usuario de GitHub

**Ejemplo:**
```bash
git remote add origin https://github.com/rcaspad/EcoWing-AI.git
```

---

### Paso 8: Subir a GitHub
```bash
git push -u origin main
```
✅ Sube todos los archivos a GitHub

---

## 🎉 ¡Listo! Tu repositorio está en GitHub

Puedes verlo en: `https://github.com/<TU-USUARIO>/EcoWing-AI`

---

## 📋 Resumen de Archivos Incluidos/Excluidos

### ✅ Archivos que SÍ se subirán a GitHub:
- ✓ `src/*.py` (código fuente - 10 scripts)
- ✓ `models/*.tflite` (modelos optimizados - 7.35 MB c/u)
- ✓ `models/labels.txt` (etiquetas de clases)
- ✓ `models/*.npy` (historial de entrenamiento)
- ✓ `docs/*.png`, `docs/*.jpg` (evidencias visuales)
- ✓ `requirements.txt` (dependencias)
- ✓ `README.md` (documentación)
- ✓ `.gitignore` (configuración)

### ❌ Archivos que NO se subirán (protegidos por .gitignore):
- ✗ `venv/`, `env/`, `.env` (entorno virtual)
- ✗ `__pycache__/`, `*.pyc` (cache de Python)
- ✗ **`data/`** ⚠️ **MUY IMPORTANTE - GIGAS DE DATASETS**
- ✗ `models/*.keras` (modelos Keras - 32+ MB cada uno)
- ✗ `models/*.h5` (modelos pesados)
- ✗ `.vscode/` (configuración del IDE)
- ✗ `.DS_Store`, `Thumbs.db` (archivos del sistema)

---

## 🔄 Comandos Útiles para el Futuro

### Ver el estado de tus archivos
```bash
git status
```

### Añadir cambios nuevos
```bash
git add .
git commit -m "Descripción de los cambios"
git push
```

### Ver historial de commits
```bash
git log --oneline
```

### Crear una nueva rama
```bash
git checkout -b nombre-rama
```

### Clonar el repositorio en otra máquina
```bash
git clone https://github.com/<TU-USUARIO>/EcoWing-AI.git
```

---

## ⚠️ Notas Importantes

1. **Límite de GitHub:** 100 MB por archivo. Los archivos `.tflite` (7.35 MB) están OK.

2. **Si tienes archivos grandes** (>100 MB), considera usar:
   - **Git LFS** (Large File Storage): https://git-lfs.github.com/
   - **Google Drive/Dropbox:** Para datasets

3. **Datasets:** NUNCA subas la carpeta `data/` a GitHub. Compártelos por separado.

4. **Modelos `.keras`:** Si necesitas compartirlos:
   - Súbelos a Google Drive
   - Añade el link en el README.md

5. **Seguridad:** Nunca subas:
   - Claves API (`.env`)
   - Credenciales (`credentials.json`)
   - Tokens de acceso

---

## 🆘 Solución de Problemas

### Error: "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/<TU-USUARIO>/EcoWing-AI.git
```

### Error: "refusing to merge unrelated histories"
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Olvidé añadir algo al .gitignore
```bash
# Edita .gitignore y luego:
git rm -r --cached .
git add .
git commit -m "Update .gitignore"
git push
```

---

**¡Éxito con tu repositorio!** 🚀
