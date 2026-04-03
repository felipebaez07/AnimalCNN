# 🐾 AnimalCNN — Clasificador de Animales con CNN

Red Neuronal Convolucional (CNN) para clasificar **15 tipos de animales**
usando PyTorch, FastAPI y Streamlit.

---

## 🐻 Animales que detecta
Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant,
Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra

---

## 🏗️ Arquitectura del modelo
- **Tipo:** CNN (Red Neuronal Convolucional)
- **Framework:** PyTorch
- **Capas:** 3 bloques Conv2D + ReLU + MaxPooling → Flatten → Dense(512) → Dense(15)
- **Input:** Imágenes 128x128 RGB
- **Salida:** 15 clases con probabilidades (softmax)
- **Método:** Aprendizaje supervisado — clasificación multiclase

---

## 📁 Estructura del proyecto
```
AnimalCNN/
├── app/
│   └── main.py          ← Backend FastAPI
├── models/
│   ├── cnn_arch.py      ← Arquitectura CNN
│   ├── trainer.py       ← Entrenamiento
│   └── saved/           ← Pesos del modelo entrenado (.pth)
├── utils/
│   ├── preprocess.py    ← Preprocesamiento de imágenes
│   └── inference.py     ← Predicción
├── ui/
│   └── app.py           ← Interfaz Streamlit
├── data/
│   └── animal_data/     ← Dataset de Kaggle (15 carpetas)
├── uploads/             ← Imágenes subidas por el usuario
└── requirements.txt
```

---

## ⚙️ Instalación y uso

### 1. Requisitos
- Python 3.12
- Git (opcional)

### 2. Crear entorno virtual
```bash
py -3.12 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Instalar dependencias
```bash
python -m pip install -r requirements.txt
```

### 4. Descargar el dataset de Kaggle
Necesitas una cuenta en kaggle.com y tu API key en `C:\Users\TU_USUARIO\.kaggle\kaggle.json`:
```json
{"username":"TU_USUARIO","key":"TU_API_KEY"}
```
Luego ejecuta:
```bash
python -c "import kaggle; kaggle.api.authenticate(); kaggle.api.dataset_download_files('likhon148/animal-data', path='data', unzip=True)"
```

### 5. Correr la aplicación

**Terminal 1 — Backend FastAPI:**
```bash
venv\Scripts\activate
python -m uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Frontend Streamlit:**
```bash
venv\Scripts\activate
python -m streamlit run ui/app.py
```

### 6. Abrir en el navegador
- **Streamlit:** http://localhost:8501
- **FastAPI docs:** http://localhost:8000/docs

---

## 🚀 Flujo de uso

1. **Entrenar:** En Streamlit selecciona "🏋️ Entrenar", ajusta épocas y haz clic en "Iniciar Entrenamiento"
2. **Predecir:** Selecciona "🔍 Predecir", sube una imagen y haz clic en "Clasificar"
3. El modelo muestra el animal detectado con su % de confianza y Top 3 predicciones

---

## 📦 Dependencias principales
```
fastapi          ← API REST backend
uvicorn          ← Servidor ASGI
streamlit        ← Interfaz web
torch            ← Framework de deep learning
torchvision      ← Transformaciones de imágenes
Pillow           ← Manejo de imágenes
requests         ← Comunicación Streamlit → FastAPI
scikit-learn     ← Utilidades ML
numpy / pandas   ← Manipulación de datos
matplotlib       ← Visualizaciones
jupyter          ← Notebooks de experimentación
```

---

## 📝 Notas
- El modelo se guarda automáticamente en `models/saved/animal_cnn.pth`
- Con 5 épocas se obtiene ~40% de precisión
- Con 20 épocas se obtiene ~65-70% de precisión
- El dataset NO se incluye en el ZIP por su tamaño — descárgalo con el comando de Kaggle
