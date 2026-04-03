# 🐾 AnimalCNN — Clasificador de Animales con CNN

Red Neuronal Convolucional (CNN) para clasificar **15 tipos de animales**
usando PyTorch, FastAPI y Streamlit.

---

## 🎯 ¿Qué hace esta aplicación?

Permite entrenar una red neuronal con imágenes de animales y luego
clasificar cualquier imagen que el usuario suba, mostrando el animal
detectado con su porcentaje de confianza y un Top 3 de predicciones.

---

## 🧠 Conceptos de Machine Learning aplicados

| Concepto | Descripción |
|---|---|
| **Tipo de aprendizaje** | Supervisado |
| **Tarea** | Clasificación multiclase (15 clases) |
| **Arquitectura** | CNN — Red Neuronal Convolucional |
| **Framework** | PyTorch |
| **Optimizador** | Adam |
| **Función de pérdida** | CrossEntropyLoss |
| **Regularización** | Dropout 0.5 |
| **Data Augmentation** | Flip, Rotación, ColorJitter |

---

## 🏗️ Arquitectura de la CNN

```
Input (128x128 RGB)
    ↓
Conv2D(32 filtros) + ReLU + MaxPool  ← detecta bordes y texturas
    ↓
Conv2D(64 filtros) + ReLU + MaxPool  ← detecta formas
    ↓
Conv2D(128 filtros) + ReLU + MaxPool ← detecta patrones complejos
    ↓
Flatten → Dense(512) + ReLU + Dropout(0.5)
    ↓
Dense(15) + Softmax
    ↓
Salida: probabilidad de cada animal
```

---

## 🐻 Animales que detecta (15 clases)

Bear · Bird · Cat · Cow · Deer · Dog · Dolphin · Elephant ·
Giraffe · Horse · Kangaroo · Lion · Panda · Tiger · Zebra

---

## 📁 Estructura del proyecto

```
AnimalCNN/
├── app/
│   └── main.py          ← Backend FastAPI (endpoints /predict/ y /train/)
├── models/
│   ├── cnn_arch.py      ← Arquitectura CNN en PyTorch
│   ├── trainer.py       ← Lógica de entrenamiento
│   └── saved/           ← Pesos del modelo entrenado (.pth)
├── utils/
│   ├── preprocess.py    ← Preprocesamiento de imágenes
│   └── inference.py     ← Predicción con el modelo cargado
├── ui/
│   └── app.py           ← Interfaz web con Streamlit
├── data/
│   └── animal_data/     ← Dataset de Kaggle (se descarga aparte)
├── uploads/             ← Imágenes subidas por el usuario
├── requirements.txt     ← Dependencias del proyecto
└── README.md
```

---

## ⚙️ Instalación paso a paso

### Requisitos previos
- Python 3.12 instalado
- Cuenta en kaggle.com (gratis)

### 1. Crear entorno virtual
```bash
py -3.12 -m venv venv
venv\Scripts\activate
```

### 2. Instalar dependencias
```bash
python -m pip install -r requirements.txt
```

### 3. Descargar el dataset de Kaggle
Crea el archivo `C:\Users\TU_USUARIO\.kaggle\kaggle.json` con tu API key:
```json
{"username":"TU_USUARIO","key":"TU_API_KEY"}
```
La API key se obtiene en kaggle.com → Settings → API → Create New Token.

Luego descarga el dataset:
```bash
python -c "import kaggle; kaggle.api.authenticate(); kaggle.api.dataset_download_files('likhon148/animal-data', path='data', unzip=True)"
```

### 4. Correr la aplicación

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

### 5. Abrir en el navegador
- **App:** http://localhost:8501
- **API docs:** http://localhost:8000/docs

---

## 🚀 Cómo usar la aplicación

### Entrenar el modelo
1. Selecciona modo **🏋️ Entrenar**
2. Ajusta épocas (recomendado: 10-20)
3. Clic en **🚀 Iniciar Entrenamiento**
4. Espera a que termine (5-20 minutos según el PC)

### Clasificar una imagen
1. Selecciona modo **🔍 Predecir**
2. Sube una foto de cualquier animal
3. Clic en **🔬 Clasificar**
4. Ve el resultado con porcentaje de confianza y Top 3

---

## 📈 Precisión esperada según épocas

| Épocas | Precisión aproximada |
|---|---|
| 5 | ~40% |
| 10 | ~55% |
| 20 | ~65-70% |
| 30 | ~70-75% |

---

## 📦 Tecnologías usadas

| Tecnología | Uso |
|---|---|
| **PyTorch** | Entrenamiento e inferencia de la CNN |
| **FastAPI** | API REST backend |
| **Streamlit** | Interfaz web |
| **Torchvision** | Transformaciones y augmentation de imágenes |
| **Pillow** | Manejo de imágenes |
| **Kaggle API** | Descarga automática del dataset |

---

## 📈 Dataset
- **Fuente:** [Kaggle — Animal Data](https://www.kaggle.com/datasets/likhon148/animal-data)
- **Total imágenes:** ~3,000+
- **Clases:** 15 animales
- **Split:** 80% entrenamiento / 20% validación

---

## 👤 Autor
Felipe Baez — Juan Bonilla - Carlos Caceres

