# app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import logging

app = FastAPI(title="Animal CNN API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/")
def root():
    return {"message": "Animal CNN API is running"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen y retorna la predicción del animal.
    """
    try:
        from utils.preprocess import preprocess_image
        from utils.inference import predict as run_predict

        # Leer bytes de la imagen subida
        image_bytes = await file.read()

        # Preprocesar
        tensor = preprocess_image(image_bytes)

        # Predecir
        result = run_predict(tensor)

        return {
            "status":     "ok",
            "prediccion": result["prediccion"],
            "confianza":  result["confianza"],
            "top3":       result["top3"]
        }

    except FileNotFoundError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/train/")
async def train_model(
    epochs:     int = Form(10),
    batch_size: int = Form(32),
    lr:         float = Form(0.001)
):
    """
    Lanza el entrenamiento de la CNN con el dataset descargado.
    """
    try:
        from models.trainer import train

        logger.info(f"Iniciando entrenamiento: epochs={epochs}, batch_size={batch_size}, lr={lr}")
        model_path = train(
            data_dir   = "data/animal_data",
            epochs     = epochs,
            batch_size = batch_size,
            lr         = lr
        )
        return {"status": "ok", "saved_model": model_path}

    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}")
        return {"status": "error", "message": str(e)}