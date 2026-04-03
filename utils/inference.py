# utils/inference.py
import torch
import torch.nn.functional as F
from pathlib import Path
from models.cnn_arch import AnimalCNN

# Variable global para cachear el modelo cargado
_model = None
_classes = None


def get_model():
    """
    Carga el modelo entrenado una sola vez y lo guarda en memoria.
    Si ya está cargado no lo vuelve a cargar.
    """
    global _model, _classes

    if _model is None:
        model_path = Path("models/saved/animal_cnn.pth")

        if not model_path.exists():
            raise FileNotFoundError(
                "No se encontró el modelo entrenado. "
                "Primero entrena el modelo desde la sección Entrenar."
            )

        checkpoint = torch.load(model_path, map_location="cpu")
        _classes   = checkpoint["classes"]
        _model     = AnimalCNN(num_classes=len(_classes))
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.eval()
        print(f"Modelo cargado. Clases: {_classes}")

    return _model, _classes


def predict(image_tensor: torch.Tensor) -> dict:
    """
    Recibe un tensor de imagen shape (1, 3, 128, 128) y retorna:
    - clase predicted (nombre del animal)
    - probabilidad de esa clase
    - top 3 predicciones con sus probabilidades
    """
    model, classes = get_model()

    with torch.no_grad():
        outputs = model(image_tensor)
        probs   = F.softmax(outputs, dim=1)[0]  # convierte logits a probabilidades

    # Top 3 predicciones
    top3_probs, top3_idx = torch.topk(probs, 3)

    top3 = []
    for prob, idx in zip(top3_probs.tolist(), top3_idx.tolist()):
        top3.append({
            "animal":       classes[idx],
            "probabilidad": round(prob * 100, 2)
        })

    return {
        "prediccion":   top3[0]["animal"],
        "confianza":    top3[0]["probabilidad"],
        "top3":         top3
    }   