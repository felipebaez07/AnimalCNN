# utils/preprocess.py
import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# ── Transformación para inferencia (igual que val_transforms) ─────────────────
inference_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    """
    Recibe los bytes de una imagen subida por el usuario y la preprocesa
    para que la CNN pueda procesarla.
    
    Pasos:
    1. Abre la imagen desde bytes
    2. Convierte a RGB (por si es PNG con canal alpha o escala de grises)
    3. Aplica resize y normalización
    4. Agrega dimensión de batch → shape (1, 3, 128, 128)
    
    Retorna un tensor listo para pasar al modelo.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = inference_transforms(img)
    tensor = tensor.unsqueeze(0)  # agrega dimensión batch: (3,128,128) → (1,3,128,128)
    return tensor