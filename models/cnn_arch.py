# models/cnn_arch.py
import torch
import torch.nn as nn

class AnimalCNN(nn.Module):
    """
    CNN para clasificar 15 tipos de animales.
    
    Arquitectura:
    - 3 bloques Conv2D + ReLU + MaxPooling para extraer features de la imagen
    - Flatten para convertir a vector
    - 2 capas Dense (fully connected) para clasificar
    - Salida: 15 neuronas (una por animal) con softmax
    """
    def __init__(self, num_classes=15):
        super(AnimalCNN, self).__init__()

        # Bloque 1: detecta bordes y texturas simples
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 canales RGB → 32 filtros
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # reduce tamaño a la mitad
        )

        # Bloque 2: detecta formas más complejas
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32 → 64 filtros
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Bloque 3: detecta patrones de alto nivel (orejas, patas, etc.)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64 → 128 filtros
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Clasificador fully connected
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),  # imagen 128x128 → después de 3 maxpool = 16x16
            nn.ReLU(),
            nn.Dropout(0.5),                 # evita overfitting
            nn.Linear(512, num_classes)      # salida: 15 clases
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


# Lista de clases en orden alfabético (igual que las carpetas)
CLASSES = [
    "Bear", "Bird", "Cat", "Cow", "Deer",
    "Dog", "Dolphin", "Elephant", "Giraffe", "Horse",
    "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"
]