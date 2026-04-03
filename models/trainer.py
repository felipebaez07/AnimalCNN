# models/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from .cnn_arch import AnimalCNN, CLASSES

# ── Transformaciones de imagen ────────────────────────────────────────────────
# Entrenamiento: con augmentation para que el modelo aprenda mejor
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),        # todas las imágenes al mismo tamaño
    transforms.RandomHorizontalFlip(),    # voltea horizontalmente al azar
    transforms.RandomRotation(15),        # rota hasta 15 grados
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # varía brillo/contraste
    transforms.ToTensor(),                # convierte a tensor [0,1]
    transforms.Normalize(                 # normaliza con media y std de ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validación: sin augmentation, solo resize y normalizar
val_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def train(data_dir="data/animal_data", epochs=10, batch_size=32, lr=0.001):
    """
    Entrena la CNN con las imágenes de animales.
    
    - data_dir: carpeta con subcarpetas por clase
    - epochs: número de vueltas completas al dataset
    - batch_size: imágenes por batch
    - lr: learning rate del optimizador Adam
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar dataset dividido en train/val (80/20)
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    
    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    print(f"Train: {train_size} imágenes | Val: {val_size} imágenes")
    print(f"Clases: {full_dataset.classes}")

    # Modelo, pérdida y optimizador
    model     = AnimalCNN(num_classes=len(full_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # ── Entrenamiento ──────────────────────────────────────────────
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # ── Validación ─────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs  = model(images)
                loss     = criterion(outputs, labels)
                val_loss    += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / train_size * 100
        val_acc   = val_correct   / val_size   * 100
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.1f}%")

        # Guardar el mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Path("models/saved/animal_cnn.pth")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes":          full_dataset.classes,
                "val_acc":          val_acc,
                "epoch":            epoch + 1
            }, save_path)
            print(f"  ✅ Mejor modelo guardado (val_acc={val_acc:.1f}%)")

    print(f"\nEntrenamiento completo. Mejor val_acc: {best_val_acc:.1f}%")
    return str(save_path)