"""Train retina classification models with configurable CLI arguments."""

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ── Reproducibility ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Train retina classification models")
parser.add_argument(
    "--dataset",
    default=os.path.join(SCRIPT_DIR, "retina_dataset"),
    help="Path to dataset directory (default: ./retina_dataset)",
)
parser.add_argument(
    "--output", default=SCRIPT_DIR, help="Path to output directory for model files (default: script dir)"
)
parser.add_argument(
    "--models",
    nargs="+",
    default=["efficientnet", "resnet", "squeezenet", "mobilenet", "convnext", "vit"],
    help="Models to train (default: all)",
)
parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
parser.add_argument("--image-size", type=int, default=224, help="Input image size (default: 224)")
parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
parser.add_argument("--scheduler-step", type=int, default=5, help="LR scheduler step size (default: 5)")
parser.add_argument("--scheduler-gamma", type=float, default=0.5, help="LR scheduler gamma (default: 0.5)")
args = parser.parse_args()

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if hasattr(torch, "use_deterministic_algorithms"):
    try:
        torch.use_deterministic_algorithms(True)
    except TypeError:
        pass

DATASET_PATH = args.dataset
OUTPUT_DIR = args.output

try:
    from retina_app.constants import CATEGORIES
except ImportError:
    CATEGORIES = ["Healthy", "Cataract", "Glaucoma", "Retina Disease"]
CLASS_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}

BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.lr
IMAGE_SIZE = args.image_size


class RetinaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        folder_mapping = {
            "1_normal": "Healthy",
            "2_cataract": "Cataract",
            "3_glaucoma": "Glaucoma",
            "4_Diabetic_Retinopathy": "Retina Disease",
        }

        for folder_name, class_name in folder_mapping.items():
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.exists(folder_path):
                class_idx = CLASS_TO_IDX[class_name]
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(folder_path, img_name)
                        self.samples.append((img_path, class_idx))

        print(f"Loaded {len(self.samples)} images")
        for cat in CATEGORIES:
            count = sum(1 for _, idx in self.samples if idx == CLASS_TO_IDX[cat])
            print(f"  {cat}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label


train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def create_model(model_name, num_classes):
    if model_name == "efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    elif model_name == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    elif model_name == "squeezenet":
        model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_channels
        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(in_features, num_classes)
        )
    elif model_name == "convnext":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = model.classifier[2].in_features
        model.classifier = nn.Sequential(
            nn.LayerNorm(in_features), nn.Flatten(), nn.Dropout(0.3), nn.Linear(in_features, num_classes)
        )
    elif model_name == "vit":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = model.heads.head.in_features
        model.heads = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    else:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_features, 1024), nn.Hardswish(), nn.Linear(1024, num_classes)
        )
    return model


def train_model(model_name):
    print(f"\n{'=' * 50}")
    print(f"Training {model_name}")
    print(f"{'=' * 50}")

    full_dataset = RetinaDataset(DATASET_PATH, transform=train_transform)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)

    indices = list(range(total_size))
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_dataset = RetinaDataset(DATASET_PATH, transform=train_transform)
    val_dataset = RetinaDataset(DATASET_PATH, transform=val_transform)

    from torch.utils.data import Subset

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() or 4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() or 4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(model_name, len(CATEGORIES))
    model = model.to(device)

    class_counts = [0] * len(CATEGORIES)
    for _, label_idx in full_dataset.samples:
        class_counts[label_idx] += 1
    total = sum(class_counts)
    class_weights = torch.tensor(
        [total / (len(CATEGORIES) * max(c, 1)) for c in class_counts],
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    training_log = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        scheduler.step()

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

        # Log epoch
        training_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "train_acc": round(train_acc, 4),
                "val_loss": round(avg_val_loss, 6),
                "val_acc": round(val_acc, 4),
                "lr": round(optimizer.param_groups[0]["lr"], 8),
            }
        )

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        models_dir = os.path.join(OUTPUT_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_path = os.path.join(models_dir, f"{model_name}_retinopathy.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": len(CATEGORIES),
                    "categories": CATEGORIES,
                    "model_type": model_name,
                    "seed": SEED,
                    "best_val_acc": best_val_acc,
                },
                save_path,
            )
            print(f"  Saved best model to {save_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1} (no improvement for {args.patience} epochs)")
                break

    # Save training log CSV
    logs_dir = os.path.join(OUTPUT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{model_name}_training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        writer.writeheader()
        writer.writerows(training_log)
    print(f"  Training log saved to {log_path}")

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
    print("Starting training on new dataset...")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Output path: {OUTPUT_DIR}")
    print(f"Categories: {CATEGORIES}")

    models_to_train = args.models

    results = {}
    for model_name in models_to_train:
        try:
            results[model_name] = train_model(model_name)
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback

            traceback.print_exc()
            results[model_name] = 0.0

    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    for model_name, acc in results.items():
        print(f"{model_name}: {acc:.2f}%")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
