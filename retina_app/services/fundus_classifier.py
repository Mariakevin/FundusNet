"""Learned fundus image classifier.

Binary classifier that distinguishes fundus images from non-fundus images.
Replaces hand-tuned heuristic thresholds with a trained model.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class FundusClassifier(nn.Module):
    """Lightweight binary CNN for fundus vs non-fundus classification.

    Uses EfficientNet-B0 backbone (frozen) with a linear probe head.
    Input: RGB image resized to 224x224.
    Output: probability of being a fundus image.
    """

    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Get the number of features from the backbone
        in_features = self.backbone.classifier[1].in_features

        # Replace classifier with binary head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone:
            # Freeze all layers except the classifier head
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True

        # Default transform
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 3, 224, 224) tensor

        Returns:
            (B, 2) logits, or (B,) probabilities if sigmoid=True

        """
        return self.backbone(x)

    def predict(self, image_tensor):
        """Predict fundus probability for a single image.

        Args:
            image_tensor: (1, 3, 224, 224) or (3, 224, 224) tensor

        Returns:
            dict with is_fundus (bool), confidence (float), probability (float)

        """
        self.eval()
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            logits = self.forward(image_tensor)
            probs = F.softmax(logits, dim=1)
            fundus_prob = probs[0, 1].item()  # class 1 = fundus

        return {
            "is_fundus": fundus_prob >= 0.5,
            "confidence": fundus_prob if fundus_prob >= 0.5 else 1 - fundus_prob,
            "probability": fundus_prob,
        }

    def predict_from_pil(self, pil_image):
        """Predict from a PIL Image.

        Args:
            pil_image: PIL.Image.Image

        Returns:
            dict with is_fundus, confidence, probability

        """
        from retina_app.services.transforms import TRANSFORM

        tensor = TRANSFORM(pil_image.convert("RGB"))
        return self.predict(tensor)

    def save(self, path):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_type": "efficientnet_b0",
                "num_classes": 2,
                "categories": ["non_fundus", "fundus"],
            },
            path,
        )

    @classmethod
    def load(cls, path, freeze_backbone=False):
        """Load model from checkpoint.

        Args:
            path: path to .pth checkpoint
            freeze_backbone: whether to freeze backbone layers

        Returns:
            FundusClassifier instance

        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(num_classes=2, freeze_backbone=freeze_backbone)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


def prepare_fundus_training_data(fundus_dir, non_fundus_dirs, output_dir):
    """Prepare training data for fundus classifier.

    Creates a directory structure suitable for training:
        output_dir/
            train/
                fundus/
                non_fundus/
            val/
                fundus/
                non_fundus/

    Args:
        fundus_dir: directory containing fundus images
        non_fundus_dirs: list of directories with non-fundus images
        output_dir: output directory for organized data

    """
    import random
    import shutil

    os.makedirs(os.path.join(output_dir, "train", "fundus"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "non_fundus"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "fundus"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "non_fundus"), exist_ok=True)

    # Collect fundus images
    fundus_images = []
    for root, _, files in os.walk(fundus_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                fundus_images.append(os.path.join(root, f))

    # Collect non-fundus images
    non_fundus_images = []
    for d in non_fundus_dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    non_fundus_images.append(os.path.join(root, f))

    random.seed(42)

    # Split 80/20
    for label, images in [("fundus", fundus_images), ("non_fundus", non_fundus_images)]:
        random.shuffle(images)
        split = int(0.8 * len(images))
        train_images = images[:split]
        val_images = images[split:]

        for i, src in enumerate(train_images):
            dst = os.path.join(output_dir, "train", label, f"{label}_{i:04d}.png")
            shutil.copy2(src, dst)

        for i, src in enumerate(val_images):
            dst = os.path.join(output_dir, "val", label, f"{label}_{i:04d}.png")
            shutil.copy2(src, dst)

    print(
        f"Fundus: {len(fundus_images)} images "
        f"({len(fundus_images) // 5} train, {len(fundus_images) - len(fundus_images) // 5} val)"
    )
    print(
        f"Non-fundus: {len(non_fundus_images)} images "
        f"({len(non_fundus_images) // 5} train, {len(non_fundus_images) - len(non_fundus_images) // 5} val)"
    )


def train_fundus_classifier(data_dir, output_path, epochs=10, lr=0.001, batch_size=16):
    """Train the fundus classifier.

    Args:
        data_dir: directory with train/ and val/ subdirectories
        output_path: path to save trained model
        epochs: number of training epochs
        lr: learning rate
        batch_size: batch size

    Returns:
        dict with training history

    """
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    # Data loading
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FundusClassifier(num_classes=2, freeze_backbone=True)
    model = model.to(device)

    # Only optimize classifier head
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    # Save
    model.save(output_path)
    print(f"Model saved to {output_path}")

    return history
