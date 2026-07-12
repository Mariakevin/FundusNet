"""Contrastive Pretraining for Retinal Image Analysis.

Implements self-supervised contrastive learning with weak augmentations
specifically designed for medical/retinal images.

Based on research findings:
- Weak augmentations work better than strong augmentations for medical images
- Contrastive pretraining improves downstream classification with limited labels
- Dense latent space formation from weak augmentations

Reference: GMS-JIGNet (2025) - Self-supervised contrastive learning for
automated detection of diabetic retinopathy from fundus images.
"""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger("retina_app")


# ── Weak Augmentations for Medical Images ─────────────────────────────────────
# Research shows weak augmentations work better for medical images
# (preserve clinical features while providing contrastive signal)
MEDICAL_WEAK_AUGMENTATION = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    ]
)

MEDICAL_STRONG_AUGMENTATION = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
    ]
)


class ContrastiveRetinaDataset(Dataset):
    """Dataset that returns two augmented views of the same image for contrastive learning."""

    def __init__(self, root_dir, weak_transform=None, strong_transform=None, image_size=224):
        self.root_dir = root_dir
        self.weak_transform = weak_transform or MEDICAL_WEAK_AUGMENTATION
        self.strong_transform = strong_transform or MEDICAL_STRONG_AUGMENTATION
        self.image_size = image_size
        self.samples = []

        from retina_app.utils import CLASS_TO_IDX

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
                        self.samples.append((os.path.join(folder_path, img_name), class_idx))

        logger.info(f"Contrastive dataset loaded: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        from PIL import Image

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))

        # Base transform: resize + normalize
        base_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Two augmented views
        view1 = base_transform(self.weak_transform(image))
        view2 = base_transform(self.strong_transform(image))

        return view1, view2, label


# ── Contrastive Loss Functions ────────────────────────────────────────────────
class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR).

    Computes contrastive loss between pairs of augmented views.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features_i, features_j):
        """Compute NT-Xent loss.

        Args:
            features_i: [batch_size, feature_dim] from view 1
            features_j: [batch_size, feature_dim] from view 2

        Returns:
            Scalar loss
        """
        batch_size = features_i.shape[0]

        features_i = F.normalize(features_i, dim=1)
        features_j = F.normalize(features_j, dim=1)

        features = torch.cat([features_i, features_j], dim=0)

        similarity = torch.mm(features, features.t()) / self.temperature

        # Mask out self-similarity (diagonal)
        diag_mask = torch.eye(2 * batch_size, device=features.device).bool()
        similarity.masked_fill_(diag_mask, -1e9)

        # For each anchor in view 1, positive is at index i + batch_size
        # For each anchor in view 2, positive is at index i
        pos_indices = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size),
                torch.arange(0, batch_size),
            ]
        ).to(features.device)

        # Numerator: similarity to positive pair
        pos_sim = similarity[torch.arange(2 * batch_size), pos_indices]

        # Denominator: log-sum-exp over all non-self pairs
        log_denom = torch.logsumexp(similarity, dim=1)

        loss = (-pos_sim + log_denom).mean()
        return loss


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins redundancy reduction loss.

    Encourages invariant representations while reducing redundancy.
    """

    def __init__(self, lambda_param=5e-3):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, features_i, features_j):
        """Compute Barlow Twins loss.

        Args:
            features_i: [batch_size, feature_dim] from view 1
            features_j: [batch_size, feature_dim] from view 2

        Returns:
            Scalar loss
        """
        batch_size = features_i.shape[0]

        # Normalize features
        features_i = F.normalize(features_i, dim=0)
        features_j = F.normalize(features_j, dim=0)

        # Compute cross-correlation matrix
        c = torch.mm(features_i.t(), features_j) / batch_size

        # On-diagonal should be 1, off-diagonal should be 0
        on_diag = (c.diag() - 1).pow(2).sum()
        off_diag = c.pow(2).sum() - c.diag().pow(2).sum()

        loss = on_diag + self.lambda_param * off_diag

        return loss


# ── Contrastive Pretraining Module ────────────────────────────────────────────
class ContrastivePretrainer:
    """Performs self-supervised contrastive pretraining on retinal images.

    Uses weak augmentations (research shows these work better for medical images)
    to learn meaningful representations before supervised fine-tuning.
    """

    def __init__(
        self,
        model,
        feature_dim=256,
        temperature=0.07,
        loss_type="nt_xent",
        lambda_barlow=5e-3,
    ):
        self.model = model

        # Projector MLP (maps model features to contrastive space)
        # Get feature dimension from model
        if hasattr(model, "head") and hasattr(model.head, "in_features"):
            in_features = model.head.in_features
        elif hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
            in_features = model.classifier.in_features
        else:
            in_features = 1024  # Default for most architectures

        self.projector = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, feature_dim),
        )

        # Loss function
        if loss_type == "barlow":
            self.criterion = BarlowTwinsLoss(lambda_param=lambda_barlow)
        else:
            self.criterion = NTXentLoss(temperature=temperature)

    def pretrain(
        self,
        train_loader,
        num_epochs=100,
        lr=1e-4,
        weight_decay=1e-4,
        device="cpu",
    ):
        """Run contrastive pretraining.

        Args:
            train_loader: DataLoader with ContrastiveRetinaDataset
            num_epochs: Number of pretraining epochs
            lr: Learning rate
            weight_decay: Weight decay
            device: Device to train on

        Returns:
            Dictionary with training history
        """
        self.model.to(device)
        self.projector.to(device)

        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        history = {"loss": []}

        for epoch in range(num_epochs):
            self.model.train()
            self.projector.train()

            epoch_loss = 0.0
            num_batches = 0

            for view1, view2, _ in train_loader:
                view1, view2 = view1.to(device), view2.to(device)

                # Forward pass through model
                features1 = self.model(view1)
                if isinstance(features1, tuple):
                    features1 = features1[0]

                features2 = self.model(view2)
                if isinstance(features2, tuple):
                    features2 = features2[0]

                # Project to contrastive space
                z1 = self.projector(features1)
                z2 = self.projector(features2)

                # Compute contrastive loss
                loss = self.criterion(z1, z2)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            history["loss"].append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Contrastive Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        logger.info(f"Contrastive pretraining complete. Final loss: {history['loss'][-1]:.4f}")
        return history

    def save_projector(self, path):
        """Save projector weights for later use."""
        torch.save(self.projector.state_dict(), path)
        logger.info(f"Projector saved to {path}")

    def load_projector(self, path):
        """Load projector weights."""
        self.projector.load_state_dict(torch.load(path))
        logger.info(f"Projector loaded from {path}")


# ── Augmentation Utilities ────────────────────────────────────────────────────
def get_medical_augmentation(strength="weak"):
    """Get augmentation pipeline appropriate for medical images.

    Args:
        strength: 'weak' or 'strong'

    Returns:
        transforms.Compose augmentation pipeline
    """
    if strength == "weak":
        return MEDICAL_WEAK_AUGMENTATION
    else:
        return MEDICAL_STRONG_AUGMENTATION


def create_contrastive_dataloader(root_dir, batch_size=32, num_workers=4, image_size=224):
    """Create a DataLoader for contrastive pretraining.

    Args:
        root_dir: Path to dataset directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size

    Returns:
        DataLoader with ContrastiveRetinaDataset
    """
    dataset = ContrastiveRetinaDataset(root_dir, image_size=image_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
