"""Knowledge Distillation for Lightweight Student Models.

Implements knowledge distillation from large teacher models (DINOv2, ensemble)
to lightweight student models for efficient edge deployment.

Based on research findings:
- DINOv2 (ViT-L) as frozen teacher provides excellent feature representations
- Knowledge distillation to lightweight students is very effective
- Can reduce model size 10-50x while maintaining 90%+ accuracy

Reference: Distilling Vision-Language Models for Fundus Disease Classification (2025)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("retina_app")


class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation.

    Combines:
    1. Task loss (cross-entropy with ground truth)
    2. Distillation loss (KL divergence between teacher and student soft targets)
    3. Feature loss (MSE between intermediate features, optional)
    """

    def __init__(self, temperature=4.0, alpha=0.7, beta=0.3, feature_loss_weight=0.0):
        """
        Args:
            temperature: Softmax temperature for soft targets
            alpha: Weight for distillation loss
            beta: Weight for task loss
            feature_loss_weight: Weight for feature matching loss (0 to disable)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.feature_loss_weight = feature_loss_weight

    def forward(self, student_logits, teacher_logits, targets, student_features=None, teacher_features=None):
        """
        Args:
            student_logits: [batch_size, num_classes] from student
            teacher_logits: [batch_size, num_classes] from teacher
            targets: [batch_size] ground truth labels
            student_features: [batch_size, feature_dim] intermediate features (optional)
            teacher_features: [batch_size, feature_dim] intermediate features (optional)

        Returns:
            Combined loss scalar
        """
        # Task loss (cross-entropy with hard labels)
        task_loss = F.cross_entropy(student_logits, targets)

        # Distillation loss (KL divergence with soft targets)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distill_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        distill_loss = distill_loss * (self.temperature**2)

        # Feature matching loss (optional)
        feature_loss = torch.tensor(0.0, device=student_logits.device)
        if self.feature_loss_weight > 0 and student_features is not None and teacher_features is not None:
            feature_loss = F.mse_loss(student_features, teacher_features)

        # Combined loss
        total_loss = self.alpha * distill_loss + self.beta * task_loss + self.feature_loss_weight * feature_loss

        return total_loss


class DINOv2Teacher:
    """DINOv2 teacher model wrapper.

    Uses DINOv2 ViT-L as a frozen feature extractor for distillation.
    DINOv2 provides excellent general-purpose visual features.
    """

    def __init__(self, model_name="dinov2_vitl14", device="cpu"):
        """
        Args:
            model_name: DINOv2 model variant
            device: Device to run on
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load pretrained DINOv2 model."""
        try:
            import timm

            self.model = timm.create_model(
                self.model_name,
                pretrained=True,
                num_classes=0,  # Remove classification head
            )
            self.model.eval()
            self.model.to(self.device)
            logger.info(f"Loaded DINOv2 teacher: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load DINOv2: {e}")
            self.model = None

    def extract_features(self, images):
        """Extract features from DINOv2 teacher.

        Args:
            images: [batch_size, 3, H, W] input images

        Returns:
            [batch_size, feature_dim] teacher features
        """
        if self.model is None:
            raise RuntimeError("DINOv2 teacher not loaded")

        with torch.no_grad():
            features = self.model(images)
            if isinstance(features, tuple):
                features = features[0]

        return features.to(self.device)

    def extract_logits(self, images, num_classes=4):
        """Extract logits from DINOv2 with a classification head.

        Args:
            images: [batch_size, 3, H, W] input images
            num_classes: Number of output classes

        Returns:
            [batch_size, num_classes] logits
        """
        features = self.extract_features(images)

        # Add a simple linear head if not present
        if not hasattr(self, "_classifier"):
            in_features = features.shape[1]
            self._classifier = nn.Linear(in_features, num_classes).to(self.device)

        logits = self._classifier(features)
        return logits


class EnsembleTeacher:
    """Ensemble teacher that averages predictions from multiple trained models.

    Uses the existing ensemble models as teachers for distillation.
    """

    def __init__(self, models_dict, weights=None, device="cpu"):
        """
        Args:
            models_dict: {model_type: model} dictionary of trained models
            weights: Optional weight dictionary for weighted averaging
            device: Device to run on
        """
        self.device = device
        self.models = models_dict
        self.weights = weights or {k: 1.0 / len(models_dict) for k in models_dict}

    def extract_logits(self, images):
        """Extract ensemble logits from all teacher models.

        Args:
            images: [batch_size, 3, H, W] input images

        Returns:
            [batch_size, num_classes] ensemble logits
        """
        all_logits = []

        for model_type, model in self.models.items():
            model.eval()
            with torch.no_grad():
                logits = model(images)
                if isinstance(logits, tuple):
                    logits = logits[0]
                all_logits.append(logits * self.weights[model_type])

        # Average logits (weighted)
        ensemble_logits = torch.stack(all_logits).sum(dim=0)

        return ensemble_logits.to(self.device)


class StudentModel(nn.Module):
    """Lightweight student model for edge deployment.

    Can be any small architecture (MobileNet, EfficientNet-B0, etc.)
    """

    def __init__(self, model_type="efficientnet_b0", num_classes=4):
        """
        Args:
            model_type: Student model architecture
            num_classes: Number of output classes
        """
        super().__init__()

        try:
            import timm

            model_map = {
                "efficientnet_b0": "efficientnet_b0.ra_in1k",
                "mobilenet_v3": "mobilenetv3_small_075.lamb_in1k",
                "mobileone": "mobileone_s0.lamb_in1k",
            }

            timm_name = model_map.get(model_type, model_type)
            self.model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
            self.model_type = model_type

        except ImportError:
            raise ImportError("timm is required for student models")

    def forward(self, x):
        return self.model(x)


def distill_knowledge(
    teacher,
    student,
    train_loader,
    num_epochs=50,
    lr=1e-4,
    temperature=4.0,
    alpha=0.7,
    device="cpu",
    save_path=None,
):
    """Perform knowledge distillation from teacher to student.

    Args:
        teacher: Teacher model (DINOv2Teacher or EnsembleTeacher)
        student: StudentModel to train
        train_loader: DataLoader with labeled training data
        num_epochs: Number of training epochs
        lr: Learning rate
        temperature: Softmax temperature for soft targets
        alpha: Weight for distillation vs task loss
        device: Device to train on
        save_path: Path to save best student model

    Returns:
        Dictionary with training history
    """
    student.to(device)
    criterion = DistillationLoss(temperature=temperature, alpha=alpha, beta=1 - alpha)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": []}

    for epoch in range(num_epochs):
        student.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            # Student forward pass
            student_logits = student(images)

            # Teacher forward pass
            with torch.no_grad():
                if isinstance(teacher, DINOv2Teacher):
                    teacher_logits = teacher.extract_logits(images)
                elif isinstance(teacher, EnsembleTeacher):
                    teacher_logits = teacher.extract_logits(images)
                else:
                    teacher_logits = teacher(images)

            # Compute distillation loss
            loss = criterion(student_logits, teacher_logits, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            if save_path:
                torch.save(student.state_dict(), save_path)
                logger.info(f"Best student model saved: {accuracy:.2f}%")

        if (epoch + 1) % 10 == 0:
            logger.info(f"Distill Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

    logger.info(f"Distillation complete. Best accuracy: {best_acc:.2f}%")

    return history


def create_distillation_pipeline(
    teacher_type="ensemble",
    student_type="efficientnet_b0",
    num_classes=4,
    device="cpu",
):
    """Create a complete distillation pipeline.

    Args:
        teacher_type: 'dinov2' or 'ensemble'
        student_type: Student model architecture
        num_classes: Number of output classes
        device: Device to use

    Returns:
        Tuple of (teacher, student, distill_fn)
    """
    if teacher_type == "dinov2":
        teacher = DINOv2Teacher(device=device)
    else:
        teacher = EnsembleTeacher({}, device=device)  # Empty until models loaded

    student = StudentModel(model_type=student_type, num_classes=num_classes)

    return teacher, student
