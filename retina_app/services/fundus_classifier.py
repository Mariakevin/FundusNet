"""Learned fundus image classifier.

Binary classifier that distinguishes fundus images from non-fundus images.
Replaces hand-tuned heuristic thresholds with a trained model.
"""

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
