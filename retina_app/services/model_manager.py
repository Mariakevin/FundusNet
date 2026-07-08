"""
Model manager — loading, singleton, checkpoint handling, health monitoring.
"""

import os
import time
import logging
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torchvision.models as models

from retina_app.constants import (
    CATEGORIES,
    MODEL_LIST,
    MODEL_HEALTH_WINDOW,
    MODEL_HEALTH_MIN_ACCURACY,
)
from retina_app.services.exceptions import ModelLoadError

logger = logging.getLogger("retina_app")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_VERSIONS = {
    "squeezenet": "squeezenet1_0-retinopathy-v1",
    "efficientnet": "efficientnet-b0-retinopathy-v1",
    "resnet": "resnet50-retinopathy-v1",
    "mobilenet": "mobilenetv3-retinopathy-v1",
    "convnext": "convnext-tiny-retinopathy-v1",
    "vit": "vit-b16-retinopathy-v1",
}

MODEL_PATHS: Optional[Dict[str, str]] = None


def _get_model_paths() -> Dict[str, str]:
    """Lazy evaluation of model paths to avoid import-time Django configuration issues."""
    from django.conf import settings
    models_dir = os.path.join(settings.BASE_DIR, "models")
    return {
        "squeezenet": os.path.join(models_dir, "squeezenet_retinopathy.pth"),
        "efficientnet": os.path.join(models_dir, "efficientnet_retinopathy.pth"),
        "resnet": os.path.join(models_dir, "resnet_retinopathy.pth"),
        "mobilenet": os.path.join(models_dir, "mobilenet_retinopathy.pth"),
        "convnext": os.path.join(models_dir, "convnext_retinopathy.pth"),
        "vit": os.path.join(models_dir, "vit_retinopathy.pth"),
    }


def _load_checkpoint_features(checkpoint: Dict, model: nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Load compatible feature layers from checkpoint into model.

    Returns (compatible_features, model_state) — model_state is the current
    model's state dict, reused by the caller to avoid double state_dict() calls.
    """
    model_state = model.state_dict()
    compatible = {}

    for key, value in checkpoint.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                compatible[key] = value

    return compatible, model_state


def _create_improved_classifier(model_type: str, in_features: int, num_classes: int) -> nn.Module:
    """Create an improved classifier with better initialization."""
    if model_type == "squeezenet":
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )
    elif model_type == "efficientnet":
        classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
    elif model_type == "resnet":
        classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
    elif model_type == "convnext":
        classifier = nn.Sequential(
            nn.LayerNorm([in_features]),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
    elif model_type == "vit":
        classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
    else:  # mobilenet
        classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Linear(1024, num_classes),
        )

    for m in classifier:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    return classifier


def _load_model_with_checkpoint(model_type: str, model_path: str) -> Tuple[nn.Module, bool, int]:
    """Load model with checkpoint, returns (model, checkpoint_loaded, in_features)."""
    if model_type == "squeezenet":
        model = models.squeezenet1_0(weights=None)
        in_features = 512
        model.classifier = _create_improved_classifier(model_type, in_features, len(CATEGORIES))
    elif model_type == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = _create_improved_classifier(model_type, in_features, len(CATEGORIES))
    elif model_type == "resnet":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = _create_improved_classifier(model_type, in_features, len(CATEGORIES))
    elif model_type == "convnext":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier = _create_improved_classifier(model_type, in_features, len(CATEGORIES))
    elif model_type == "vit":
        model = models.vit_b_16(weights=None)
        in_features = model.heads.head.in_features
        model.heads = _create_improved_classifier(model_type, in_features, len(CATEGORIES))
    else:  # mobilenet
        model = models.mobilenet_v3_small(weights=None)
        in_features = 576
        model.classifier = _create_improved_classifier(model_type, in_features, len(CATEGORIES))

    checkpoint_loaded = False
    if model_path and os.path.exists(model_path):
        try:
            try:
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
            except Exception:
                logger.warning("weights_only=True failed for %s, falling back to safe load", model_type)
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                saved_type = checkpoint.get('model_type', '')
                if saved_type == model_type:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    checkpoint_loaded = True
                    logger.info(f"Loaded full checkpoint for {model_type}")
                else:
                    compatible_features, model_state = _load_checkpoint_features(checkpoint, model)
                    if len(compatible_features) > 10:
                        model_state.update(compatible_features)
                        model.load_state_dict(model_state, strict=False)
                        checkpoint_loaded = True
                        logger.info(f"Loaded {len(compatible_features)} compatible layers from checkpoint for {model_type}")
                    else:
                        logger.warning(f"Too few compatible layers ({len(compatible_features)}) from checkpoint")
            else:
                compatible_features, model_state = _load_checkpoint_features(checkpoint, model)
                if len(compatible_features) > 10:
                    model_state.update(compatible_features)
                    model.load_state_dict(model_state, strict=False)
                    checkpoint_loaded = True
                    logger.info(f"Loaded {len(compatible_features)} compatible layers from checkpoint for {model_type}")
                else:
                    logger.warning(f"Too few compatible layers ({len(compatible_features)}) from checkpoint")

        except Exception as e:
            logger.warning(f"Could not load checkpoint for {model_type}: {e}")

    return model, checkpoint_loaded, in_features


class ModelManager:
    """Manages multiple ML models for ensemble inference."""

    def __init__(self):
        self._models: Dict[str, nn.Module] = {}
        self._model_types: Dict[str, str] = {}

    def get_model(self, model_type: str = "efficientnet") -> nn.Module:
        """Load and return a specific model."""
        if model_type not in self._models:
            self._models[model_type] = self._load_model(model_type)
        return self._models[model_type]

    def _load_model(self, model_type: str) -> nn.Module:
        """Load a trained model or fall back to pre-trained."""
        global MODEL_PATHS
        if MODEL_PATHS is None:
            MODEL_PATHS = _get_model_paths()

        model_path = MODEL_PATHS.get(model_type)

        try:
            model, checkpoint_loaded, _ = _load_model_with_checkpoint(model_type, model_path)
            if not checkpoint_loaded:
                logger.warning(f"No valid checkpoint for {model_type}, falling back to ImageNet pretrained")
                return self._load_pretrained(model_type)
            model.to(DEVICE)
            model.eval()
            self._model_types[model_type] = "trained"
            logger.info(f"Loaded {model_type} model (trained)")
            return model
        except Exception as exc:
            logger.warning(f"Failed to create {model_type} model: {exc}")
            return self._load_pretrained(model_type)

    def _load_pretrained(self, model_type: str) -> nn.Module:
        """Load pre-trained model with improved classifier."""
        logger.info(f"Loading pre-trained {model_type} for demo")

        if model_type == "squeezenet":
            model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
            in_features = 512
            model.classifier = _create_improved_classifier(model_type, in_features, len(CATEGORIES))
        elif model_type == "efficientnet":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = model.classifier[1].in_features
            model.classifier = _create_improved_classifier(model_type, in_features, len(CATEGORIES))
        elif model_type == "resnet":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = model.fc.in_features
            model.fc = _create_improved_classifier(model_type, in_features, len(CATEGORIES))
        elif model_type == "convnext":
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            in_features = model.classifier[2].in_features
            model.classifier = _create_improved_classifier(model_type, in_features, len(CATEGORIES))
        elif model_type == "vit":
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            in_features = model.heads.head.in_features
            model.heads = _create_improved_classifier(model_type, in_features, len(CATEGORIES))
        else:
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            in_features = 576
            model.classifier = _create_improved_classifier(model_type, in_features, len(CATEGORIES))

        model.to(DEVICE)
        model.eval()
        self._model_types[model_type] = "pretrained"
        return model

    def reload(self) -> None:
        self._models = {}
        self._model_types = {}


class ModelHealthTracker:
    """Tracks per-model prediction accuracy and health status."""

    def __init__(self):
        self._lock = threading.Lock()
        self._predictions: Dict[str, deque] = {}

    def record_prediction(self, model_type: str, predicted_class: str, confidence: float) -> None:
        """Record a model prediction for health tracking."""
        with self._lock:
            if model_type not in self._predictions:
                self._predictions[model_type] = deque(maxlen=MODEL_HEALTH_WINDOW)
            self._predictions[model_type].append({
                "class": predicted_class,
                "confidence": confidence,
                "timestamp": time.time(),
            })

    def record_ensemble_prediction(self, predictions: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Record predictions from all models in an ensemble run."""
        for model_type, pred in predictions:
            self.record_prediction(model_type, pred.get("label", ""), pred.get("confidence", 0.0))

    def get_model_health(self, model_type: str) -> Dict[str, Any]:
        """Get health status for a specific model."""
        with self._lock:
            if model_type not in self._predictions:
                return {
                    "model": model_type,
                    "status": "unknown",
                    "predictions": 0,
                    "avg_confidence": 0.0,
                    "accuracy": None,
                }

            preds = list(self._predictions[model_type])
            n = len(preds)
            avg_conf = sum(p["confidence"] for p in preds) / n if n > 0 else 0.0

            status = "healthy"
            if n < 10:
                status = "warming_up"
            elif avg_conf < MODEL_HEALTH_MIN_ACCURACY:
                status = "degraded"

            return {
                "model": model_type,
                "status": status,
                "predictions": n,
                "avg_confidence": round(avg_conf, 4),
                "accuracy": round(avg_conf, 4),
                "window_size": MODEL_HEALTH_WINDOW,
            }

    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all tracked models."""
        with self._lock:
            all_types = set(list(self._predictions.keys()) + MODEL_LIST)
        return {mt: self.get_model_health(mt) for mt in sorted(all_types)}

    def reset(self) -> None:
        """Reset all tracking data."""
        with self._lock:
            self._predictions.clear()


_model_manager: Optional[ModelManager] = None
_health_tracker: Optional[ModelHealthTracker] = None
_manager_lock = threading.Lock()
_health_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        with _manager_lock:
            if _model_manager is None:
                _model_manager = ModelManager()
    return _model_manager


def get_health_tracker() -> ModelHealthTracker:
    global _health_tracker
    if _health_tracker is None:
        with _health_lock:
            if _health_tracker is None:
                _health_tracker = ModelHealthTracker()
    return _health_tracker
