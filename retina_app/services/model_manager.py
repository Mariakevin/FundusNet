"""Model manager — loading, singleton, checkpoint handling, health monitoring.
Supports PyTorch and ONNX runtime for fast inference.
"""

import logging
import os
import threading
import time
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from retina_app.constants import (
    CATEGORIES,
    MODEL_HEALTH_MIN_ACCURACY,
    MODEL_HEALTH_WINDOW,
    MODEL_LIST,
)

logger = logging.getLogger("retina_app")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_VERSIONS = {
    "swin": "swin-tiny-retinopathy-v1",
    "maxvit": "maxvit-base-retinopathy-v1",
    "convnext_v2": "convnextv2-base-retinopathy-v1",
    "efficientnet_v2": "efficientnetv2-m-retinopathy-v1",
    "deit": "deit3-base-retinopathy-v1",
}

MODEL_PATHS: dict[str, str] | None = None

# Optional ONNX runtime
_ort_session = None
_ort_available = False
try:
    import onnxruntime as ort

    _ort_available = True
except ImportError:
    ort = None  # type: ignore[assignment]


def _get_model_paths() -> dict[str, str]:
    """Lazy evaluation of model paths to avoid import-time Django configuration issues."""
    from django.conf import settings

    models_dir = os.path.join(settings.BASE_DIR, "models")
    return {
        "swin": os.path.join(models_dir, "swin_retinopathy.onnx"),
        "maxvit": os.path.join(models_dir, "maxvit_retinopathy.onnx"),
        "convnext_v2": os.path.join(models_dir, "convnext_v2_retinopathy.onnx"),
        "efficientnet_v2": os.path.join(models_dir, "efficientnet_v2_retinopathy.onnx"),
        "deit": os.path.join(models_dir, "deit_retinopathy.onnx"),
    }


def _get_pytorch_model_paths() -> dict[str, str]:
    """Lazy evaluation of PyTorch model paths for fallback."""
    from django.conf import settings

    models_dir = os.path.join(settings.BASE_DIR, "models")
    return {
        "swin": os.path.join(models_dir, "swin_retinopathy.pth"),
        "maxvit": os.path.join(models_dir, "maxvit_retinopathy.pth"),
        "convnext_v2": os.path.join(models_dir, "convnext_v2_retinopathy.pth"),
        "efficientnet_v2": os.path.join(models_dir, "efficientnet_v2_retinopathy.pth"),
        "deit": os.path.join(models_dir, "deit_retinopathy.pth"),
    }


def _create_timm_model(model_type, num_classes=4, pretrained=False):
    """Create model using timm library for modern architectures."""
    try:
        import timm

        model_map = {
            "swin": "swin_tiny_patch4_window7_224.ms_in22k",
            "maxvit": "maxvit_base_224.sw_in1k",
            "convnext_v2": "convnextv2_base.fcmae_ft_in1k",
            "efficientnet_v2": "efficientnet_v2_m.orig_in21k_ft_in1k",
            "deit": "deit3_base_patch16_224.fb_in22k_ft_in1k",
        }
        if model_type in model_map:
            model = timm.create_model(model_map[model_type], pretrained=pretrained, num_classes=num_classes)
            return model
    except ImportError:
        pass
    return None


def _load_onnx_session(model_type: str):
    """Load ONNX runtime session for a model."""
    global _ort_session
    if not _ort_available:
        return None

    global MODEL_PATHS
    if MODEL_PATHS is None:
        MODEL_PATHS = _get_model_paths()

    onnx_path = MODEL_PATHS.get(model_type)
    if onnx_path and os.path.exists(onnx_path):
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            _ort_session = ort.InferenceSession(onnx_path, providers=providers)
            logger.info(f"Loaded ONNX session for {model_type} from {onnx_path}")
            return _ort_session
        except Exception as e:
            logger.warning(f"Failed to load ONNX for {model_type}: {e}")
            _ort_session = None
    return _ort_session


def _run_onnx_inference(model_type: str, tensor: torch.Tensor) -> dict[str, Any] | None:
    """Run inference using ONNX runtime."""
    session = _load_onnx_session(model_type)
    if session is None:
        return None

    try:
        # Convert tensor to numpy for ONNX
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        np_input = tensor.detach().cpu().numpy()

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: np_input})
        logits = outputs[0][0]

        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        probs = probs.tolist()

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        label = CATEGORIES[predicted_idx]

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {CATEGORIES[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "model": model_type,
        }
    except Exception as e:
        logger.warning(f"ONNX inference failed for {model_type}: {e}")
        return None


def _load_checkpoint_features(
    checkpoint: dict, model: nn.Module
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load compatible feature layers from checkpoint into model."""
    model_state = model.state_dict()
    compatible = {}
    for key, value in checkpoint.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
    return compatible, model_state


def _load_model_with_checkpoint(model_type: str, model_path: str) -> tuple[nn.Module, bool, int]:
    """Load model with checkpoint, returns (model, checkpoint_loaded, in_features)."""
    model = _create_timm_model(model_type, num_classes=len(CATEGORIES), pretrained=False)
    if model is None:
        raise ValueError(f"Unknown model type: {model_type}")

    # Get in_features from model head
    in_features = len(CATEGORIES)
    if hasattr(model, "classifier"):
        if hasattr(model.classifier, "in_features"):
            in_features = model.classifier.in_features
    elif hasattr(model, "head") and hasattr(model.head, "in_features"):
        in_features = model.head.in_features

    checkpoint_loaded = False
    if model_path and os.path.exists(model_path):
        try:
            try:
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
            except Exception:
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                saved_type = checkpoint.get("model_type", "")
                if saved_type == model_type:
                    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    checkpoint_loaded = True
                    logger.info(f"Loaded full checkpoint for {model_type}")
                else:
                    compatible_features, model_state = _load_checkpoint_features(checkpoint, model)
                    if len(compatible_features) > 10:
                        model_state.update(compatible_features)
                        model.load_state_dict(model_state, strict=False)
                        checkpoint_loaded = True
                        logger.info(
                            f"Loaded {len(compatible_features)} compatible layers from checkpoint for {model_type}"
                        )
            else:
                compatible_features, model_state = _load_checkpoint_features(checkpoint, model)
                if len(compatible_features) > 10:
                    model_state.update(compatible_features)
                    model.load_state_dict(model_state, strict=False)
                    checkpoint_loaded = True
                    logger.info(f"Loaded {len(compatible_features)} compatible layers from checkpoint for {model_type}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint for {model_type}: {e}")

    return model, checkpoint_loaded, in_features


class ModelManager:
    """Manages multiple ML models for ensemble inference with ONNX support."""

    def __init__(self):
        self._models: dict[str, nn.Module] = {}
        self._model_types: dict[str, str] = {}

    def get_model(self, model_type: str = "convnext_v2") -> nn.Module:
        """Load and return a specific model."""
        if model_type not in self._models:
            self._models[model_type] = self._load_model(model_type)
        return self._models[model_type]

    def _load_model(self, model_type: str) -> nn.Module:
        """Load a trained model or fall back to pre-trained."""
        global MODEL_PATHS
        if MODEL_PATHS is None:
            MODEL_PATHS = _get_model_paths()

        # Try ONNX first
        if _ort_available and _load_onnx_session(model_type):
            logger.info(f"Using ONNX runtime for {model_type}")
            self._model_types[model_type] = "onnx"
            return nn.Module()  # Placeholder - ONNX runs separately

        # Try PyTorch checkpoint
        try:
            pytorch_paths = _get_pytorch_model_paths()
            model_path = pytorch_paths.get(model_type, "")
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
        """Load pre-trained model with timm."""
        logger.info(f"Loading pre-trained {model_type} for demo")
        model = _create_timm_model(model_type, num_classes=len(CATEGORIES), pretrained=True)
        if model is None:
            raise ValueError(f"Unknown model type: {model_type}")
        model.to(DEVICE)
        model.eval()
        self._model_types[model_type] = "pretrained"
        return model

    def run_inference(self, model_type: str, tensor: torch.Tensor) -> dict[str, Any] | None:
        """Run inference on a model, preferring ONNX if available."""
        # Try ONNX first
        if model_type in self._model_types and self._model_types[model_type] == "onnx":
            result = _run_onnx_inference(model_type, tensor)
            if result:
                return result
            logger.warning(f"ONNX inference failed for {model_type}, falling back to PyTorch")

        # Fallback to PyTorch
        model = self.get_model(model_type)
        if not isinstance(model, nn.Module) or not hasattr(model, "forward"):
            return None

        try:
            with torch.no_grad():
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.to(DEVICE)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy().tolist()

                predicted_idx = int(np.argmax(probs))
                confidence = float(probs[predicted_idx])
                label = CATEGORIES[predicted_idx]

                return {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "probabilities": {CATEGORIES[i]: round(float(p), 4) for i, p in enumerate(probs)},
                    "model": model_type,
                }
        except Exception as e:
            logger.warning(f"PyTorch inference failed for {model_type}: {e}")
            return None

    def reload(self) -> None:
        self._models = {}
        self._model_types = {}


class ModelHealthTracker:
    """Tracks per-model prediction accuracy and health status."""

    def __init__(self):
        self._lock = threading.Lock()
        self._predictions: dict[str, deque] = {}

    def record_prediction(self, model_type: str, predicted_class: str, confidence: float) -> None:
        """Record a model prediction for health tracking."""
        with self._lock:
            if model_type not in self._predictions:
                self._predictions[model_type] = deque(maxlen=MODEL_HEALTH_WINDOW)
            self._predictions[model_type].append(
                {
                    "class": predicted_class,
                    "confidence": confidence,
                    "timestamp": time.time(),
                }
            )

    def record_ensemble_prediction(self, predictions: list[tuple[str, dict[str, Any]]]) -> None:
        """Record predictions from all models in an ensemble run."""
        for model_type, pred in predictions:
            self.record_prediction(model_type, pred.get("label", ""), pred.get("confidence", 0.0))

    def get_model_health(self, model_type: str) -> dict[str, Any]:
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

    def get_all_health(self) -> dict[str, dict[str, Any]]:
        """Get health status for all tracked models."""
        with self._lock:
            all_types = set(list(self._predictions.keys()) + MODEL_LIST)
        return {mt: self.get_model_health(mt) for mt in sorted(all_types)}

    def reset(self) -> None:
        """Reset all tracking data."""
        with self._lock:
            self._predictions.clear()


_model_manager: ModelManager | None = None
_health_tracker: ModelHealthTracker | None = None
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
