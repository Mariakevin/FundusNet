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
    MODEL_NAME_MAP,
)

logger = logging.getLogger("retina_app")

# --- Model Loading State ---
_models_loading = False
_models_loaded = False
_models_load_error = None
_models_lock = threading.Lock()

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
_ort_sessions: dict[str, Any] = {}
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

        if model_type in MODEL_NAME_MAP:
            model = timm.create_model(MODEL_NAME_MAP[model_type], pretrained=pretrained, num_classes=num_classes)
            return model
    except ImportError:
        pass
    return None


def _load_onnx_session(model_type: str):
    """Load ONNX runtime session for a model. Uses per-model cache to avoid overwrites."""
    if not _ort_available:
        return None

    if model_type in _ort_sessions:
        return _ort_sessions[model_type]

    global MODEL_PATHS
    if MODEL_PATHS is None:
        MODEL_PATHS = _get_model_paths()

    onnx_path = MODEL_PATHS.get(model_type)
    if onnx_path and os.path.exists(onnx_path):
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            session = ort.InferenceSession(onnx_path, providers=providers)
            _ort_sessions[model_type] = session
            logger.info("Loaded ONNX session for %s from %s", model_type, onnx_path)
            return session
        except Exception as e:
            logger.warning("Failed to load ONNX for %s: %s", model_type, e)
    return None


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
        logger.warning("ONNX inference failed for %s: %s", model_type, e)
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
                    logger.info("Loaded full checkpoint for %s", model_type)
                else:
                    compatible_features, model_state = _load_checkpoint_features(checkpoint, model)
                    if len(compatible_features) > 10:
                        model_state.update(compatible_features)
                        model.load_state_dict(model_state, strict=False)
                        checkpoint_loaded = True
                        logger.info(
                            "Loaded %d compatible layers from checkpoint for %s", len(compatible_features), model_type
                        )
            else:
                compatible_features, model_state = _load_checkpoint_features(checkpoint, model)
                if len(compatible_features) > 10:
                    model_state.update(compatible_features)
                    model.load_state_dict(model_state, strict=False)
                    checkpoint_loaded = True
                    logger.info("Loaded %d compatible layers from checkpoint for %s", len(compatible_features), model_type)
        except Exception as e:
            logger.warning("Could not load checkpoint for %s: %s", model_type, e)

    return model, checkpoint_loaded, in_features


class _OnnxModelWrapper(nn.Module):
    """Wraps an ONNX session so ensemble.py can call model(tensor) transparently."""

    def __init__(self, model_type: str, session):
        super().__init__()
        self._model_type = model_type
        self._session = session
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        np_input = x.detach().cpu().numpy()
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: np_input})
        return torch.tensor(outputs[0], dtype=torch.float32)


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
        if _ort_available:
            session = _load_onnx_session(model_type)
            if session is not None:
                logger.info("Using ONNX runtime for %s", model_type)
                self._model_types[model_type] = "onnx"
                return _OnnxModelWrapper(model_type, session)

        # Try PyTorch checkpoint
        try:
            pytorch_paths = _get_pytorch_model_paths()
            model_path = pytorch_paths.get(model_type, "")
            model, checkpoint_loaded, _ = _load_model_with_checkpoint(model_type, model_path)
            if not checkpoint_loaded:
                logger.warning(
                    "No valid checkpoint for %s — attempting pretrained fallback (PRETRAINED_FALLBACK_ENABLED=%s)",
                    model_type,
                    os.environ.get("FUNDUSNET_PRETRAINED", "check constants.py"),
                )
                return self._load_pretrained(model_type)
            model.to(DEVICE)
            model.eval()
            self._model_types[model_type] = "trained"
            logger.info("Loaded %s model (trained)", model_type)
            return model
        except Exception as exc:
            logger.warning("Failed to create %s model: %s", model_type, exc)
            return self._load_pretrained(model_type)

    def _load_pretrained(self, model_type: str) -> nn.Module:
        """Load pre-trained model with timm."""
        from retina_app.constants import PRETRAINED_FALLBACK_ENABLED

        if not PRETRAINED_FALLBACK_ENABLED:
            logger.warning(
                "PRETRAINED_FALLBACK_ENABLED=False — loading %s with random weights (no download)",
                model_type,
            )
            model = _create_timm_model(model_type, num_classes=len(CATEGORIES), pretrained=False)
            if model is None:
                raise ValueError(f"Unknown model type: {model_type}")
            model.to(DEVICE)
            model.eval()
            self._model_types[model_type] = "random"
            return model

        logger.info("Loading pre-trained %s for demo", model_type)
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
            logger.warning("ONNX inference failed for %s, falling back to PyTorch", model_type)

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
            logger.warning("PyTorch inference failed for %s: %s", model_type, e)
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


def get_model_loading_status() -> dict:
    """Return current model loading status for the UI."""
    with _models_lock:
        return {
            "loading": _models_loading,
            "loaded": _models_loaded,
            "error": _models_load_error,
            "n_models": len(_model_manager._models) if _model_manager else 0,
        }


def preload_models_background() -> None:
    """Preload all models in background threads at server startup."""
    global _models_loading, _models_loaded, _models_load_error

    with _models_lock:
        if _models_loading or _models_loaded:
            return
        _models_loading = True
        _models_load_error = None

    def _do_preload():
        global _models_loading, _models_loaded, _models_load_error
        try:
            manager = get_model_manager()
            loaded = 0
            for model_type in MODEL_LIST:
                try:
                    manager.get_model(model_type)
                    loaded += 1
                    logger.info("Preloaded model %s (%d/%d)", model_type, loaded, len(MODEL_LIST))
                except Exception as exc:
                    logger.warning("Failed to preload %s: %s", model_type, exc)

            with _models_lock:
                _models_loaded = True
                _models_loading = False
                if loaded == 0:
                    _models_load_error = "No models could be loaded"
            logger.info("Model preloading complete: %d/%d loaded", loaded, len(MODEL_LIST))
        except Exception as exc:
            with _models_lock:
                _models_loading = False
                _models_load_error = str(exc)
            logger.error("Model preloading failed: %s", exc)

    thread = threading.Thread(target=_do_preload, daemon=True, name="model-preloader")
    thread.start()
    logger.info("Started background model preloader thread")
