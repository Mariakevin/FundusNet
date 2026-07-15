"""Inference orchestrator — thin facade over preprocessing, model_manager, ensemble, cache.

All public API remains importable from this module for backward compatibility.
"""

import atexit
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
import torch
from django.conf import settings as django_settings
from PIL import Image

from retina_app.constants import (
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_LOW,
    ENABLE_MC_DROPOUT,
    ENSEMBLE_MIN_MODELS,
    FUNDUS_VALIDATION_ENABLED,
    GRADCAM_MODEL,
    MAX_WORKERS,
    MODEL_LIST,
    MODEL_WEIGHTS,
)
from retina_app.services.ensemble import (
    _predict_single_model,
    detect_model_disagreement,
    predict_models_parallel,
    predict_with_uncertainty_ensemble,
    selective_ensemble,
)
from retina_app.services.exceptions import (
    ImageCorruptError,
    ImageValidationError,
    InferenceError,
    NotAFundusImageError,
)
from retina_app.services.fundus_validator import validate_fundus_image
from retina_app.services.gradcam import generate_gradcam_for_image
from retina_app.services.image_cache import (
    _get_image_hash,
    get_cache_entry,
    set_cache_entry,
)
from retina_app.services.model_manager import (
    MODEL_VERSIONS,
    get_health_tracker,
    get_model_manager,
)
from retina_app.services.preprocessing import (
    apply_clahe,
    check_image_quality,
    preprocess_fundus,
    validate_image_file,
)
from retina_app.services.refusal import check_all_refusals

logger = logging.getLogger("retina_app")

_executor = None
_executor_lock = threading.Lock()


def get_executor():
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
                atexit.register(_executor.shutdown, wait=False)
    return _executor


def _load_and_validate_image(image_path: str) -> Image.Image:
    """Load image once and validate. Returns loaded PIL Image."""
    try:
        with Image.open(image_path) as img:
            img.verify()
            img.seek(0)
            pil_image = Image.open(image_path)
            pil_image.load()
    except Exception as exc:
        logger.warning("Corrupted image file %s: %s", image_path, exc)
        raise ImageCorruptError(f"Corrupted or unreadable image: {exc}") from exc

    try:
        validate_image_file(image_path, pil_image=pil_image)
    except ImageValidationError as exc:
        logger.warning("Image validation failed for %s: %s", image_path, exc)
        raise InferenceError(f"Invalid image: {exc}") from exc

    if FUNDUS_VALIDATION_ENABLED:
        fundus_check = validate_fundus_image(image_path, pil_image=pil_image)
        if not fundus_check["is_fundus"]:
            logger.warning(
                "Non-fundus image rejected: score=%.3f, signals=%s, path=%s",
                fundus_check["confidence"],
                fundus_check["signals"],
                image_path,
            )
            raise NotAFundusImageError(fundus_check["message"])

    try:
        quality_check = check_image_quality(image_path, quality_threshold=0.25, pil_image=pil_image)
        if not quality_check["passed"]:
            logger.warning("Image quality check failed: %s", quality_check["quality_level"])
    except Exception as exc:
        logger.warning("Quality check failed, continuing: %s", exc)

    return pil_image


def _apply_preprocessing(image_path: str, use_clahe: bool) -> tuple[str, str]:
    """Apply CLAHE preprocessing if requested. Returns (target_path, preprocessed_path)."""
    if not use_clahe:
        return image_path, None

    try:
        preprocessed = preprocess_fundus(image_path, enhance=False, detect_roi=False)
        preprocessed = apply_clahe(preprocessed)
        ext = os.path.splitext(image_path)[1]
        img_hash = _get_image_hash(image_path)
        tmp_dir = tempfile.gettempdir()
        preprocessed_path = os.path.join(tmp_dir, f"fundus_{img_hash}_clahe{ext}")
        cv2.imwrite(preprocessed_path, cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR))
        logger.info("Applied preprocessing pipeline to %s", image_path)
        return preprocessed_path, preprocessed_path
    except Exception as exc:
        logger.warning("Preprocessing failed, using original: %s", exc)
        return image_path, None


def _run_ensemble(
    model_manager,
    target_path: str,
    use_tta: bool,
) -> tuple[dict, list, str, str]:
    """Run ensemble inference. Returns (final_result, predictions, model_ver, model_source)."""
    logger.info(
        "Running ensemble with %d models: %s",
        len(model_manager._models),
        list(model_manager._models.keys()),
    )

    predictions = predict_models_parallel(model_manager._models, target_path, use_tta, get_executor())

    health_tracker = get_health_tracker()
    health_tracker.record_ensemble_prediction(predictions)

    disagreement = detect_model_disagreement(predictions)
    if disagreement["disagreement"]:
        logger.info(
            "Model disagreement detected: %s, agreement=%.2f",
            disagreement["class_votes"],
            disagreement["agreement_level"],
        )

    final_result = selective_ensemble(predictions)
    n_models = final_result.get("n_models", len(predictions))
    model_ver = f"ensemble-v{n_models}-models-tta" if use_tta else f"ensemble-v{n_models}-models"

    model_types = [model_manager._model_types.get(mt, "unknown") for mt, _ in predictions]
    has_trained = "trained" in model_types
    model_source = "trained" if has_trained else "pretrained"

    return final_result, predictions, model_ver, model_source


def _run_single_model(
    model_manager,
    target_path: str,
    use_tta: bool,
) -> tuple[dict, list, str, str]:
    """Run single model inference. Returns (final_result, predictions, model_ver, model_source)."""
    model = model_manager.get_model(MODEL_LIST[0])
    final_result = _predict_single_model(model, target_path, use_tta=use_tta)
    model_ver = MODEL_VERSIONS.get(MODEL_LIST[0], "demo")
    model_source = model_manager._model_types.get(MODEL_LIST[0], "pretrained")
    return final_result, [], model_ver, model_source


def _run_uncertainty(
    model_manager,
    target_path: str,
) -> dict | None:
    """Run MC Dropout uncertainty quantification."""
    if not ENABLE_MC_DROPOUT:
        return None
    try:
        data = predict_with_uncertainty_ensemble(
            model_manager._models,
            target_path,
            model_weights=MODEL_WEIGHTS,
        )
        logger.info(
            "MC Dropout uncertainty: entropy=%.4f, uncertain=%s",
            data["entropy"],
            data["is_uncertain"],
        )
        return data
    except Exception as exc:
        logger.warning("MC Dropout uncertainty failed: %s", exc)
        return None


def _build_confidence_warning(confidence: float) -> tuple[str | None, str | None]:
    """Determine confidence warning level and message."""
    if confidence < CONFIDENCE_THRESHOLD_LOW:
        logger.warning("Low confidence prediction: %.2f < %.2f", confidence, CONFIDENCE_THRESHOLD_LOW)
        return "low", "Low confidence. Result may be unreliable. Consider retaking the image with better lighting."
    elif confidence < CONFIDENCE_THRESHOLD_HIGH:
        return "medium", "Medium confidence. Result is reasonably reliable."
    return None, None


def predict_image(
    image_path: str,
    use_ensemble: bool = True,
    use_tta: bool = False,
    use_clahe: bool = False,
    use_uncertainty: bool = False,
    use_gradcam: bool = True,
) -> dict[str, Any]:
    """Run inference with optional ensemble of multiple models and test-time augmentation.

    Args:
        image_path: Path to the retinal image
        use_ensemble: Use multi-model ensemble
        use_tta: Use test-time augmentation
        use_clahe: Apply CLAHE preprocessing
        use_uncertainty: Run MC Dropout uncertainty quantification
        use_gradcam: Generate Grad-CAM explainability heatmap
    """
    start_time = time.time()

    # 1. Load and validate
    _load_and_validate_image(image_path)

    # 2. Preprocess
    target_path, preprocessed_path = _apply_preprocessing(image_path, use_clahe)

    # 3. Check cache
    img_hash = _get_image_hash(target_path)
    model_manager = get_model_manager()
    model_version_key = "-".join(sorted(model_manager._models.keys())) if model_manager._models else "none"
    cache_key = (
        f"{img_hash}_{model_version_key}_"
        f"{'ensemble' if use_ensemble else 'single'}_"
        f"{'tta' if use_tta else 'no_tta'}_"
        f"{'clahe' if use_clahe else 'raw'}"
    )

    cached = get_cache_entry(cache_key)
    if cached is not None:
        result = dict(cached)
        result["cached"] = True
        result["latency"] = time.time() - start_time
        return result

    # 4. Load models if needed
    if use_ensemble and len(model_manager._models) < ENSEMBLE_MIN_MODELS:
        for model_type in MODEL_LIST:
            _ = model_manager.get_model(model_type)
            if len(model_manager._models) >= ENSEMBLE_MIN_MODELS:
                break

    # 5. Run inference
    try:
        if use_ensemble and len(model_manager._models) >= ENSEMBLE_MIN_MODELS:
            final_result, predictions, model_ver, model_source = _run_ensemble(
                model_manager, target_path, use_tta
            )
        else:
            final_result, predictions, model_ver, model_source = _run_single_model(
                model_manager, target_path, use_tta
            )

        latency = time.time() - start_time
        confidence = final_result["confidence"]

        # Log mode
        mode_str = (
            "Ensemble+TTA" if use_ensemble and use_tta
            else "Ensemble" if use_ensemble
            else "TTA" if use_tta
            else "Standard"
        )
        logger.info("%s inference: %s (%.2f) in %.2fs", mode_str, final_result["label"], confidence, latency)

        # 6. Confidence warning
        confidence_warning, confidence_message = _build_confidence_warning(confidence)

        # 7. Uncertainty
        uncertainty_data = _run_uncertainty(model_manager, target_path) if use_uncertainty and use_ensemble else None

        # 8. Grad-CAM
        gradcam_data = None
        if use_gradcam:
            gradcam_data = generate_gradcam_for_image(
                model_manager,
                target_path,
                GRADCAM_MODEL,
                django_settings.MEDIA_ROOT,
                django_settings.MEDIA_URL,
            )

        # 9. Refusal check
        refusal = check_all_refusals(
            confidence=confidence,
            probabilities=final_result.get("probabilities", []),
            predictions=predictions,
            uncertainty_data=uncertainty_data,
            use_ensemble=use_ensemble,
        )

        if refusal.is_refused:
            final_result["label"] = refusal.label
            final_result["confidence"] = 0.0
            confidence = 0.0
            confidence_warning = "low"

        # 10. Build result
        result = {
            "label": final_result["label"],
            "confidence": confidence,
            "probabilities": [float(x) for x in final_result.get("probabilities", [])],
            "model_version": model_ver,
            "model_source": model_source,
            "latency": latency,
            "cached": False,
            "use_ensemble": use_ensemble,
            "use_tta": use_tta,
            "use_clahe": use_clahe,
            "n_models": len(predictions) if predictions else 1,
            "preprocessing_viz_url": None,
            "is_refused": refusal.is_refused,
        }

        if confidence_warning:
            result["confidence_warning"] = confidence_warning
            result["confidence_message"] = confidence_message

        if "warnings" in final_result:
            result["warnings"] = final_result["warnings"]

        if uncertainty_data:
            result["uncertainty"] = {
                "entropy": uncertainty_data.get("entropy", 0.0),
                "is_uncertain": uncertainty_data.get("is_uncertain", False),
                "n_passes": uncertainty_data.get("n_passes", 0),
            }

        if gradcam_data:
            result["gradcam_url"] = gradcam_data["url"]

        if refusal.is_refused:
            result["refusal_message"] = refusal.reason

        set_cache_entry(cache_key, result)

        # Cleanup
        if preprocessed_path and os.path.exists(preprocessed_path):
            try:
                os.remove(preprocessed_path)
                logger.debug("Cleaned up preprocessed file: %s", preprocessed_path)
            except Exception as cleanup_error:
                logger.warning("Failed to cleanup preprocessed file: %s", cleanup_error)

        return result

    except ImageValidationError:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        logger.error("GPU out of memory: %s", exc)
        raise InferenceError("Server overload. Please try again later.") from exc
    except TimeoutError as exc:
        logger.error("Prediction timeout: %s", exc)
        raise InferenceError("Prediction timed out. Please try with a smaller image.") from exc
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise InferenceError(f"Prediction failed: {type(exc).__name__}. Please try again.") from exc
