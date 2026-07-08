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
    ADAPTIVE_CLAHE_ENABLED,
    CATEGORIES,
    COLOR_CONSTANCY_ENABLED,
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_LOW,
    CONFIDENCE_THRESHOLD_REFUSE,
    ENABLE_MC_DROPOUT,
    ENSEMBLE_MIN_MODELS,
    FUNDUS_MIN_TOP1_TOP2_RATIO,
    FUNDUS_VALIDATION_ENABLED,
    GRADCAM_MODEL,
    MAX_WORKERS,
    MODEL_LIST,
    MODEL_WEIGHTS,
    NOISE_REDUCTION_ENABLED,
    OOD_ENTROPY_THRESHOLD,
    PREPROCESSING_VIZ_ENABLED,
    UNCERTAINTY_REFUSAL_MESSAGE,
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
from retina_app.services.gradcam import generate_gradcam, get_gradcam_output_path
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
    apply_adaptive_clahe,
    apply_clahe,
    apply_color_constancy,
    check_image_quality,
    generate_preprocessing_viz,
    preprocess_fundus,
    reduce_noise,
    save_preprocessing_viz,
    validate_image_file,
)

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

    # ... (validation and preprocessing unchanged) ...

    try:
        validate_image_file(image_path)
    except ImageValidationError as exc:
        logger.warning(f"Image validation failed for {image_path}: {exc}")
        raise InferenceError(f"Invalid image: {exc}") from exc

    # --- Fundus Image Validation ---
    if FUNDUS_VALIDATION_ENABLED:
        fundus_check = validate_fundus_image(image_path)
        if not fundus_check["is_fundus"]:
            logger.warning(
                "Non-fundus image rejected: score=%.3f, signals=%s, path=%s",
                fundus_check["confidence"],
                fundus_check["signals"],
                image_path,
            )
            raise NotAFundusImageError(fundus_check["message"])

    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as exc:
        logger.warning("Corrupted image file %s: %s", image_path, exc)
        raise ImageCorruptError(f"Corrupted or unreadable image: {exc}") from exc

    try:
        quality_check = check_image_quality(image_path, quality_threshold=0.25)
        if not quality_check["passed"]:
            logger.warning(f"Image quality check failed: {quality_check['quality_level']}")
    except Exception as exc:
        logger.warning(f"Quality check failed, continuing: {exc}")

    preprocessed_path = None
    if use_clahe:
        try:
            preprocessed = preprocess_fundus(image_path, enhance=False, detect_roi=False)
            if ADAPTIVE_CLAHE_ENABLED:
                preprocessed = apply_adaptive_clahe(preprocessed)
            else:
                preprocessed = apply_clahe(preprocessed)
            if NOISE_REDUCTION_ENABLED:
                preprocessed = reduce_noise(preprocessed)
            if COLOR_CONSTANCY_ENABLED:
                preprocessed = apply_color_constancy(preprocessed)
            ext = os.path.splitext(image_path)[1]
            img_hash = _get_image_hash(image_path)
            tmp_dir = tempfile.gettempdir()
            preprocessed_path = os.path.join(tmp_dir, f"fundus_{img_hash}_clahe{ext}")
            cv2.imwrite(preprocessed_path, cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR))
            logger.info("Applied preprocessing pipeline to %s", image_path)
        except Exception as exc:
            logger.warning("Preprocessing failed, using original: %s", exc)
            preprocessed_path = None

    target_path = preprocessed_path if preprocessed_path else image_path

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
        cached["cached"] = True
        cached["latency"] = time.time() - start_time
        return cached

    if use_ensemble and len(model_manager._models) < ENSEMBLE_MIN_MODELS:
        for model_type in MODEL_LIST:
            _ = model_manager.get_model(model_type)
            if len(model_manager._models) >= ENSEMBLE_MIN_MODELS:
                break

    n_models_used = 1
    try:
        if use_ensemble and len(model_manager._models) >= ENSEMBLE_MIN_MODELS:
            logger.info(
                f"Running ensemble with {len(model_manager._models)} models: {list(model_manager._models.keys())}"
            )

            predictions = predict_models_parallel(model_manager._models, target_path, use_tta, get_executor())

            health_tracker = get_health_tracker()
            health_tracker.record_ensemble_prediction(predictions)

            disagreement = detect_model_disagreement(predictions)
            if disagreement["disagreement"]:
                logger.info(
                    f"Model disagreement detected: {disagreement['class_votes']}, "
                    f"agreement={disagreement['agreement_level']:.2f}"
                )

            final_result = selective_ensemble(predictions)
            n_models_used = final_result.get("n_models", len(predictions))
            model_ver = f"ensemble-v{n_models_used}-models-tta" if use_tta else f"ensemble-v{n_models_used}-models"

            model_types = [model_manager._model_types.get(mt, "unknown") for mt, _ in predictions]
            has_trained = "trained" in model_types
            model_source = "trained" if has_trained else "pretrained"
        else:
            model = model_manager.get_model(MODEL_LIST[0])
            final_result = _predict_single_model(model, target_path, use_tta=use_tta)
            model_ver = MODEL_VERSIONS.get(MODEL_LIST[0], "demo")
            model_source = model_manager._model_types.get(MODEL_LIST[0], "pretrained")

        latency = time.time() - start_time

        mode_str = (
            "Ensemble+TTA"
            if use_ensemble and use_tta
            else "Ensemble"
            if use_ensemble
            else "TTA"
            if use_tta
            else "Standard"
        )

        confidence = final_result["confidence"]
        confidence_warning = None
        if confidence < CONFIDENCE_THRESHOLD_LOW:
            confidence_warning = "low"
            logger.warning(f"Low confidence prediction: {confidence:.2f} < {CONFIDENCE_THRESHOLD_LOW}")
        elif confidence < CONFIDENCE_THRESHOLD_HIGH:
            confidence_warning = "medium"

        logger.info(
            "%s inference: %s (%.2f) in %.2fs",
            mode_str,
            final_result["label"],
            confidence,
            latency,
        )

        # --- MC Dropout Uncertainty ---
        uncertainty_data = None
        if use_uncertainty and ENABLE_MC_DROPOUT and use_ensemble:
            try:
                uncertainty_data = predict_with_uncertainty_ensemble(
                    model_manager._models,
                    target_path,
                    model_weights=MODEL_WEIGHTS,
                )
                logger.info(
                    f"MC Dropout uncertainty: entropy={uncertainty_data['entropy']:.4f}, "
                    f"uncertain={uncertainty_data['is_uncertain']}"
                )
            except Exception as exc:
                logger.warning(f"MC Dropout uncertainty failed: {exc}")

        # --- Grad-CAM Explainability ---
        gradcam_data = None
        if use_gradcam:
            try:
                gradcam_model_type = GRADCAM_MODEL
                if gradcam_model_type in model_manager._models:
                    gradcam_model = model_manager._models[gradcam_model_type]
                    gradcam_output = get_gradcam_output_path(django_settings.MEDIA_ROOT, os.path.basename(image_path))
                    gradcam_result = generate_gradcam(
                        gradcam_model,
                        target_path,
                        gradcam_model_type,
                        output_path=gradcam_output,
                    )
                    gradcam_url = f"{django_settings.MEDIA_URL}gradcam/{os.path.basename(gradcam_output)}"
                    gradcam_data = {
                        "url": gradcam_url,
                        "predicted_class": gradcam_result["predicted_class"],
                        "confidence": gradcam_result["confidence"],
                    }
                    logger.info("Grad-CAM generated: %s", gradcam_url)
            except Exception as exc:
                logger.warning("Grad-CAM generation failed: %s", exc)

        # --- Selective Refusal ---
        is_refused = False
        if uncertainty_data and uncertainty_data.get("is_uncertain"):
            is_refused = True
            final_result["label"] = "Uncertain"
            final_result["confidence"] = 0.0
            confidence_warning = "low"
            logger.warning(f"Classification refused: uncertainty={uncertainty_data['entropy']:.4f} > threshold")
        elif confidence < CONFIDENCE_THRESHOLD_REFUSE:
            is_refused = True
            final_result["label"] = "Uncertain"
            final_result["confidence"] = 0.0
            confidence_warning = "low"
            logger.warning(f"Classification refused: low confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD_REFUSE}")

        # OOD detection via normalized entropy of ensemble probabilities.
        # OOD images produce near-uniform predictions (high entropy) even when
        # the max confidence is above the refusal threshold. In-distribution
        # fundus images have peaked distributions (low entropy).
        if not is_refused and use_ensemble:
            probs = final_result.get("probabilities")
            if probs and len(probs) > 1:
                import numpy as np
                from scipy.stats import entropy as scipy_entropy

                norm_entropy = scipy_entropy(probs) / np.log(len(probs))
                if norm_entropy > OOD_ENTROPY_THRESHOLD:
                    is_refused = True
                    final_result["label"] = "Uncertain"
                    confidence = 0.0
                    confidence_warning = "low"
                    logger.warning(f"Classification refused: OOD entropy {norm_entropy:.4f} > {OOD_ENTROPY_THRESHOLD}")

        # Top-1 / Top-2 margin check — OOD images have a narrow margin between
        # the top class and the runner-up, while real fundus predictions have a
        # clear winner.
        if not is_refused and use_ensemble:
            probs = final_result.get("probabilities")
            if probs and len(probs) >= 2:
                sorted_probs = sorted(probs, reverse=True)
                margin = sorted_probs[0] / max(sorted_probs[1], 1e-8)
                if margin < FUNDUS_MIN_TOP1_TOP2_RATIO:
                    is_refused = True
                    final_result["label"] = "Uncertain"
                    confidence = 0.0
                    confidence_warning = "low"
                    logger.warning(
                        f"Classification refused: top-1/top-2 margin {margin:.2f}"
                        f" < {FUNDUS_MIN_TOP1_TOP2_RATIO}"
                    )

        # Energy-based OOD detection — uses raw logits (before softmax).
        # OOD images produce low-energy logits across ALL classes, even when
        # the softmax-normalized top class seems confident. Energy score:
        #   E(x) = log(sum(exp(logits_i)))
        # Higher = in-distribution, Lower = OOD.
        if not is_refused and use_ensemble:
            all_logits = []
            for _, pred in predictions:
                logits = pred.get("logits")
                if logits and len(logits) == len(CATEGORIES):
                    all_logits.append(logits)
            if len(all_logits) >= 2:
                avg_logits = np.mean(all_logits, axis=0)
                energy = np.log(np.sum(np.exp(avg_logits - np.max(avg_logits)))) + np.max(avg_logits)
                n_classes = len(avg_logits)
                # Normalize energy by the number of classes
                # (higher n_classes = higher energy naturally)
                energy_per_class = energy / n_classes
                # In-distribution fundus images typically have energy_per_class > 0.8
                if energy_per_class < 0.6:
                    is_refused = True
                    final_result["label"] = "Uncertain"
                    confidence = 0.0
                    confidence_warning = "low"
                    logger.warning(
                        f"Classification refused: low energy score {energy_per_class:.4f}"
                    )

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
            "n_models": n_models_used,
        }

        if confidence_warning:
            result["confidence_warning"] = confidence_warning
            result["confidence_message"] = (
                "Low confidence. Result may be unreliable. Consider retaking the image with better lighting."
                if confidence_warning == "low"
                else "Medium confidence. Result is reasonably reliable."
            )

        if "warnings" in final_result:
            result["warnings"] = final_result["warnings"]

        # Uncertainty data
        if uncertainty_data:
            result["uncertainty"] = {
                "entropy": uncertainty_data.get("entropy", 0.0),
                "is_uncertain": uncertainty_data.get("is_uncertain", False),
                "n_passes": uncertainty_data.get("n_passes", 0),
            }

        # Grad-CAM data
        if gradcam_data:
            result["gradcam_url"] = gradcam_data["url"]

        # Selective refusal
        if is_refused:
            result["is_refused"] = True
            result["refusal_message"] = UNCERTAINTY_REFUSAL_MESSAGE
        else:
            result["is_refused"] = False

        # --- Preprocessing Visualization ---
        if PREPROCESSING_VIZ_ENABLED:
            try:
                viz_panels = generate_preprocessing_viz(image_path)
                viz_dir = os.path.join(django_settings.MEDIA_ROOT, "preprocessing_viz")
                os.makedirs(viz_dir, exist_ok=True)
                viz_filename = f"viz_{os.path.basename(image_path)}"
                viz_path = os.path.join(viz_dir, viz_filename)
                save_preprocessing_viz(viz_panels, viz_path)
                viz_url = f"{django_settings.MEDIA_URL}preprocessing_viz/{viz_filename}"
                result["preprocessing_viz_url"] = viz_url
                logger.info("Preprocessing visualization saved: %s", viz_url)
            except Exception as exc:
                logger.warning("Preprocessing visualization failed: %s", exc)

        set_cache_entry(cache_key, result)

        if preprocessed_path and os.path.exists(preprocessed_path):
            try:
                os.remove(preprocessed_path)
                logger.debug(f"Cleaned up preprocessed file: {preprocessed_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup preprocessed file: {cleanup_error}")

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
