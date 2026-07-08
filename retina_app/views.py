"""
Views for FundusNet — single-page screening tool.
"""

import os
import mimetypes
import logging
from django.http import HttpRequest, HttpResponse, FileResponse, Http404
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage

from .forms import ImageUploadForm
from .models import UploadedImage
from .constants import CATEGORIES
from .services.inference import predict_image
from .services.exceptions import (
    InferenceError,
    ImageValidationError,
    ImageCorruptError,
    ImageSizeError,
    ImageDimensionError,
    ModelLoadError,
    PreprocessingError,
    NotAFundusImageError,
)

logger = logging.getLogger("retina_app")


def index_view(request: HttpRequest) -> HttpResponse:
    """Single-page view: upload zone + inline result display."""
    result = None
    error_message = None
    image_url = None

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()
            image_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.image.name)
            image_url = uploaded_image.image.url

            try:
                prediction_data = predict_image(image_path, use_ensemble=True)
                probabilities = prediction_data.get("probabilities", [])
                prob_labels = []
                if probabilities:
                    prob_labels = [(cat, prob * 100) for cat, prob in zip(CATEGORIES, probabilities)]
                    prob_labels.sort(key=lambda x: x[1], reverse=True)

                result = {
                    "prediction": prediction_data["label"],
                    "confidence": prediction_data["confidence"] * 100,
                    "model_version": prediction_data["model_version"],
                    "image_url": image_url,
                    "prob_labels": prob_labels,
                    "uncertainty": prediction_data.get("uncertainty"),
                    "gradcam_url": prediction_data.get("gradcam_url"),
                    "is_refused": prediction_data.get("is_refused", False),
                    "refusal_message": prediction_data.get("refusal_message", ""),
                    "preprocessing_viz_url": prediction_data.get("preprocessing_viz_url"),
                }
            except InferenceError as exc:
                error_message = f"Analysis failed: {str(exc)}"
            except (ImageValidationError, ImageCorruptError, ImageSizeError, ImageDimensionError) as exc:
                error_message = f"Image error: {str(exc)}"
            except ModelLoadError:
                error_message = "Model unavailable. Please try again later."
            except PreprocessingError as exc:
                error_message = f"Processing failed: {str(exc)}"
            except NotAFundusImageError as exc:
                error_message = str(exc)
        else:
            error_message = "Invalid image. Please select a JPG or PNG file under 10MB."

    return render(request, "index.html", {
        "result": result,
        "error_message": error_message,
        "image_url": image_url,
    })


def protected_media(request: HttpRequest, path: str) -> HttpResponse:
    """Serve media files without auth."""
    full_path = os.path.join(settings.MEDIA_ROOT, path)

    try:
        real_path = os.path.realpath(full_path)
        media_root = os.path.realpath(settings.MEDIA_ROOT)
    except OSError:
        raise Http404("File not found")

    if not real_path.startswith(media_root):
        raise Http404("File not found")

    if not default_storage.exists(full_path):
        raise Http404("File not found")

    content_type, _ = mimetypes.guess_type(full_path)
    if not content_type:
        content_type = "application/octet-stream"

    try:
        file = default_storage.open(full_path)
        response = FileResponse(file, content_type=content_type)
        response["Cache-Control"] = "public, max-age=86400"
        return response
    except FileNotFoundError:
        raise Http404("File not found")
