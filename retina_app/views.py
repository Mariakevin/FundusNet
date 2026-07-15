"""Views for FundusNet — single-page screening tool."""

import csv
import io
import json
import logging
import mimetypes
import os

from django.conf import settings
from django.core.files.storage import default_storage
from django.http import FileResponse, Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render

from .constants import ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, CATEGORIES, MAX_FILE_SIZE
from .forms import ImageUploadForm
from .models import PredictionRecord, UploadedImage
from .services.exceptions import (
    ImageCorruptError,
    ImageDimensionError,
    ImageSizeError,
    ImageValidationError,
    InferenceError,
    ModelLoadError,
    NotAFundusImageError,
    PreprocessingError,
)
from .services.inference import predict_image

logger = logging.getLogger("retina_app")

from .services.model_manager import get_model_loading_status


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

                PredictionRecord.objects.create(
                    uploaded_image=uploaded_image,
                    predicted_class=prediction_data["label"],
                    confidence=prediction_data["confidence"],
                    model_version=prediction_data["model_version"],
                )
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

    model_status = get_model_loading_status()

    return render(
        request,
        "index.html",
        {
            "result": result,
            "error_message": error_message,
            "image_url": image_url,
            "model_status": model_status,
        },
    )


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


def history_view(request: HttpRequest) -> HttpResponse:
    """Show recent prediction history."""
    records = PredictionRecord.objects.filter(is_deleted=False).order_by("-created_at")[:20]
    records_data = [
        {
            "id": r.id,
            "predicted_class": r.predicted_class,
            "confidence": r.confidence * 100,
            "model_version": r.model_version,
            "created_at": r.created_at,
        }
        for r in records
    ]
    return render(request, "history.html", {"records": records_data})


def export_view(request: HttpRequest, record_id: int) -> HttpResponse:
    """Export a single prediction as JSON or CSV."""
    fmt = request.GET.get("format", "json")

    try:
        record = PredictionRecord.objects.get(id=record_id, is_deleted=False)
    except PredictionRecord.DoesNotExist:
        raise Http404("Record not found")

    data = {
        "id": record.id,
        "predicted_class": record.predicted_class,
        "confidence": record.confidence,
        "model_version": record.model_version,
        "patient_identifier": record.patient_identifier,
        "clinical_notes": record.clinical_notes,
        "created_at": record.created_at.isoformat(),
    }

    if fmt == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
        response = HttpResponse(output.getvalue(), content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="prediction_{record_id}.csv"'
        return response

    response = HttpResponse(json.dumps(data, indent=2), content_type="application/json")
    response["Content-Disposition"] = f'attachment; filename="prediction_{record_id}.json"'
    return response


def batch_view(request: HttpRequest) -> HttpResponse:
    """Batch upload view for multiple images."""
    results = []
    error_message = None

    if request.method == "POST":
        files = request.FILES.getlist("images")
        if not files:
            error_message = "No images provided."
        elif len(files) > 100:
            error_message = "Maximum 100 images per batch."
        else:
            for f in files:
                ext = os.path.splitext(f.name)[1].lower()
                if ext not in ALLOWED_EXTENSIONS:
                    results.append({"filename": f.name, "error": f"Invalid file type: {ext}", "success": False})
                    continue
                if f.size > MAX_FILE_SIZE:
                    results.append({"filename": f.name, "error": "File too large (max 10MB)", "success": False})
                    continue
                if f.content_type and f.content_type not in ALLOWED_MIME_TYPES:
                    results.append({"filename": f.name, "error": f"Invalid MIME type: {f.content_type}", "success": False})
                    continue

                uploaded = UploadedImage.objects.create(image=f)
                image_path = os.path.join(settings.MEDIA_ROOT, uploaded.image.name)
                try:
                    prediction_data = predict_image(image_path, use_ensemble=True, use_gradcam=False)
                    PredictionRecord.objects.create(
                        uploaded_image=uploaded,
                        predicted_class=prediction_data["label"],
                        confidence=prediction_data["confidence"],
                        model_version=prediction_data["model_version"],
                    )
                    results.append({
                        "filename": f.name,
                        "label": prediction_data["label"],
                        "confidence": prediction_data["confidence"] * 100,
                        "success": True,
                    })
                except (InferenceError, ImageValidationError, ImageCorruptError, ImageSizeError,
                        ImageDimensionError, ModelLoadError, PreprocessingError, NotAFundusImageError) as exc:
                    results.append({"filename": f.name, "error": str(exc), "success": False})
                except Exception as exc:
                    logger.error("Batch prediction failed for %s: %s", f.name, exc)
                    results.append({"filename": f.name, "error": "Unexpected error", "success": False})

    return render(request, "batch.html", {"results": results, "error_message": error_message})
