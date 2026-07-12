"""REST API endpoints for FundusNet.

Provides:
- Single image inference endpoint
- Batch inference endpoint
- Model health/status endpoint
- Experiment results endpoint
- Model registry management
- Rate limiting and pagination
"""

from __future__ import annotations

import logging
import os
import time

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)


# ── Rate Limiter ──────────────────────────────────────────────────────────────


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = __import__("threading").Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            if key not in self._requests:
                self._requests[key] = []
            cutoff = now - self.window_seconds
            self._requests[key] = [t for t in self._requests[key] if t > cutoff]
            if len(self._requests[key]) >= self.max_requests:
                return False
            self._requests[key].append(now)
            return True

    def get_remaining(self, key: str) -> int:
        now = time.time()
        with self._lock:
            if key not in self._requests:
                return self.max_requests
            cutoff = now - self.window_seconds
            recent = [t for t in self._requests[key] if t > cutoff]
            return max(0, self.max_requests - len(recent))


_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)


def _get_client_ip(request) -> str:
    x_forwarded = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded:
        return x_forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "unknown")


def _rate_limit_check(request) -> JsonResponse | None:
    client_ip = _get_client_ip(request)
    if not _rate_limiter.is_allowed(client_ip):
        remaining = _rate_limiter.get_remaining(client_ip)
        return JsonResponse(
            {"error": "Rate limit exceeded", "retry_after": 60},
            status=429,
            headers={"Retry-After": "60", "X-RateLimit-Remaining": str(remaining)},
        )
    return None


# ── Pagination ────────────────────────────────────────────────────────────────


def _paginate(request, items: list, default_limit: int = 20) -> tuple[list, dict]:
    """Apply cursor-based pagination to a list of items."""
    try:
        limit = int(request.GET.get("limit", default_limit))
        offset = int(request.GET.get("offset", 0))
    except (ValueError, TypeError):
        limit = default_limit
        offset = 0

    limit = min(limit, 100)
    paginated = items[offset : offset + limit]

    meta = {
        "total": len(items),
        "limit": limit,
        "offset": offset,
        "has_next": offset + limit < len(items),
        "has_prev": offset > 0,
    }
    return paginated, meta


# ── API Endpoints ─────────────────────────────────────────────────────────────


@csrf_exempt
@require_http_methods(["POST"])
def predict_single(request):
    """Single image inference endpoint.

    POST /api/v1/predict/
    Body: multipart/form-data with 'image' field
    Optional params: use_ensemble, use_tta, use_gradcam
    """
    rate_limit = _rate_limit_check(request)
    if rate_limit:
        return rate_limit

    if "image" not in request.FILES:
        return JsonResponse({"error": "No image provided"}, status=400)

    image_file = request.FILES["image"]

    # Validate file
    if image_file.size > 10 * 1024 * 1024:
        return JsonResponse({"error": "File too large (max 10MB)"}, status=400)

    allowed_types = ["image/jpeg", "image/png"]
    if image_file.content_type not in allowed_types:
        return JsonResponse({"error": "Invalid file type. Use JPG or PNG."}, status=400)

    # Save temp file
    ext = os.path.splitext(image_file.name)[1]
    tmp_path = os.path.join(settings.MEDIA_ROOT, f"api_upload_{int(time.time())}{ext}")

    try:
        with open(tmp_path, "wb") as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Parse options
        use_ensemble = request.POST.get("use_ensemble", "true").lower() == "true"
        use_tta = request.POST.get("use_tta", "false").lower() == "true"
        use_gradcam = request.POST.get("use_gradcam", "true").lower() == "true"

        from retina_app.services.inference import predict_image

        start = time.time()
        result = predict_image(
            tmp_path,
            use_ensemble=use_ensemble,
            use_tta=use_tta,
            use_gradcam=use_gradcam,
        )
        latency = time.time() - start

        return JsonResponse(
            {
                "success": True,
                "result": result,
                "api_latency": round(latency, 4),
            }
        )

    except Exception as e:
        logger.error(f"API prediction failed: {e}")
        return JsonResponse({"error": str(e)}, status=500)

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@csrf_exempt
@require_http_methods(["POST"])
def predict_batch(request):
    """Batch inference endpoint.

    POST /api/v1/predict/batch/
    Body: JSON with 'image_paths' list and optional config
    """
    rate_limit = _rate_limit_check(request)
    if rate_limit:
        return rate_limit

    import json

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    image_paths = data.get("image_paths", [])
    if not image_paths:
        return JsonResponse({"error": "No image paths provided"}, status=400)

    if len(image_paths) > 100:
        return JsonResponse({"error": "Maximum 100 images per batch"}, status=400)

    # Validate all paths exist
    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        return JsonResponse(
            {"error": f"Files not found: {missing[:5]}"},
            status=400,
        )

    config = data.get("config", {})
    priority = data.get("priority", 2)  # NORMAL

    from retina_app.services.batch_inference import get_batch_service

    service = get_batch_service()
    job_id = service.submit_job(
        image_paths=image_paths,
        config=config,
        priority=priority,
    )

    return JsonResponse(
        {
            "success": True,
            "job_id": job_id,
            "status_url": f"/api/v1/jobs/{job_id}/",
            "total_images": len(image_paths),
        }
    )


@require_http_methods(["GET"])
def job_status(request, job_id: str):
    """Get job status and results.

    GET /api/v1/jobs/<job_id>/
    """
    from retina_app.services.batch_inference import get_batch_service

    service = get_batch_service()
    status = service.get_job_status(job_id)

    if status is None:
        return JsonResponse({"error": "Job not found"}, status=404)

    return JsonResponse(status)


@require_http_methods(["GET"])
def model_health(request):
    """Get model health status.

    GET /api/v1/health/
    """
    from retina_app.services.model_manager import get_health_tracker

    tracker = get_health_tracker()
    health = tracker.get_all_health()

    return JsonResponse(
        {
            "status": "healthy",
            "models": health,
            "timestamp": time.time(),
        }
    )


@require_http_methods(["GET"])
def model_registry(request):
    """List registered models.

    GET /api/v1/registry/
    Optional: ?model_name=convnext_v2
    """
    from retina_app.ml.registry import ModelRegistry

    registry = ModelRegistry(registry_dir=os.path.join(settings.BASE_DIR, "model_registry"))

    model_name = request.GET.get("model_name")
    artifacts = registry.list_models(model_name)

    items = [
        {
            "model": a.model_name,
            "version": a.version,
            "stage": a.stage,
            "metrics": a.metrics,
            "created_at": a.created_at,
            "tags": a.tags,
        }
        for a in artifacts
    ]

    paginated, meta = _paginate(request, items)
    return JsonResponse({"data": paginated, "meta": meta})


@require_http_methods(["GET"])
def leaderboard(request):
    """Get model leaderboard.

    GET /api/v1/leaderboard/
    """
    from retina_app.ml.registry import ModelRegistry

    registry = ModelRegistry(registry_dir=os.path.join(settings.BASE_DIR, "model_registry"))

    model_name = request.GET.get("model_name")
    entries = registry.leaderboard(model_name)

    paginated, meta = _paginate(request, entries)
    return JsonResponse({"data": paginated, "meta": meta})


@require_http_methods(["GET"])
def experiments(request):
    """List training experiments.

    GET /api/v1/experiments/
    """
    from retina_app.ml.registry import ExperimentTracker

    tracker = ExperimentTracker(tracker_dir=os.path.join(settings.BASE_DIR, "experiments"))

    experiment_list = tracker.list_experiments()
    items = [
        {
            "name": e.experiment_name,
            "model": e.model_name,
            "mean_val_acc": e.mean_metrics.get("val_acc", 0),
            "std_val_acc": e.std_metrics.get("val_acc", 0),
            "duration": e.duration_seconds,
            "created_at": e.created_at,
        }
        for e in experiment_list
    ]

    paginated, meta = _paginate(request, items)
    return JsonResponse({"data": paginated, "meta": meta})


@require_http_methods(["GET"])
def service_stats(request):
    """Get batch inference service statistics.

    GET /api/v1/stats/
    """
    from retina_app.services.batch_inference import get_batch_service

    service = get_batch_service()
    stats = service.get_stats()

    return JsonResponse({"stats": stats})


@require_http_methods(["GET"])
def api_root(request):
    """API root — lists all available endpoints.

    GET /api/v1/
    """
    return JsonResponse(
        {
            "name": "FundusNet API",
            "version": "v1",
            "endpoints": {
                "predict": "/api/v1/predict/",
                "predict_batch": "/api/v1/predict/batch/",
                "jobs": "/api/v1/jobs/<job_id>/",
                "health": "/api/v1/health/",
                "registry": "/api/v1/registry/",
                "leaderboard": "/api/v1/leaderboard/",
                "experiments": "/api/v1/experiments/",
                "stats": "/api/v1/stats/",
            },
        }
    )
