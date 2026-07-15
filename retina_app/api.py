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

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from retina_app.constants import ALLOWED_MIME_TYPES, MAX_FILE_SIZE

logger = logging.getLogger(__name__)


# ── API Key Authentication ─────────────────────────────────────────────────────

API_KEY_HEADER = "X-API-Key"


def _get_api_keys() -> set[str]:
    """Load API keys from environment or settings."""
    keys: set[str] = set()
    env_keys = os.environ.get("FUNDUSNET_API_KEYS", "")
    if env_keys:
        keys.update(k.strip() for k in env_keys.split(",") if k.strip())
    settings_key = getattr(settings, "API_KEY", None)
    if settings_key:
        keys.add(settings_key)
    return keys


def _check_api_key(request) -> JsonResponse | None:
    """Verify API key. Returns None if valid, JsonResponse if not."""
    api_keys = _get_api_keys()
    if not api_keys:
        return None  # No keys configured = open access

    provided = request.META.get(f"HTTP_{API_KEY_HEADER.replace('-', '_').upper()}", "")
    if not provided:
        provided = request.headers.get(API_KEY_HEADER, "")

    if not provided or provided not in api_keys:
        return JsonResponse(
            {"error": "Invalid or missing API key", "header": API_KEY_HEADER},
            status=401,
        )
    return None


# ── File-Based Rate Limiter (cross-worker) ─────────────────────────────────────


class FileRateLimiter:
    """File-based rate limiter using atomic writes for cross-worker safety.

    Uses tempfile + rename for atomic writes. Each key gets its own file.
    Race condition is minimal because:
    1. We read, filter, append, write atomically per-process
    2. Worst case: two workers both allow a request (slight over-limit)
    3. Rate limit is soft, not security-critical
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._dir = os.path.join(tempfile.gettempdir(), "fundusnet_ratelimit")
        os.makedirs(self._dir, exist_ok=True)
        self._global_lock = threading.Lock()

    def _get_file(self, key: str) -> str:
        safe = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self._dir, f"{safe}.json")

    def _get_lock(self, key: str) -> threading.Lock:
        return self._global_lock

    def _read_timestamps(self, filepath: str) -> list[float]:
        try:
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    data = json.load(f)
                return data.get("timestamps", [])
        except (json.JSONDecodeError, OSError):
            pass
        return []

    def _write_timestamps(self, filepath: str, timestamps: list[float]) -> None:
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=self._dir, suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump({"timestamps": timestamps}, f)
                os.replace(tmp_path, filepath)
            except OSError:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except OSError:
            pass

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        filepath = self._get_file(key)
        lock = self._get_lock(key)

        with lock:
            raw_timestamps = self._read_timestamps(filepath)
            timestamps = [t for t in raw_timestamps if t > now - self.window_seconds]

            if len(timestamps) >= self.max_requests:
                return False

            timestamps.append(now)
            self._write_timestamps(filepath, timestamps)
            return True

    def get_remaining(self, key: str) -> int:
        now = time.time()
        filepath = self._get_file(key)

        raw_timestamps = self._read_timestamps(filepath)
        timestamps = [t for t in raw_timestamps if t > now - self.window_seconds]
        return max(0, self.max_requests - len(timestamps))


_rate_limiter = FileRateLimiter(max_requests=30, window_seconds=60)


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
    Headers: X-API-Key (if configured)
    Body: multipart/form-data with 'image' field
    Optional params: use_ensemble, use_tta, use_gradcam
    """
    auth_error = _check_api_key(request)
    if auth_error:
        return auth_error

    rate_limit = _rate_limit_check(request)
    if rate_limit:
        return rate_limit

    if "image" not in request.FILES:
        return JsonResponse({"error": "No image provided"}, status=400)

    image_file = request.FILES["image"]

    # Validate file
    if image_file.size > MAX_FILE_SIZE:
        return JsonResponse({"error": "File too large (max 10MB)"}, status=400)

    if image_file.content_type not in ALLOWED_MIME_TYPES:
        return JsonResponse({"error": "Invalid file type. Use JPG, PNG, BMP, WEBP, or TIFF."}, status=400)

    # Save temp file with UUID to avoid race condition
    ext = os.path.splitext(image_file.name)[1]
    tmp_path = os.path.join(settings.MEDIA_ROOT, f"api_upload_{uuid.uuid4().hex[:12]}{ext}")

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
        logger.error("API prediction failed: %s", e)
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
    Headers: X-API-Key (if configured)
    Body: JSON with 'image_paths' list and optional config
    """
    auth_error = _check_api_key(request)
    if auth_error:
        return auth_error

    rate_limit = _rate_limit_check(request)
    if rate_limit:
        return rate_limit

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    image_paths = data.get("image_paths", [])
    if not image_paths:
        return JsonResponse({"error": "No image paths provided"}, status=400)

    if len(image_paths) > 100:
        return JsonResponse({"error": "Maximum 100 images per batch"}, status=400)

    # Validate all paths exist and are within allowed directories
    allowed_roots = [
        os.path.realpath(settings.MEDIA_ROOT),
        os.path.realpath(os.path.join(settings.BASE_DIR, "retina_dataset")),
    ]

    def _is_path_allowed(path: str) -> bool:
        real_path = os.path.realpath(path)
        return any(real_path.startswith(root) for root in allowed_roots)

    invalid_paths = [p for p in image_paths if not os.path.exists(p) or not _is_path_allowed(p)]
    if invalid_paths:
        return JsonResponse(
            {"error": f"Invalid or restricted paths: {invalid_paths[:5]}"},
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

    overall_status = "healthy"
    for model_name, model_health in health.items():
        if isinstance(model_health, dict) and model_health.get("status") != "healthy":
            overall_status = "degraded"
            break

    return JsonResponse(
        {
            "status": overall_status,
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
