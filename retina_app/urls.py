from django.urls import path

from .api import (
    api_root,
    experiments,
    job_status,
    leaderboard,
    model_health,
    model_registry,
    predict_batch,
    predict_single,
    service_stats,
)
from .views import batch_view, export_view, history_view, index_view, protected_media

urlpatterns = [
    path("", index_view, name="index"),
    path("history/", history_view, name="history"),
    path("export/<int:record_id>/", export_view, name="export"),
    path("batch/", batch_view, name="batch"),
    path("media/<path:path>", protected_media, name="protected_media"),
    # API v1
    path("api/v1/", api_root, name="api-root"),
    path("api/v1/predict/", predict_single, name="api-predict"),
    path("api/v1/predict/batch/", predict_batch, name="api-predict-batch"),
    path("api/v1/jobs/<str:job_id>/", job_status, name="api-job-status"),
    path("api/v1/health/", model_health, name="api-health"),
    path("api/v1/registry/", model_registry, name="api-registry"),
    path("api/v1/leaderboard/", leaderboard, name="api-leaderboard"),
    path("api/v1/experiments/", experiments, name="api-experiments"),
    path("api/v1/stats/", service_stats, name="api-stats"),
]
