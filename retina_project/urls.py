from django.contrib import admin
from django.http import HttpRequest, JsonResponse
from django.urls import include, path
from django.shortcuts import render


def _is_api_request(request: HttpRequest) -> bool:
    """Check if the request is for an API endpoint."""
    return request.path.startswith("/api/") or request.content_type == "application/json"


def handler400(request: HttpRequest, exception):
    if _is_api_request(request):
        return JsonResponse({"error": "Bad request", "code": 400}, status=400)
    return render(request, "errors/400.html", status=400)


def handler403(request: HttpRequest, exception):
    if _is_api_request(request):
        return JsonResponse({"error": "Forbidden", "code": 403}, status=403)
    return render(request, "errors/403.html", status=403)


def handler404(request: HttpRequest, exception):
    if _is_api_request(request):
        return JsonResponse({"error": "Not found", "code": 404}, status=404)
    return render(request, "errors/404.html", status=404)


def handler500(request: HttpRequest):
    if _is_api_request(request):
        return JsonResponse({"error": "Internal server error", "code": 500}, status=500)
    return render(request, "errors/500.html", status=500)


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("retina_app.urls")),
]
