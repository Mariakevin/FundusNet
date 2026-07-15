from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path


def handler400(request, exception):
    return JsonResponse({"error": "Bad request", "code": 400}, status=400)


def handler403(request, exception):
    return JsonResponse({"error": "Forbidden", "code": 403}, status=403)


def handler404(request, exception):
    return JsonResponse({"error": "Not found", "code": 404}, status=404)


def handler500(request):
    return JsonResponse({"error": "Internal server error", "code": 500}, status=500)


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("retina_app.urls")),
]
