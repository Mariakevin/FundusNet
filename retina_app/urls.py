from django.urls import path
from .views import index_view, protected_media

urlpatterns = [
    path("", index_view, name="index"),
    path("media/<path:path>", protected_media, name="protected_media"),
]
