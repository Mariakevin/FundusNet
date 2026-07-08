from django.contrib import admin
from .models import PredictionRecord, UploadedImage


@admin.register(UploadedImage)
class UploadedImageAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "image", "uploaded_at")
    list_filter = ("uploaded_at",)
    search_fields = ("user__username",)
    list_select_related = ("user",)
    date_hierarchy = "uploaded_at"


@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "predicted_class", "confidence", "model_version", "created_at")
    list_filter = ("predicted_class", "model_version", "created_at")
    search_fields = ("user__username", "predicted_class", "model_version")
    list_select_related = ("user", "uploaded_image")
    date_hierarchy = "created_at"
