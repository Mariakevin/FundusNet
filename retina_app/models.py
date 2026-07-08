from django.db import models


class UploadedImage(models.Model):
    user = models.ForeignKey(
        "auth.User", on_delete=models.SET_NULL, null=True, blank=True
    )
    image = models.ImageField(upload_to="uploads/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.image.name


class PredictionRecord(models.Model):
    user = models.ForeignKey(
        "auth.User", on_delete=models.SET_NULL, null=True, blank=True,
        related_name="predictions"
    )
    uploaded_image = models.ForeignKey(
        UploadedImage, on_delete=models.SET_NULL, related_name="predictions",
        null=True, blank=True
    )
    patient_identifier = models.CharField(max_length=50, blank=True)
    clinical_notes = models.TextField(blank=True)
    predicted_class = models.CharField(max_length=64)
    confidence = models.FloatField()
    model_version = models.CharField(max_length=128, default="ensemble-v3")
    created_at = models.DateTimeField(auto_now_add=True)
    is_deleted = models.BooleanField(default=False)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["-created_at"]),
            models.Index(fields=["predicted_class"]),
        ]

    def __str__(self):
        return f"{self.predicted_class} ({self.confidence:.2%})"
