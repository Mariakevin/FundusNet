from django import forms

from .models import UploadedImage
from .constants import ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, MAX_FILE_SIZE


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ["image"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["image"].widget.attrs.update({
            "class": "form-input",
            "accept": "image/jpeg,image/png,image/bmp,image/webp",
            "required": True,
            "data-max-size": str(MAX_FILE_SIZE),
        })

    def clean_image(self):
        image = self.cleaned_data.get("image")
        if not image:
            raise forms.ValidationError("Please upload a retinal image.")

        extension = "." + image.name.split(".")[-1].lower() if "." in image.name else ""
        if extension not in ALLOWED_EXTENSIONS:
            raise forms.ValidationError(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

        content_type = getattr(image, "content_type", "")
        if content_type and content_type not in ALLOWED_MIME_TYPES:
            raise forms.ValidationError("Invalid file type. Please upload a supported image format.")

        if image.size > MAX_FILE_SIZE:
            raise forms.ValidationError(f"File is too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB.")
        
        return image
