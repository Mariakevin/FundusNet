"""Tests for the SPA-only RetinaAI app — index_view and protected_media."""
from django.test import TestCase, override_settings
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from io import BytesIO
from PIL import Image
from unittest.mock import patch
import tempfile
import os

from retina_app.models import UploadedImage


def create_test_image_bytes(fmt="JPEG", size=(100, 100), color="red"):
    img = Image.new("RGB", size, color=color)
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


class IndexViewGetTests(TestCase):
    """GET / should show the upload form."""

    def test_get_returns_200(self):
        response = self.client.get(reverse("index"))
        self.assertEqual(response.status_code, 200)

    def test_get_shows_upload_form(self):
        response = self.client.get(reverse("index"))
        self.assertContains(response, "input")
        self.assertContains(response, "form")

    def test_get_no_result(self):
        response = self.client.get(reverse("index"))
        self.assertNotContains(response, "Prediction Result")


class IndexViewPostTests(TestCase):
    """POST / should analyze an image and return inline results."""

    def setUp(self):
        self.url = reverse("index")

    def _valid_image(self):
        return SimpleUploadedFile("test.jpg", create_test_image_bytes(), content_type="image/jpeg")

    def test_post_valid_image_200(self):
        response = self.client.post(self.url, {"image": self._valid_image()})
        self.assertEqual(response.status_code, 200)

    @patch("retina_app.views.predict_image")
    def test_post_valid_image_shows_result(self, mock_predict):
        mock_predict.return_value = {
            "label": "Healthy",
            "confidence": 0.95,
            "model_version": "ensemble-v3",
            "probabilities": [0.95, 0.02, 0.02, 0.01],
            "uncertainty": 0.05,
            "gradcam_url": None,
            "is_refused": False,
            "refusal_message": "",
            "preprocessing_viz_url": None,
        }
        response = self.client.post(self.url, {"image": self._valid_image()})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Healthy")
        self.assertContains(response, "95.0")

    @patch("retina_app.views.predict_image")
    def test_post_returns_prob_labels(self, mock_predict):
        mock_predict.return_value = {
            "label": "Cataract",
            "confidence": 0.82,
            "model_version": "ensemble-v3",
            "probabilities": [0.08, 0.82, 0.06, 0.04],
            "uncertainty": 0.12,
            "gradcam_url": None,
            "is_refused": False,
            "refusal_message": "",
            "preprocessing_viz_url": None,
        }
        response = self.client.post(self.url, {"image": self._valid_image()})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cataract")

    def test_post_no_file_shows_error(self):
        response = self.client.post(self.url, {})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Invalid image")

    def test_post_invalid_file_type_shows_error(self):
        bad_file = SimpleUploadedFile("test.txt", b"not an image", content_type="text/plain")
        response = self.client.post(self.url, {"image": bad_file})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Invalid image")

    @patch("retina_app.views.predict_image")
    def test_post_inference_error_shows_error(self, mock_predict):
        from retina_app.services.exceptions import InferenceError
        mock_predict.side_effect = InferenceError("Test inference error")
        response = self.client.post(self.url, {"image": self._valid_image()})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Analysis failed")

    @patch("retina_app.views.predict_image")
    def test_post_model_load_error_shows_error(self, mock_predict):
        from retina_app.services.exceptions import ModelLoadError
        mock_predict.side_effect = ModelLoadError("No model loaded")
        response = self.client.post(self.url, {"image": self._valid_image()})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Model unavailable")

    @patch("retina_app.views.predict_image")
    def test_post_not_a_fundus_shows_error(self, mock_predict):
        from retina_app.services.exceptions import NotAFundusImageError
        mock_predict.side_effect = NotAFundusImageError("Not a retinal image")
        response = self.client.post(self.url, {"image": self._valid_image()})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Not a retinal image")


class IndexViewFundusValidationTests(TestCase):
    """Test fundus validation rejection in the view."""

    def setUp(self):
        self.url = reverse("index")

    @patch("retina_app.views.predict_image")
    def test_not_a_fundus_error_rejected(self, mock_predict):
        from retina_app.services.exceptions import NotAFundusImageError
        mock_predict.side_effect = NotAFundusImageError("Image is not a retinal fundus photograph")
        img = SimpleUploadedFile("test.jpg", create_test_image_bytes(), content_type="image/jpeg")
        response = self.client.post(self.url, {"image": img})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "not a retinal fundus")


class ProtectedMediaTests(TestCase):
    """Test /media/ file serving."""

    def setUp(self):
        self.media_dir = tempfile.mkdtemp()
        self.test_file_path = os.path.join(self.media_dir, "test_image.jpg")
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(self.test_file_path, format="JPEG")

    @override_settings(MEDIA_ROOT=lambda: None)
    def test_media_404_for_missing_file(self):
        with override_settings(MEDIA_ROOT=self.media_dir):
            url = reverse("protected_media", kwargs={"path": "nonexistent.jpg"})
            response = self.client.get(url)
            self.assertEqual(response.status_code, 404)

    @override_settings(MEDIA_ROOT=lambda: None)
    def test_media_serves_existing_file(self):
        with override_settings(MEDIA_ROOT=self.media_dir):
            url = reverse("protected_media", kwargs={"path": "test_image.jpg"})
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)

    @override_settings(MEDIA_ROOT=lambda: None)
    def test_media_404_for_path_traversal(self):
        with override_settings(MEDIA_ROOT=self.media_dir):
            url = reverse("protected_media", kwargs={"path": "../../../etc/passwd"})
            response = self.client.get(url)
            self.assertEqual(response.status_code, 404)


class IndexViewImageStorageTests(TestCase):
    """Test that uploaded images are saved correctly."""

    def setUp(self):
        self.url = reverse("index")

    def test_image_saved_to_database(self):
        img = SimpleUploadedFile("test.jpg", create_test_image_bytes(), content_type="image/jpeg")
        self.client.post(self.url, {"image": img})
        self.assertGreaterEqual(UploadedImage.objects.count(), 1)

    @patch("retina_app.views.predict_image")
    def test_upload_record_has_user_none(self, mock_predict):
        mock_predict.return_value = {
            "label": "Healthy",
            "confidence": 0.9,
            "model_version": "ensemble-v3",
            "probabilities": [0.9, 0.05, 0.03, 0.02],
            "uncertainty": 0.1,
            "gradcam_url": None,
            "is_refused": False,
            "refusal_message": "",
            "preprocessing_viz_url": None,
        }
        img = SimpleUploadedFile("test.jpg", create_test_image_bytes(), content_type="image/jpeg")
        self.client.post(self.url, {"image": img})
        record = UploadedImage.objects.first()
        self.assertIsNone(record.user)
