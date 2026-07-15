"""Tests for API endpoints."""

import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from io import BytesIO

from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image

from retina_app.constants import CATEGORIES, MODEL_LIST


def create_test_image(format="JPEG", size=(224, 224)):
    """Create a test image for upload testing."""
    img = Image.new("RGB", size, color="red")
    buf = BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf


class APIRootTest(TestCase):
    """Test API root endpoint."""

    def setUp(self):
        self.client = Client()

    def test_api_root_returns_200(self):
        response = self.client.get("/api/v1/")
        self.assertEqual(response.status_code, 200)

    def test_api_root_returns_json(self):
        response = self.client.get("/api/v1/")
        self.assertEqual(response["Content-Type"], "application/json")

    def test_api_root_has_endpoints(self):
        response = self.client.get("/api/v1/")
        data = json.loads(response.content)
        self.assertIn("endpoints", data)


class PredictSingleTest(TestCase):
    """Test single image prediction endpoint."""

    def setUp(self):
        self.client = Client()

    def test_predict_no_image_returns_400(self):
        response = self.client.post("/api/v1/predict/")
        self.assertEqual(response.status_code, 400)

    def test_predict_empty_file_returns_400(self):
        response = self.client.post(
            "/api/v1/predict/",
            {"image": SimpleUploadedFile("test.jpg", b"", content_type="image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_invalid_type_returns_400(self):
        response = self.client.post(
            "/api/v1/predict/",
            {"image": SimpleUploadedFile("test.txt", b"not an image", content_type="text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_get_not_allowed(self):
        response = self.client.get("/api/v1/predict/")
        self.assertIn(response.status_code, [405, 400])


class ModelHealthTest(TestCase):
    """Test model health endpoint."""

    def setUp(self):
        self.client = Client()

    def test_health_returns_200(self):
        response = self.client.get("/api/v1/health/")
        self.assertEqual(response.status_code, 200)

    def test_health_returns_json(self):
        response = self.client.get("/api/v1/health/")
        self.assertEqual(response["Content-Type"], "application/json")

    def test_health_has_status(self):
        response = self.client.get("/api/v1/health/")
        data = json.loads(response.content)
        self.assertIn("status", data)


class ServiceStatsTest(TestCase):
    """Test service stats endpoint."""

    def setUp(self):
        self.client = Client()

    def test_stats_returns_200(self):
        response = self.client.get("/api/v1/stats/")
        self.assertEqual(response.status_code, 200)


class ConstantsTest(TestCase):
    """Test constants are properly defined."""

    def test_categories_count(self):
        self.assertEqual(len(CATEGORIES), 4)

    def test_categories_values(self):
        self.assertIn("Healthy", CATEGORIES)
        self.assertIn("Cataract", CATEGORIES)
        self.assertIn("Glaucoma", CATEGORIES)
        self.assertIn("Retina Disease", CATEGORIES)

    def test_model_list_has_5_models(self):
        self.assertEqual(len(MODEL_LIST), 5)

    def test_model_list_has_expected_models(self):
        for model in ["swin", "maxvit", "convnext_v2", "efficientnet_v2", "deit"]:
            self.assertIn(model, MODEL_LIST)
