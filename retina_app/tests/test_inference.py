from django.test import TestCase
from PIL import Image

from retina_app.services.exceptions import (
    ImageCorruptError,
    ImageSizeError,
    ImageValidationError,
    InferenceError,
)


class InferenceExceptionsTest(TestCase):
    def test_inference_error_has_code(self):
        err = InferenceError("Test error")
        self.assertEqual(err.message, "Test error")
        self.assertEqual(err.code, "INFERENCE_ERROR")

    def test_image_validation_error_has_code(self):
        err = ImageValidationError("Invalid image")
        self.assertEqual(err.code, "IMAGE_VALIDATION_ERROR")

    def test_image_corrupt_error_has_code(self):
        err = ImageCorruptError()
        self.assertEqual(err.code, "IMAGE_CORRUPT_ERROR")

    def test_image_size_error_has_code(self):
        err = ImageSizeError("File too large")
        self.assertEqual(err.code, "IMAGE_SIZE_ERROR")


class InferenceCategoriesTest(TestCase):
    def test_categories_defined(self):
        from retina_app.constants import CATEGORIES

        self.assertEqual(CATEGORIES, ["Healthy", "Cataract", "Glaucoma", "Retina Disease"])

    def test_categories_count(self):
        from retina_app.constants import CATEGORIES

        self.assertEqual(len(CATEGORIES), 4)


class InferenceModelConfigTest(TestCase):
    def test_model_list_defined(self):
        from retina_app.constants import MODEL_LIST

        self.assertIn("squeezenet", MODEL_LIST)
        self.assertIn("efficientnet", MODEL_LIST)

    def test_model_weights_defined(self):
        from retina_app.constants import MODEL_WEIGHTS

        total_weight = sum(MODEL_WEIGHTS.values())
        self.assertAlmostEqual(total_weight, 1.0, places=1)

    def test_ensemble_min_models(self):
        from retina_app.constants import ENSEMBLE_MIN_MODELS

        self.assertEqual(ENSEMBLE_MIN_MODELS, 2)


class InferenceValidationTest(TestCase):
    def test_valid_image_extensions(self):
        from retina_app.constants import ALLOWED_EXTENSIONS

        self.assertIn(".jpg", ALLOWED_EXTENSIONS)
        self.assertIn(".png", ALLOWED_EXTENSIONS)
        self.assertIn(".jpeg", ALLOWED_EXTENSIONS)

    def test_max_image_size_configured(self):
        from retina_app.constants import MAX_FILE_SIZE

        self.assertEqual(MAX_FILE_SIZE, 10 * 1024 * 1024)

    def test_min_dimension_configured(self):
        from retina_app.constants import MIN_IMAGE_DIMENSION

        self.assertEqual(MIN_IMAGE_DIMENSION, 64)


class InferenceTransformTest(TestCase):
    def test_transform_defined(self):
        from retina_app.services.transforms import TRANSFORM

        self.assertIsNotNone(TRANSFORM)

    def test_transform_has_resize(self):
        from retina_app.services.transforms import TRANSFORM

        transform_names = [t.__class__.__name__ for t in TRANSFORM.transforms]
        self.assertIn("Resize", transform_names)
        # ToTensor may be nested inside a sub-Compose; verify transform works
        import numpy as np

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        tensor = TRANSFORM(img)
        self.assertEqual(tensor.shape[0], 3)  # 3 channels


class InferenceConfidenceThresholdsTest(TestCase):
    def test_low_threshold_defined(self):
        from retina_app.constants import CONFIDENCE_THRESHOLD_LOW

        self.assertEqual(CONFIDENCE_THRESHOLD_LOW, 0.5)

    def test_high_threshold_defined(self):
        from retina_app.constants import CONFIDENCE_THRESHOLD_HIGH

        self.assertEqual(CONFIDENCE_THRESHOLD_HIGH, 0.7)
