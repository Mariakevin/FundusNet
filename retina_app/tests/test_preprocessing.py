import os
import tempfile

import numpy as np
from django.test import TestCase
from PIL import Image

from retina_app.constants import (
    MAX_FILE_SIZE,
)
from retina_app.services.exceptions import ImageValidationError
from retina_app.services.preprocessing import (
    apply_clahe,
    assess_image_quality,
    check_image_quality,
    detect_fundus_roi,
    enhance_fundus_image,
    extract_green_channel,
    preprocess_fundus,
    validate_image_file,
)


def _create_test_image(width=224, height=224, fmt="JPEG"):
    """Create a temporary test image and return its path."""
    img = Image.new("RGB", (width, height), color="red")
    tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt.lower()}", delete=False)
    img.save(tmp, format=fmt)
    tmp.close()
    return tmp.name


def _create_cv2_test_image(width=224, height=224):
    """Create a numpy RGB image suitable for preprocessing functions."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


class ValidateImageFileTest(TestCase):
    def test_nonexistent_file_raises(self):
        with self.assertRaises(ImageValidationError):
            validate_image_file("/tmp/nonexistent_abc123.jpg")

    def test_empty_file_raises(self):
        path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        path.close()
        try:
            with self.assertRaises(ImageValidationError) as ctx:
                validate_image_file(path.name)
            self.assertIn("empty", str(ctx.exception))
        finally:
            os.unlink(path.name)

    def test_unsupported_extension_raises(self):
        path = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        path.write(b"fake data")
        path.close()
        try:
            with self.assertRaises(ImageValidationError) as ctx:
                validate_image_file(path.name)
            self.assertIn("Unsupported file type", str(ctx.exception))
        finally:
            os.unlink(path.name)

    def test_oversized_file_raises(self):
        path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        path.write(b"x" * (MAX_FILE_SIZE + 1))
        path.close()
        try:
            with self.assertRaises(ImageValidationError) as ctx:
                validate_image_file(path.name)
            self.assertIn("too large", str(ctx.exception).lower())
        finally:
            os.unlink(path.name)

    def test_too_small_image_raises(self):
        img = Image.new("RGB", (32, 32), color="blue")
        path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(path, format="JPEG")
        path.close()
        try:
            with self.assertRaises(ImageValidationError) as ctx:
                validate_image_file(path.name)
            self.assertIn("too small", str(ctx.exception).lower())
        finally:
            os.unlink(path.name)

    def test_valid_image_passes(self):
        path = _create_test_image(224, 224)
        try:
            validate_image_file(path)
        finally:
            os.unlink(path)

    def test_valid_png_passes(self):
        path = _create_test_image(224, 224, fmt="PNG")
        try:
            validate_image_file(path)
        finally:
            os.unlink(path)


class ApplyClaheTest(TestCase):
    def test_grayscale_clahe(self):
        img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        result = apply_clahe(img)
        self.assertEqual(result.shape, img.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_color_clahe(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = apply_clahe(img)
        self.assertEqual(result.shape, img.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_clahe_custom_params(self):
        img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        result = apply_clahe(img, clip_limit=4.0, tile_grid_size=(16, 16))
        self.assertEqual(result.shape, img.shape)


class ExtractGreenChannelTest(TestCase):
    def test_extracts_channel_1(self):
        img = _create_cv2_test_image()
        green = extract_green_channel(img)
        self.assertEqual(green.shape, (224, 224))
        np.testing.assert_array_equal(green, img[:, :, 1])


class EnhanceFundusTest(TestCase):
    def test_enhance_returns_uint8(self):
        img = _create_cv2_test_image()
        result = enhance_fundus_image(img)
        self.assertEqual(result.dtype, np.uint8)

    def test_enhance_preserves_shape(self):
        img = _create_cv2_test_image()
        result = enhance_fundus_image(img)
        self.assertEqual(result.shape, img.shape)


class DetectFundusRoiTest(TestCase):
    def test_returns_tuple(self):
        img = _create_cv2_test_image()
        result, center, radius = detect_fundus_roi(img)
        self.assertEqual(result.shape[2], 3)
        self.assertEqual(len(center), 2)
        self.assertGreater(radius, 0)


class AssessImageQualityTest(TestCase):
    def test_returns_expected_keys(self):
        img = _create_cv2_test_image()
        quality = assess_image_quality(img)
        self.assertIn("overall_quality", quality)
        self.assertIn("quality_level", quality)
        self.assertIn("blur_score", quality)
        self.assertIn("brightness_score", quality)
        self.assertIn("contrast_score", quality)
        self.assertIn("edge_score", quality)

    def test_quality_level_values(self):
        img = _create_cv2_test_image()
        quality = assess_image_quality(img)
        self.assertIn(quality["quality_level"], ("good", "fair", "poor"))

    def test_quality_score_range(self):
        img = _create_cv2_test_image()
        quality = assess_image_quality(img)
        self.assertGreaterEqual(quality["overall_quality"], 0.0)


class CheckImageQualityTest(TestCase):
    def test_invalid_image_returns_not_passed(self):
        result = check_image_quality("/tmp/nonexistent_abc123.jpg")
        self.assertFalse(result["passed"])
        self.assertIn("error", result)

    def test_valid_image_returns_dict(self):
        path = _create_test_image(224, 224)
        try:
            result = check_image_quality(path)
            self.assertIn("passed", result)
            self.assertIn("quality_level", result)
        finally:
            os.unlink(path)


class PreprocessFundusTest(TestCase):
    def test_invalid_path_raises(self):
        with self.assertRaises(ImageValidationError):
            preprocess_fundus("/tmp/nonexistent_abc123.jpg")

    def test_enhance_only(self):
        path = _create_test_image(224, 224)
        try:
            result = preprocess_fundus(path, enhance=True, detect_roi=False)
            self.assertEqual(result.ndim, 3)
            self.assertEqual(result.shape[2], 3)
        finally:
            os.unlink(path)

    def test_no_enhancement(self):
        path = _create_test_image(224, 224)
        try:
            result = preprocess_fundus(path, enhance=False, detect_roi=False)
            self.assertEqual(result.ndim, 3)
        finally:
            os.unlink(path)
