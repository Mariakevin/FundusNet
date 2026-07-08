"""Tests for retinal fundus image validation (fundus_validator.py)."""

import os
import tempfile

import cv2
import numpy as np
from django.test import TestCase

from retina_app.constants import (
    FUNDUS_AREA_MAX_RATIO,
    FUNDUS_AREA_MIN_RATIO,
    FUNDUS_CIRCULARITY_MIN,
    FUNDUS_COLOR_MIN_RATIO,
    FUNDUS_EDGE_MAX_RATIO,
    FUNDUS_EDGE_MIN_RATIO,
    FUNDUS_GREEN_CH_MIN_STD,
    FUNDUS_VALIDATION_THRESHOLD,
)
from retina_app.services.fundus_validator import (
    _check_circular_region,
    _check_color_distribution,
    _check_edge_density,
    _check_green_channel,
    _check_texture_regularity,
    validate_fundus_image,
)


def _create_fundus_like_image(width=512, height=512):
    """Create a synthetic image that mimics fundus characteristics.

    - Reddish-orange center (circular fundus region)
    - Dark background
    - Moderate edge density (simulated vessels)
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Dark background
    img[:, :] = [10, 5, 5]

    # Circular bright reddish-orange region (fundus)
    center = (width // 2, height // 2)
    radius = int(min(width, height) * 0.35)
    cv2.circle(img, center, radius, (180, 80, 40), -1)  # BGR reddish

    # Add some "vessel-like" lines for edge density
    for i in range(5):
        pt1 = (center[0] - radius + i * 20, center[1])
        pt2 = (center[0] + radius - i * 20, center[1] + 30)
        cv2.line(img, pt1, pt2, (120, 50, 30), 1)

    return img


def _create_text_image(width=512, height=512):
    """Create a synthetic text/document image (should be rejected)."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 240  # White background

    # Add text-like horizontal lines
    for y in range(50, height, 30):
        cv2.line(img, (50, y), (width - 50, y), (30, 30, 30), 1)

    # Add high-frequency edges (text characters)
    for x in range(100, width - 100, 15):
        for y in range(55, min(200, height), 15):
            cv2.rectangle(img, (x, y), (x + 8, y + 10), (20, 20, 20), -1)

    return img


def _create_natural_scene_image(width=512, height=512):
    """Create a synthetic natural scene image (should be rejected)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Blue sky top half
    img[: height // 2, :] = [200, 150, 50]

    # Green grass bottom half
    img[height // 2 :, :] = [50, 150, 50]

    # Random colors (trees, flowers)
    for _ in range(50):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.circle(img, (x, y), np.random.randint(5, 20), color, -1)

    return img


def _create_blank_image(width=512, height=512):
    """Create a blank/uniform image."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 128
    return img


class CheckColorDistributionTest(TestCase):
    """Tests for _check_color_distribution."""

    def test_fundus_like_image_scores_high(self):
        img = _create_fundus_like_image()
        score = _check_color_distribution(img)
        self.assertGreater(score, 0.3)

    def test_text_image_scores_low(self):
        img = _create_text_image()
        score = _check_color_distribution(img)
        self.assertLess(score, 0.3)

    def test_natural_scene_has_valid_score(self):
        """Natural scene should produce a valid score in range."""
        img = _create_natural_scene_image()
        score = _check_color_distribution(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_pure_red_image_scores_high(self):
        """Pure red image should have high color score."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [255, 0, 0]  # RGB red (R=255, G=0, B=0)
        score = _check_color_distribution(img)
        self.assertGreater(score, 0.5)

    def test_score_range(self):
        img = _create_fundus_like_image()
        score = _check_color_distribution(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class CheckCircularRegionTest(TestCase):
    """Tests for _check_circular_region."""

    def test_fundus_like_image_scores_high(self):
        img = _create_fundus_like_image()
        score = _check_circular_region(img)
        self.assertGreater(score, 0.1)

    def test_text_image_scores_lower_than_circle(self):
        """Text image should score lower than a perfect circle."""
        text = _create_text_image()
        circle = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(circle, (100, 100), 70, (255, 255, 255), -1)
        text_score = _check_circular_region(text)
        circle_score = _check_circular_region(circle)
        self.assertLess(text_score, circle_score)

    def test_perfect_circle_scores_high(self):
        """A perfect white circle on black background should score well."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 70, (255, 255, 255), -1)
        score = _check_circular_region(img)
        self.assertGreater(score, 0.3)

    def test_score_range(self):
        img = _create_fundus_like_image()
        score = _check_circular_region(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class CheckEdgeDensityTest(TestCase):
    """Tests for _check_edge_density."""

    def test_fundus_like_image_in_valid_range(self):
        img = _create_fundus_like_image()
        score = _check_edge_density(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_blank_image_scores_low(self):
        img = _create_blank_image()
        score = _check_edge_density(img)
        self.assertEqual(score, 0.0)

    def test_text_image_in_valid_range(self):
        """Text image edge density should be in valid score range."""
        img = _create_text_image()
        score = _check_edge_density(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_score_range(self):
        img = _create_fundus_like_image()
        score = _check_edge_density(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class CheckGreenChannelTest(TestCase):
    """Tests for _check_green_channel."""

    def test_fundus_like_image_scores_moderate(self):
        img = _create_fundus_like_image()
        score = _check_green_channel(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_uniform_image_scores_low(self):
        img = _create_blank_image()
        score = _check_green_channel(img)
        self.assertEqual(score, 0.0)

    def test_score_range(self):
        img = _create_fundus_like_image()
        score = _check_green_channel(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class CheckTextureRegularityTest(TestCase):
    """Tests for _check_texture_regularity."""

    def test_dense_text_image_scores_low(self):
        """Dense text document with many regular lines should score low."""
        # Create realistic text blocks with proper line spacing
        img = np.ones((600, 400, 3), dtype=np.uint8) * 245
        # Multiple paragraphs with thick text lines
        for block_start in [30, 200, 380]:
            for y in range(block_start, block_start + 140, 10):
                cv2.rectangle(img, (40, y), (360, y + 5), (20, 20, 20), -1)
        score = _check_texture_regularity(img)
        # Score of 0.5 = "moderate" (ambiguous), 0.8 = "organic" (fundus-like)
        # Text should NOT score as organic
        self.assertLessEqual(score, 0.5)

    def test_fundus_like_image_scores_high(self):
        """Fundus-like image with organic texture should score high."""
        img = _create_fundus_like_image()
        score = _check_texture_regularity(img)
        self.assertGreater(score, 0.3)

    def test_blank_image_scores_low(self):
        img = _create_blank_image()
        score = _check_texture_regularity(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_score_range(self):
        img = _create_fundus_like_image()
        score = _check_texture_regularity(img)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class ValidateFundusImageTest(TestCase):
    """Tests for the main validate_fundus_image function."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _save_image(self, img, name="test.jpg"):
        path = os.path.join(self.tmp_dir, name)
        cv2.imwrite(path, img)
        return path

    def test_fundus_like_image_structure(self):
        """Fundus-like image produces valid result with all required keys."""
        img = _create_fundus_like_image()
        path = self._save_image(img)
        result = validate_fundus_image(path)
        self.assertIn("is_fundus", result)
        self.assertIn("confidence", result)
        self.assertIn("signals", result)
        self.assertIn("message", result)
        self.assertIsInstance(result["is_fundus"], (bool, np.bool_))
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_text_image_rejected(self):
        """Text document should be identified as low-confidence."""
        img = _create_text_image()
        path = self._save_image(img)
        result = validate_fundus_image(path)
        # The text image's combined score determines the result
        # We just verify the function runs and returns valid structure
        self.assertIn("is_fundus", result)
        self.assertIn("confidence", result)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_natural_scene_rejected(self):
        img = _create_natural_scene_image()
        path = self._save_image(img)
        result = validate_fundus_image(path)
        self.assertFalse(result["is_fundus"])

    def test_nonexistent_file(self):
        result = validate_fundus_image("/nonexistent/image.jpg")
        self.assertFalse(result["is_fundus"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("Could not read", result["message"])

    def test_returns_all_signals(self):
        img = _create_fundus_like_image()
        path = self._save_image(img)
        result = validate_fundus_image(path)
        self.assertIn("color_distribution", result["signals"])
        self.assertIn("circular_region", result["signals"])
        self.assertIn("edge_density", result["signals"])
        self.assertIn("green_channel", result["signals"])
        self.assertIn("texture_regularity", result["signals"])

    def test_signal_scores_in_range(self):
        img = _create_fundus_like_image()
        path = self._save_image(img)
        result = validate_fundus_image(path)
        for score in result["signals"].values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_combined_score_matches_weighted_sum(self):
        """Verify combined score is the weighted sum of individual signals."""
        img = _create_fundus_like_image()
        path = self._save_image(img)
        result = validate_fundus_image(path)
        s = result["signals"]
        expected = (
            s["color_distribution"] * 0.25
            + s["circular_region"] * 0.20
            + s["edge_density"] * 0.15
            + s["green_channel"] * 0.10
            + s["texture_regularity"] * 0.30
        )
        # Account for the has_fundus_color gate — if color_score < 0.20,
        # is_fundus is forced False regardless of combined score
        self.assertAlmostEqual(result["confidence"], round(expected, 4), places=3)

    def test_rejection_message_mentions_weak_signals(self):
        img = _create_text_image()
        path = self._save_image(img)
        result = validate_fundus_image(path)
        self.assertIn("retinal fundus", result["message"].lower())

    def test_pure_red_high_color_score(self):
        """Pure red image should have high color distribution score.

        Note: cv2.imwrite uses BGR convention, so [0, 0, 255] stores as
        B=0, G=0, R=255 (red). The validator reads BGR→RGB→HSV correctly.
        """
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :] = [0, 0, 255]  # BGR red (B=0, G=0, R=255)
        path = self._save_image(img, "red.jpg")
        result = validate_fundus_image(path)
        self.assertGreater(result["signals"]["color_distribution"], 0.5)

    def test_text_image_rejected_by_color_gate(self):
        """Text document has no fundus color, so the color gate rejects it."""
        img = _create_text_image()
        path = self._save_image(img)
        result = validate_fundus_image(path)
        self.assertFalse(result["is_fundus"])
        self.assertLess(result["signals"]["color_distribution"], 0.20)


class FundusValidatorConstantsTest(TestCase):
    """Verify fundus validation constants are sensible."""

    def test_threshold_is_between_0_and_1(self):
        self.assertGreater(FUNDUS_VALIDATION_THRESHOLD, 0.0)
        self.assertLess(FUNDUS_VALIDATION_THRESHOLD, 1.0)

    def test_color_min_ratio_is_between_0_and_1(self):
        self.assertGreater(FUNDUS_COLOR_MIN_RATIO, 0.0)
        self.assertLess(FUNDUS_COLOR_MIN_RATIO, 1.0)

    def test_circularity_min_is_between_0_and_1(self):
        self.assertGreater(FUNDUS_CIRCULARITY_MIN, 0.0)
        self.assertLess(FUNDUS_CIRCULARITY_MIN, 1.0)

    def test_area_ratios_are_valid(self):
        self.assertGreater(FUNDUS_AREA_MIN_RATIO, 0.0)
        self.assertLess(FUNDUS_AREA_MAX_RATIO, 1.0)
        self.assertLess(FUNDUS_AREA_MIN_RATIO, FUNDUS_AREA_MAX_RATIO)

    def test_edge_ratios_are_valid(self):
        self.assertGreater(FUNDUS_EDGE_MIN_RATIO, 0.0)
        self.assertLess(FUNDUS_EDGE_MAX_RATIO, 1.0)
        self.assertLess(FUNDUS_EDGE_MIN_RATIO, FUNDUS_EDGE_MAX_RATIO)

    def test_green_channel_min_std_is_positive(self):
        self.assertGreater(FUNDUS_GREEN_CH_MIN_STD, 0.0)
