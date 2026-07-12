"""Fundus image preprocessing pipeline.
Minimal, focused preprocessing: ROI detection, CLAHE, quality assessment.
"""

import logging
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from retina_app.constants import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    MAX_IMAGE_DIMENSION,
    MIN_IMAGE_DIMENSION,
)
from retina_app.services.exceptions import ImageValidationError

logger = logging.getLogger("retina_app")


def validate_image_file(image_path: str) -> None:
    """Validate image file type, size, and dimensions."""
    if not os.path.exists(image_path):
        raise ImageValidationError(f"File not found: {image_path}")

    file_size = os.path.getsize(image_path)
    if file_size > MAX_FILE_SIZE:
        raise ImageValidationError(
            f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB. Got {file_size / (1024 * 1024):.1f}MB"
        )

    if file_size == 0:
        raise ImageValidationError("File is empty")

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(f"Unsupported file type: {ext}. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")

    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                raise ImageValidationError(
                    f"Image too large. Maximum dimension is {MAX_IMAGE_DIMENSION}px. Got {width}x{height}px"
                )
            if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                raise ImageValidationError(
                    f"Image too small. Minimum dimension is {MIN_IMAGE_DIMENSION}px. Got {width}x{height}px"
                )
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError(f"Unable to read image dimensions: {exc}")


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE for enhanced fundus contrast."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_ch, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """Extract green channel for better blood vessel contrast."""
    return image[:, :, 1]


def enhance_fundus_image(image: np.ndarray) -> np.ndarray:
    """Apply fundus-specific preprocessing: green channel enhancement + CLAHE."""
    green_channel = extract_green_channel(image)
    green_channel = apply_clahe(green_channel)
    green_normalized = cv2.normalize(green_channel, None, 0, 255, cv2.NORM_MINMAX)

    result = image.copy()
    result[:, :, 1] = green_normalized.astype(np.uint8)
    return result


def detect_fundus_roi(image: np.ndarray) -> tuple:
    """Detect the fundus region of interest (circular ROI)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=50, maxRadius=0
    )

    if circles is not None and len(circles) > 0:
        circles = np.uint16(np.around(circles[0]))
        circle = sorted(circles, key=lambda c: c[2], reverse=True)[0]
        cx, cy, r = circle

        h, w = image.shape[:2]
        r = int(r * 0.95)
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(w, cx + r)
        y2 = min(h, cy + r)

        cropped = image[y1:y2, x1:x2]
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.circle(mask, (cx - x1, cy - y1), r, 255, -1)
        masked = cv2.bitwise_and(cropped, cropped, mask=cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
        return masked, (cx, cy), r

    return image, (image.shape[1] // 2, image.shape[0] // 2), min(image.shape[:2]) // 2


def assess_image_quality(image: np.ndarray) -> dict[str, Any]:
    """Assess fundus image quality and return quality metrics."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = laplacian_var / 1000.0

    brightness = np.mean(gray)
    brightness_score = 1.0 - abs(brightness - 128) / 128

    contrast = np.std(gray)
    contrast_score = min(contrast / 50.0, 1.0)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = np.mean(hsv[:, :, 1])
    saturation_score = min(saturation / 100.0, 1.0)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    edge_score = min(edge_density * 100, 1.0)

    overall_quality = (
        blur_score * 0.3 + brightness_score * 0.15 + contrast_score * 0.25 + saturation_score * 0.1 + edge_score * 0.2
    )

    quality_level = "good"
    if overall_quality < 0.3:
        quality_level = "poor"
    elif overall_quality < 0.5:
        quality_level = "fair"

    return {
        "overall_quality": overall_quality,
        "quality_level": quality_level,
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "edge_score": edge_score,
    }


def check_image_quality(image_path: str, quality_threshold: float = 0.3) -> dict[str, Any]:
    """Check if image meets quality standards."""
    image = cv2.imread(image_path)
    if image is None:
        return {"passed": False, "error": "Could not read image"}

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    quality = assess_image_quality(image)
    passed = quality["overall_quality"] >= quality_threshold

    return {
        "passed": passed,
        "quality_level": quality["quality_level"],
        "overall_quality": quality["overall_quality"],
    }


def preprocess_fundus(image_path: str, enhance: bool = True, detect_roi: bool = False) -> np.ndarray:
    """Complete fundus preprocessing pipeline."""
    try:
        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        image = np.array(pil_img)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    except Exception:
        image = cv2.imread(image_path)
        if image is None:
            raise ImageValidationError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if detect_roi:
        image, _, _ = detect_fundus_roi(image)

    if enhance:
        image = enhance_fundus_image(image)

    return image
