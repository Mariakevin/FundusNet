"""Fundus image preprocessing pipeline.
CLAHE enhancement, ROI detection, quality assessment,
adaptive CLAHE, noise reduction, color constancy.
"""

import logging
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from retina_app.constants import (
    ADAPTIVE_CLAHE_BRIGHT_CLIP,
    ADAPTIVE_CLAHE_BRIGHT_THRESHOLD,
    ADAPTIVE_CLAHE_BRIGHT_TILE,
    ADAPTIVE_CLAHE_DARK_CLIP,
    ADAPTIVE_CLAHE_DARK_THRESHOLD,
    ADAPTIVE_CLAHE_DARK_TILE,
    ADAPTIVE_CLAHE_NORMAL_CLIP,
    ADAPTIVE_CLAHE_NORMAL_TILE,
    ALLOWED_EXTENSIONS,
    COLOR_CONSTANCY_METHOD,
    COLOR_CONSTANCY_WHITE_PATCH_PERCENTILE,
    MAX_FILE_SIZE,
    MAX_IMAGE_DIMENSION,
    MIN_IMAGE_DIMENSION,
    NOISE_REDUCTION_SPECULAR_ENABLED,
    NOISE_REDUCTION_SPECULAR_KERNEL,
    NOISE_REDUCTION_STRENGTH,
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
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced fundus contrast."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_ch, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)


def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """Extract green channel for better blood vessel contrast in fundus images."""
    return image[:, :, 1]


def enhance_fundus_image(image: np.ndarray, apply_clahe_flag: bool = True) -> np.ndarray:
    """Apply fundus-specific preprocessing pipeline."""
    green_channel = extract_green_channel(image)

    if apply_clahe_flag:
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
        image = enhance_fundus_image(image, apply_clahe_flag=True)

    return image


def apply_adaptive_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE with parameters adapted to image brightness.

    Dark images get higher clip limit for more contrast enhancement.
    Bright images get lower clip limit and larger tile grid for subtler effect.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)

    if brightness < ADAPTIVE_CLAHE_DARK_THRESHOLD:
        clip_limit = ADAPTIVE_CLAHE_DARK_CLIP
        tile_grid = ADAPTIVE_CLAHE_DARK_TILE
    elif brightness > ADAPTIVE_CLAHE_BRIGHT_THRESHOLD:
        clip_limit = ADAPTIVE_CLAHE_BRIGHT_CLIP
        tile_grid = ADAPTIVE_CLAHE_BRIGHT_TILE
    else:
        clip_limit = ADAPTIVE_CLAHE_NORMAL_CLIP
        tile_grid = ADAPTIVE_CLAHE_NORMAL_TILE

    return apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid)


def reduce_noise(image: np.ndarray) -> np.ndarray:
    """Reduce noise in fundus images using non-local means denoising.

    Also optionally removes specular reflections using morphological operations.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(
        image_bgr,
        None,
        NOISE_REDUCTION_STRENGTH,
        NOISE_REDUCTION_STRENGTH,
        7,
        21,
    )
    result = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    if NOISE_REDUCTION_SPECULAR_ENABLED:
        result = _remove_specular_reflections(result)

    return result


def _remove_specular_reflections(image: np.ndarray) -> np.ndarray:
    """Remove specular reflections (bright spots) from fundus images using morphological operations."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (NOISE_REDUCTION_SPECULAR_KERNEL, NOISE_REDUCTION_SPECULAR_KERNEL),
    )
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    mean_brightness = np.mean(gray)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(
        image_bgr,
        dilated,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA,
    )
    result = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

    mask_3ch = dilated[:, :, np.newaxis].astype(np.float32) / 255.0
    fallback = np.full_like(image, int(mean_brightness), dtype=np.uint8)
    result = (result * mask_3ch + fallback * (1 - mask_3ch)).astype(np.uint8)

    return result


def apply_color_constancy(image: np.ndarray) -> np.ndarray:
    """Apply color constancy correction to normalize illumination.

    Methods:
    - gray_world: Normalizes so mean of each channel equals the mean of all channels.
    - white_patch: Normalizes using the brightest pixel in each channel.

    """
    if COLOR_CONSTANCY_METHOD == "white_patch":
        return _white_patch_correction(image)
    return _gray_world_correction(image)


def _gray_world_correction(image: np.ndarray) -> np.ndarray:
    """Gray-world color constancy: normalize each channel to have the same mean."""
    img = image.astype(np.float64)
    means = img.mean(axis=(0, 1))
    global_mean = means.mean()

    for c in range(3):
        if means[c] > 0:
            img[:, :, c] *= global_mean / means[c]

    return np.clip(img, 0, 255).astype(np.uint8)


def _white_patch_correction(image: np.ndarray) -> np.ndarray:
    """White-patch color constancy: normalize using brightest pixel in each channel."""
    img = image.astype(np.float64)
    for c in range(3):
        max_val = np.percentile(img[:, :, c], COLOR_CONSTANCY_WHITE_PATCH_PERCENTILE)
        if max_val > 0:
            img[:, :, c] *= 255.0 / max_val

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_preprocessing_viz(image_path: str) -> dict[str, np.ndarray]:
    """Generate 4-panel preprocessing visualization: original, CLAHE, denoised, color corrected.

    Returns dict with panel names as keys and numpy arrays as values.
    Returns empty dict if image cannot be read.
    """
    try:
        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        original = np.array(pil_img)
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        elif original.shape[2] == 4:
            original = cv2.cvtColor(original, cv2.COLOR_RGBA2RGB)
    except Exception:
        try:
            original = cv2.imread(image_path)
            if original is None:
                logger.warning("Could not read image: %s", image_path)
                return {}
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        except Exception:
            logger.warning("Could not read image: %s", image_path)
            return {}

    clahe = apply_adaptive_clahe(original)
    denoised = reduce_noise(original)
    color_corrected = apply_color_constancy(original)

    return {
        "original": original,
        "clahe": clahe,
        "denoised": denoised,
        "color_corrected": color_corrected,
    }


def save_preprocessing_viz(panels: dict[str, np.ndarray], output_path: str) -> str:
    """Save preprocessing visualization as a 2x2 grid image."""
    h, w = panels["original"].shape[:2]
    panel_h, panel_w = h // 2, w // 2

    canvas = np.zeros((panel_h * 2, panel_w * 2, 3), dtype=np.uint8)

    for idx, (name, panel) in enumerate(panels.items()):
        row, col = divmod(idx, 2)
        resized = cv2.resize(panel, (panel_w, panel_h))
        canvas[row * panel_h : (row + 1) * panel_h, col * panel_w : (col + 1) * panel_w] = resized

    labels = ["Original", "Adaptive CLAHE", "Denoised", "Color Corrected"]
    for idx, label in enumerate(labels):
        row, col = divmod(idx, 2)
        x = col * panel_w + 10
        y = row * panel_h + 25
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return output_path
