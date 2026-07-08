"""Retinal fundus image validation using multi-signal heuristic analysis.

Rejects non-fundus images (text, screenshots, natural scenes) before inference.
Uses 4 independent computer vision signals — no extra trained model needed.

Signals:
1. Color distribution — fundus images are reddish-orange dominant
2. Circular region — fundus has a bright circular field of view on dark background
3. Edge density — fundus has moderate edge density from vessels and disc boundaries
4. Green channel — fundus green channel has specific variance for vessel contrast
"""

import logging
from typing import Any

import cv2
import numpy as np

from retina_app.constants import (
    FUNDUS_AREA_MAX_RATIO,
    FUNDUS_AREA_MIN_RATIO,
    FUNDUS_CIRCULARITY_MIN,
    FUNDUS_COLOR_MIN_RATIO,
    FUNDUS_EDGE_MAX_RATIO,
    FUNDUS_EDGE_MIN_RATIO,
    FUNDUS_GREEN_CH_MIN_STD,
    FUNDUS_LEARNED_MODEL_PATH,
    FUNDUS_LEARNED_VALIDATOR_ENABLED,
    FUNDUS_VALIDATION_THRESHOLD,
)

logger = logging.getLogger("retina_app")


def _check_color_distribution(image: np.ndarray, hsv: np.ndarray = None) -> float:
    """Check if image has fundus-like reddish-orange color distribution.

    Fundus images have hue concentrated in red-orange range (0-30 and 150-180
    in OpenCV's 0-180 HSV scale) with moderate-to-high saturation.

    Returns score 0.0-1.0 (1.0 = strong fundus color signature).
    """
    if hsv is None:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    total_pixels = h.shape[0] * h.shape[1]

    # Red-orange hue ranges in OpenCV (H: 0-180)
    # Red wraps around: 0-10 and 160-180
    # Orange: 10-25
    # Warm tones: 25-35
    red_mask = ((h < 10) | (h > 160)) & (s > 30) & (v > 30)
    orange_mask = (h >= 10) & (h < 35) & (s > 25) & (v > 30)

    fundus_pixels = np.sum(red_mask | orange_mask)
    ratio = fundus_pixels / total_pixels

    # Score: 1.0 at FUNDUS_COLOR_MIN_RATIO, scales up to 1.0 at 60%
    if ratio >= 0.60:
        return 1.0
    elif ratio >= FUNDUS_COLOR_MIN_RATIO:
        return 0.5 + 0.5 * (ratio - FUNDUS_COLOR_MIN_RATIO) / (0.60 - FUNDUS_COLOR_MIN_RATIO)
    elif ratio >= 0.10:
        return ratio / FUNDUS_COLOR_MIN_RATIO * 0.5
    else:
        return 0.0


def _check_circular_region(image: np.ndarray, gray: np.ndarray = None) -> float:
    """Check for a bright circular region on dark background (fundus field of view).

    Fundus images have a characteristic bright circular fundus area surrounded
    by dark/black background from the camera aperture.  Documents/photos of
    text have a large bright region but lack the dark surround.

    Returns score 0.0-1.0 (1.0 = strong circular fundus region).
    """
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    image_area = h * w

    # Otsu threshold to separate bright fundus from dark background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    # Get the largest contour (should be the fundus region)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter == 0:
        return 0.0

    # Circularity: 4*pi*area / perimeter^2 (1.0 for perfect circle)
    circularity = 4 * np.pi * area / (perimeter * perimeter)

    # Area ratio: fundus typically occupies 30-85% of image
    area_ratio = area / image_area

    # --- Dark surround check ---
    # Fundus images have dark pixels around the bright circular region.
    # Documents have bright pixels extending to edges (no dark surround).
    # Create a mask of the largest contour and check the border region.
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    # Check border ring (outer 8% of image) for darkness
    border_width = max(int(min(h, w) * 0.08), 5)
    border_mask = np.zeros((h, w), dtype=np.uint8)
    border_mask[:border_width, :] = 255  # top
    border_mask[-border_width:, :] = 255  # bottom
    border_mask[:, :border_width] = 255  # left
    border_mask[:, -border_width:] = 255  # right

    # Pixels that are in border AND outside the bright region
    dark_border = cv2.bitwise_and(border_mask, cv2.bitwise_not(mask))
    border_pixel_count = np.sum(border_mask > 0)
    dark_pixel_count = np.sum(dark_border > 0)

    if border_pixel_count == 0:
        dark_ratio = 0.0
    else:
        dark_ratio = dark_pixel_count / border_pixel_count

    # Fundus: dark_ratio should be high (dark surround exists)
    # Document: dark_ratio is low (bright paper extends to edges)
    surround_score = 0.0
    if dark_ratio > 0.4:
        surround_score = min(1.0, (dark_ratio - 0.4) / 0.4)
    elif dark_ratio > 0.2:
        surround_score = (dark_ratio - 0.2) / 0.4 * 0.5

    # Score based on circularity, area ratio, and dark surround
    circ_score = 0.0
    if circularity >= FUNDUS_CIRCULARITY_MIN:
        circ_score = min(1.0, circularity / 0.8)

    area_score = 0.0
    if FUNDUS_AREA_MIN_RATIO <= area_ratio <= FUNDUS_AREA_MAX_RATIO:
        dist_from_ideal = abs(area_ratio - 0.55)
        area_score = max(0.0, 1.0 - dist_from_ideal / 0.40)

    return 0.4 * circ_score + 0.3 * area_score + 0.3 * surround_score


def _check_edge_density(image: np.ndarray, gray: np.ndarray = None) -> float:
    """Check edge density for fundus-like texture patterns.

    Fundus images have moderate edge density from blood vessels and optic
    disc boundaries. Text images have very high density with regular
    patterns. Very uniform images have very low density.

    Returns score 0.0-1.0 (1.0 = ideal fundus edge density).
    """
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size

    # Fundus images typically have 2-15% edge pixels
    if FUNDUS_EDGE_MIN_RATIO <= edge_ratio <= FUNDUS_EDGE_MAX_RATIO:
        # Peak at 5-8% (typical fundus with vessels + disc)
        ideal = 0.06
        dist = abs(edge_ratio - ideal)
        return max(0.0, 1.0 - dist / 0.12)
    elif edge_ratio < FUNDUS_EDGE_MIN_RATIO:
        # Too few edges (blank/uniform image)
        return 0.0
    else:
        # Too many edges (text, detailed natural scene)
        # Penalize aggressively — text typically >20% edge ratio
        # Fundus never exceeds ~25%
        excess = edge_ratio - FUNDUS_EDGE_MAX_RATIO
        return max(0.0, 0.2 - excess * 3)


def _check_green_channel(image: np.ndarray) -> float:
    """Check green channel statistics for fundus-like vessel contrast.

    In fundus images, the green channel shows the best contrast for blood
    vessels. The standard deviation should be moderate (20-60), not too
    uniform (text/documents) or too variable (natural scenes).

    Returns score 0.0-1.0 (1.0 = ideal fundus green channel variance).
    """
    green = image[:, :, 1]
    std_dev = np.std(green)

    # Fundus images: green channel std typically 20-60
    if 20 <= std_dev <= 60:
        # Peak at 35-45 (typical fundus)
        ideal = 40
        dist = abs(std_dev - ideal)
        return max(0.0, 1.0 - dist / 25)
    elif FUNDUS_GREEN_CH_MIN_STD <= std_dev < 20:
        return 0.3 + 0.7 * (std_dev - FUNDUS_GREEN_CH_MIN_STD) / (20 - FUNDUS_GREEN_CH_MIN_STD)
    elif 60 < std_dev <= 80:
        return 0.5 * (1.0 - (std_dev - 60) / 20)
    else:
        return 0.0


def _check_texture_regularity(image: np.ndarray, gray: np.ndarray = None) -> float:
    """Check for text/document patterns using horizontal line detection.

    Documents and text images have strong horizontal line structure
    (text baselines, ruled lines, uniform row spacing). Fundus images
    have organic, irregular textures from blood vessels and the optic disc.

    Uses horizontal projection profile to detect regular line patterns.

    Returns score 0.0-1.0 (1.0 = organic/fundus-like, 0.0 = text/document).
    """
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection: sum of dark pixels per row
    h_proj = np.sum(binary, axis=1) / 255.0

    # Normalize to [0, 1]
    if h_proj.max() > 0:
        h_proj = h_proj / h_proj.max()
    else:
        return 0.0

    # Detect regular peaks (text lines create periodic peaks)
    # Smooth the projection to find major peaks
    kernel_size = max(3, h // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(h_proj, np.ones(kernel_size) / kernel_size, mode="same")

    # Count peaks (local maxima above threshold)
    peak_count = 0
    threshold = 0.15
    in_peak = False
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > threshold and smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            if not in_peak:
                peak_count += 1
                in_peak = True
        elif smoothed[i] < threshold * 0.5:
            in_peak = False

    # Documents typically have 5+ regularly spaced peaks (text lines)
    # Fundus images have 0-3 irregular peaks
    if peak_count >= 8:
        # Very regular — likely text document
        return 0.0
    elif peak_count >= 5:
        # Somewhat regular — possibly document or structured image
        return 0.2
    elif peak_count <= 2:
        # Irregular — organic (fundus-like)
        return 0.8
    else:
        # Moderate — ambiguous
        return 0.5


def validate_fundus_image(image_path: str) -> dict[str, Any]:
    """Validate whether an image is a retinal fundus photograph.

    Uses 4 independent heuristic signals to determine if an image is
    a fundus photograph suitable for retinal disease classification.

    Args:
        image_path: Path to the image file

    Returns:
        dict with keys:
            is_fundus (bool): Whether the image passes fundus validation
            confidence (float): Combined confidence score 0.0-1.0
            signals (dict): Individual signal scores
            message (str): Human-readable result message

    """
    # Read image with OpenCV
    image = cv2.imread(image_path)
    if image is None:
        return {
            "is_fundus": False,
            "confidence": 0.0,
            "signals": {},
            "message": "Could not read image for fundus validation",
        }

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pre-compute color spaces once (avoids redundant conversions across 5 signals)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Compute individual signals (pass pre-computed images to avoid redundant conversions)
    color_score = _check_color_distribution(image, hsv=hsv)
    circular_score = _check_circular_region(image, gray=gray)
    edge_score = _check_edge_density(image, gray=gray)
    green_score = _check_green_channel(image)
    texture_score = _check_texture_regularity(image, gray=gray)

    signals = {
        "color_distribution": round(color_score, 4),
        "circular_region": round(circular_score, 4),
        "edge_density": round(edge_score, 4),
        "green_channel": round(green_score, 4),
        "texture_regularity": round(texture_score, 4),
    }

    # Weighted combination — 5 signals
    # texture_regularity gets high weight because it's the strongest
    # discriminator between text/documents and fundus images.
    combined_score = (
        color_score * 0.25 + circular_score * 0.20 + edge_score * 0.15 + green_score * 0.10 + texture_score * 0.30
    )

    # Second gate: require color signal to be decent.
    # A fundus image ALWAYS has reddish-orange tones. If color_score is 0,
    # the image lacks the most fundamental fundus characteristic, regardless
    # of how other signals score. This catches clean UIs and minimal images
    # that pass on edge/texture alone.
    has_fundus_color = color_score >= 0.20

    is_fundus = combined_score >= FUNDUS_VALIDATION_THRESHOLD and has_fundus_color

    # Generate human-readable message
    if is_fundus:
        message = "Image identified as a retinal fundus photograph."
    else:
        # Identify which signals failed most
        weak_signals = []
        if color_score < 0.3:
            weak_signals.append("color pattern")
        if circular_score < 0.3:
            weak_signals.append("circular structure")
        if edge_score < 0.3:
            weak_signals.append("texture patterns")
        if green_score < 0.3:
            weak_signals.append("retinal features")
        if texture_score < 0.3:
            weak_signals.append("document/text detected")

        if weak_signals:
            detail = ", ".join(weak_signals)
            message = (
                f"Image does not appear to be a retinal fundus photograph "
                f"(weak signals: {detail}). "
                f"Please upload a retinal fundus image taken with a fundus camera."
            )
        else:
            message = (
                "Image does not appear to be a retinal fundus photograph. "
                "Please upload a retinal fundus image taken with a fundus camera."
            )

    logger.debug(
        "Fundus validation: score=%.3f, is_fundus=%s, signals=%s",
        combined_score,
        is_fundus,
        signals,
    )

    result = {
        "is_fundus": is_fundus,
        "confidence": round(combined_score, 4),
        "signals": signals,
        "message": message,
    }

    # Optional: learned classifier as second gate
    if FUNDUS_LEARNED_VALIDATOR_ENABLED and is_fundus:
        try:
            import os

            from retina_app.services.fundus_classifier import FundusClassifier

            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                FUNDUS_LEARNED_MODEL_PATH,
            )
            if os.path.exists(model_path):
                classifier = FundusClassifier.load(model_path)
                from PIL import Image as PILImage

                pil_img = PILImage.open(image_path).convert("RGB")
                learned_result = classifier.predict_from_pil(pil_img)

                result["learned_fundus_prob"] = learned_result["probability"]
                result["learned_fundus_vote"] = learned_result["is_fundus"]

                # Both must agree for final decision
                if not learned_result["is_fundus"]:
                    result["is_fundus"] = False
                    result["message"] = (
                        "Image passed heuristic validation but was rejected by "
                        "learned fundus classifier (probability: "
                        f"{learned_result['probability']:.3f}). "
                        "Please upload a clear retinal fundus image."
                    )
        except Exception as e:
            logger.debug("Learned fundus classifier unavailable: %s", e)

    return result
