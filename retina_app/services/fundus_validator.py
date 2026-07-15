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

    # Convex hull: fundus region is convex (smooth circle/ellipse).
    # Faces/organic shapes have concave features (eyes, mouth, hair gaps)
    # that make contour area much smaller than convex hull area.
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0.0  # 1.0 = perfectly convex

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

    # --- Corner darkness check ---
    # Fundus images ALWAYS have at least 3 dark corners because the
    # circular field of view is smaller than the camera sensor.
    # Non-fundus images (landscapes, objects, screenshots) fill the
    # frame and have 0-1 dark corners.
    # A corner is "dark" if the mean pixel value in a small kernel
    # at the corner is below a threshold (Otsu's own threshold
    # for the image).
    corner_size = max(8, min(h, w) // 20)
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    corners = [
        gray[:corner_size, :corner_size],  # top-left
        gray[:corner_size, -corner_size:],  # top-right
        gray[-corner_size:, :corner_size],  # bottom-left
        gray[-corner_size:, -corner_size:],  # bottom-right
    ]
    dark_corners = sum(1 for c in corners if np.mean(c) < otsu_thresh * 0.85)
    corner_score = dark_corners / 4.0  # 0.0 → 1.0

    # Score based on circularity, convexity, area ratio, dark surround, and corner darkness
    circ_score = 0.0
    if circularity >= FUNDUS_CIRCULARITY_MIN:
        circ_score = min(1.0, circularity / 0.8)

    # Convexity score: 0 for non-convex (faces, irregular shapes), 1 for fundus-like
    convex_score = max(0.0, min(1.0, (convexity - 0.7) / 0.25))

    area_score = 0.0
    if FUNDUS_AREA_MIN_RATIO <= area_ratio <= FUNDUS_AREA_MAX_RATIO:
        dist_from_ideal = abs(area_ratio - 0.55)
        area_score = max(0.0, 1.0 - dist_from_ideal / 0.40)

    return 0.25 * circ_score + 0.20 * convex_score + 0.15 * area_score + 0.20 * surround_score + 0.20 * corner_score


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


def _check_channel_ratio(image: np.ndarray) -> float:
    """Check red-to-green channel ratio for fundus-like characteristics.

    In fundus images, the red channel (retinal surface illumination) is
    consistently brighter than the green channel (blood vessel contrast).
    Natural scenes and object photos often have green as bright or brighter
    than red.

    Fundus:  red_mean / green_mean typically 1.4–3.0
    Natural: red_mean / green_mean typically 0.7–1.3

    Returns score 0.0-1.0 (1.0 = strong fundus channel relationship).
    """
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)

    # Central region weight: fundus structures are in the center
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 3
    yy, xx = np.ogrid[:h, :w]
    center_mask = ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= radius**2

    r_center = r[center_mask].mean()
    g_center = g[center_mask].mean()

    r_full = r.mean()
    g_full = g.mean()

    # Blend: 60% center, 40% full
    r_mean = 0.6 * r_center + 0.4 * r_full
    g_mean = 0.6 * g_center + 0.4 * g_full

    if g_mean < 1.0:
        return 0.0

    ratio = r_mean / g_mean

    # Fundus: ratio 1.4–3.0, peak at ~1.8–2.2
    if 1.4 <= ratio <= 3.0:
        ideal = 2.0
        dist = abs(ratio - ideal)
        return max(0.3, 1.0 - dist / 1.2)
    elif 1.1 <= ratio < 1.4:
        return 0.2 + 0.8 * (ratio - 1.1) / 0.3
    elif 3.0 < ratio <= 3.5:
        return 0.5 * (1.0 - (ratio - 3.0) / 0.5)
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


def validate_fundus_image(image_path: str, pil_image=None) -> dict[str, Any]:
    """Validate whether an image is a retinal fundus photograph.

    Uses 4 independent heuristic signals to determine if an image is
    a fundus photograph suitable for retinal disease classification.

    Args:
        image_path: Path to the image file
        pil_image: Optional pre-loaded PIL Image (avoids re-reading)

    Returns:
        dict with keys:
            is_fundus (bool): Whether the image passes fundus validation
            confidence (float): Combined confidence score 0.0-1.0
            signals (dict): Individual signal scores
            message (str): Human-readable result message

    """
    # Use pre-loaded image or read from disk
    if pil_image is not None:
        image = np.array(pil_image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image = cv2.imread(image_path)
        if image is None:
            return {
                "is_fundus": False,
                "confidence": 0.0,
                "signals": {},
                "message": "Could not read image for fundus validation",
            }
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pre-compute color spaces once (avoids redundant conversions across 5 signals)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Compute individual signals (pass pre-computed images to avoid redundant conversions)
    color_score = _check_color_distribution(image, hsv=hsv)
    circular_score = _check_circular_region(image, gray=gray)
    edge_score = _check_edge_density(image, gray=gray)
    green_score = _check_green_channel(image)
    channel_ratio_score = _check_channel_ratio(image)
    texture_score = _check_texture_regularity(image, gray=gray)

    signals = {
        "color_distribution": round(color_score, 4),
        "circular_region": round(circular_score, 4),
        "edge_density": round(edge_score, 4),
        "green_channel": round(green_score, 4),
        "channel_ratio": round(channel_ratio_score, 4),
        "texture_regularity": round(texture_score, 4),
    }

    # Weighted combination — 6 signals
    # channel_ratio and texture_regularity get high weight because they
    # are the strongest discriminators between fundus and non-fundus images.
    combined_score = (
        color_score * 0.20
        + circular_score * 0.20
        + edge_score * 0.10
        + green_score * 0.08
        + channel_ratio_score * 0.20
        + texture_score * 0.22
    )

    # Second gate: require color signal to be decent.
    # A fundus image ALWAYS has reddish-orange tones. If color_score is 0,
    # the image lacks the most fundamental fundus characteristic, regardless
    # of how other signals score. This catches clean UIs and minimal images
    # that pass on edge/texture alone.
    has_fundus_color = color_score >= 0.25

    # Third gate: require circular structure.
    # A fundus image ALWAYS has a bright circular field of view with a dark
    # surround from the camera aperture. Faces, landscapes, and other organic
    # images lack this and must be rejected.
    has_circular_structure = circular_score >= 0.35

    # Determine which gate failed (for messaging)
    gate_failures = []
    if not has_fundus_color:
        gate_failures.append("fundus color signature")
    if not has_circular_structure:
        gate_failures.append("circular fundus structure")

    is_fundus = combined_score >= FUNDUS_VALIDATION_THRESHOLD and has_fundus_color and has_circular_structure

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
        elif circular_score < 0.4:
            weak_signals.append("weak circular structure")
        if edge_score < 0.3:
            weak_signals.append("texture patterns")
        if green_score < 0.3:
            weak_signals.append("retinal features")
        if channel_ratio_score < 0.3:
            weak_signals.append("channel contrast")
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

    return result
