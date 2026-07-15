"""Refusal logic for out-of-distribution and low-confidence predictions.

Combines 4 independent checks:
1. Uncertainty threshold (MC Dropout entropy)
2. Confidence threshold (max probability)
3. Normalized entropy (OOD detection)
4. Top-1/Top-2 margin (ambiguity detection)
5. Energy-based OOD (logit analysis)

If any check triggers, the prediction is refused and the user is
directed to consult an ophthalmologist.
"""

import logging
from typing import Any

import numpy as np

from retina_app.constants import (
    CATEGORIES,
    CONFIDENCE_THRESHOLD_REFUSE,
    FUNDUS_MIN_TOP1_TOP2_RATIO,
    OOD_ENTROPY_THRESHOLD,
    UNCERTAINTY_REFUSAL_MESSAGE,
)

logger = logging.getLogger("retina_app")


class RefusalResult:
    """Result of refusal check."""

    __slots__ = ("is_refused", "reason", "confidence", "label")

    def __init__(
        self,
        is_refused: bool = False,
        reason: str = "",
        confidence: float = 0.0,
        label: str = "Uncertain",
    ):
        self.is_refused = is_refused
        self.reason = reason
        self.confidence = confidence
        self.label = label

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_refused": self.is_refused,
            "refusal_reason": self.reason,
            "refusal_message": UNCERTAINTY_REFUSAL_MESSAGE if self.is_refused else "",
        }


def check_uncertainty(
    uncertainty_data: dict[str, Any] | None,
) -> RefusalResult | None:
    """Check 1: MC Dropout uncertainty threshold."""
    if uncertainty_data and uncertainty_data.get("is_uncertain"):
        entropy = uncertainty_data.get("entropy", 0.0)
        logger.warning("Classification refused: uncertainty=%.4f > threshold", entropy)
        return RefusalResult(
            is_refused=True,
            reason=f"High uncertainty (entropy={entropy:.4f})",
        )
    return None


def check_confidence(confidence: float) -> RefusalResult | None:
    """Check 2: Minimum confidence threshold."""
    if confidence < CONFIDENCE_THRESHOLD_REFUSE:
        logger.warning(
            "Classification refused: low confidence %.2f < %.2f",
            confidence,
            CONFIDENCE_THRESHOLD_REFUSE,
        )
        return RefusalResult(
            is_refused=True,
            reason=f"Low confidence ({confidence:.2f} < {CONFIDENCE_THRESHOLD_REFUSE})",
        )
    return None


def check_entropy(probabilities: list[float]) -> RefusalResult | None:
    """Check 3: Normalized entropy OOD detection.

    OOD images produce near-uniform predictions (high entropy) even when
    the max confidence is above the refusal threshold. In-distribution
    fundus images have peaked distributions (low entropy).
    """
    if not probabilities or len(probabilities) <= 1:
        return None

    try:
        from scipy.stats import entropy as scipy_entropy

        probs = np.array(probabilities, dtype=np.float64)
        probs = probs / probs.sum()
        norm_entropy = scipy_entropy(probs) / np.log(len(probs))

        if norm_entropy > OOD_ENTROPY_THRESHOLD:
            logger.warning(
                "Classification refused: OOD entropy %.4f > %.4f",
                norm_entropy,
                OOD_ENTROPY_THRESHOLD,
            )
            return RefusalResult(
                is_refused=True,
                reason=f"High normalized entropy ({norm_entropy:.4f} > {OOD_ENTROPY_THRESHOLD})",
            )
    except ImportError:
        logger.warning("scipy not available, skipping entropy check")

    return None


def check_margin(probabilities: list[float]) -> RefusalResult | None:
    """Check 4: Top-1/Top-2 margin.

    OOD images have a narrow margin between the top class and the runner-up,
    while real fundus predictions have a clear winner.
    """
    if not probabilities or len(probabilities) < 2:
        return None

    sorted_probs = sorted(probabilities, reverse=True)
    margin = sorted_probs[0] / max(sorted_probs[1], 1e-8)

    if margin < FUNDUS_MIN_TOP1_TOP2_RATIO:
        logger.warning(
            "Classification refused: top-1/top-2 margin %.2f < %.2f",
            margin,
            FUNDUS_MIN_TOP1_TOP2_RATIO,
        )
        return RefusalResult(
            is_refused=True,
            reason=f"Narrow margin ({margin:.2f} < {FUNDUS_MIN_TOP1_TOP2_RATIO})",
        )
    return None


def check_energy(predictions: list[tuple[str, dict]]) -> RefusalResult | None:
    """Check 5: Energy-based OOD detection.

    Uses raw logits (before softmax). OOD images produce low-energy logits
    across ALL classes, even when the softmax-normalized top class seems
    confident. Energy score: E(x) = log(sum(exp(logits_i)))
    Higher = in-distribution, Lower = OOD.
    """
    if not predictions or len(predictions) < 2:
        return None

    all_logits = []
    for _, pred in predictions:
        logits = pred.get("logits")
        if logits and len(logits) == len(CATEGORIES):
            all_logits.append(logits)

    if len(all_logits) < 2:
        return None

    avg_logits = np.mean(all_logits, axis=0)
    energy = np.log(np.sum(np.exp(avg_logits - np.max(avg_logits)))) + np.max(avg_logits)
    n_classes = len(avg_logits)
    energy_per_class = energy / n_classes

    if energy_per_class < 0.6:
        logger.warning("Classification refused: low energy score %.4f", energy_per_class)
        return RefusalResult(
            is_refused=True,
            reason=f"Low energy score ({energy_per_class:.4f} < 0.6)",
        )
    return None


def check_all_refusals(
    confidence: float,
    probabilities: list[float],
    predictions: list[tuple[str, dict]],
    uncertainty_data: dict[str, Any] | None,
    use_ensemble: bool,
) -> RefusalResult:
    """Run all refusal checks in order. Return first triggered refusal.

    Checks run in order of cheapest to most expensive:
    1. Uncertainty (already computed)
    2. Confidence (simple comparison)
    3. Entropy (scipy)
    4. Margin (sorting)
    5. Energy (logit analysis, most expensive)
    """
    # Check 1: Uncertainty
    result = check_uncertainty(uncertainty_data)
    if result:
        return result

    # Check 2: Confidence
    result = check_confidence(confidence)
    if result:
        return result

    if not use_ensemble:
        return RefusalResult(is_refused=False)

    # Check 3: Entropy
    result = check_entropy(probabilities)
    if result:
        return result

    # Check 4: Margin
    result = check_margin(probabilities)
    if result:
        return result

    # Check 5: Energy
    result = check_energy(predictions)
    if result:
        return result

    return RefusalResult(is_refused=False)
