"""Grad-CAM Explainability.

Generate heatmaps showing which regions of the retinal image
drove the classification decision. Based on Selvaraju et al. (ICCV 2017).
"""

import logging
import os
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from retina_app.constants import CATEGORIES, GRADCAM_ALPHA, GRADCAM_COLORMAP
from retina_app.services.transforms import TRANSFORM

logger = logging.getLogger("retina_app")


# Map model types to their last convolutional layer names
GRADCAM_TARGET_LAYERS = {
    "swin": "features",
    "maxvit": "stages",
    "convnext_v2": "features",
    "efficientnet_v2": "features",
    "deit": "encoder",
}


def _get_target_layer(model: nn.Module, model_type: str) -> nn.Module:
    """Get the target layer for Grad-CAM based on model architecture."""
    if model_type == "swin":
        # Swin Transformer has hierarchical features
        if hasattr(model, "features"):
            return model.features[-1]
    elif model_type == "maxvit":
        # MaxViT has stages containing attention blocks
        if hasattr(model, "stages"):
            return model.stages[-1]
    elif model_type in ("convnext_v2", "efficientnet_v2"):
        if hasattr(model, "features"):
            return model.features[-1]
    elif model_type == "deit":
        if hasattr(model, "encoder"):
            return model.encoder

    # Fallback: auto-detect
    if hasattr(model, "features"):
        return model.features[-1]
    elif hasattr(model, "encoder"):
        return model.encoder
    elif hasattr(model, "layer4"):
        return model.layer4
    return next(model.modules())


class GradCAM:
    """Grad-CAM implementation for CNN models.

    Captures activations and gradients from the target layer,
    computes class-discriminative localization maps.
    """

    def __init__(self, model: nn.Module, model_type: str):
        self.model = model
        self.model_type = model_type
        self.activations = None
        self.gradients = None

        target_layer = _get_target_layer(model, model_type)
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed input [1, C, H, W]
            target_class: Class index to explain. If None, uses predicted class.

        Returns:
            Tuple of (heatmap as numpy array [H, W] with values in [0, 1],
                      predicted class index, confidence score)

        """
        from retina_app.services.model_manager import DEVICE

        self.model.eval()

        # Forward pass — compute output once
        output = self.model(input_tensor.to(DEVICE))

        if isinstance(output, tuple):
            output = output[0]

        # Get target class
        if target_class is None:
            if len(output.shape) > 1:
                target_class = output.argmax(dim=1).item()
            else:
                target_class = output.argmax().item()

        # Compute confidence from output
        with torch.no_grad():
            probs = F.softmax(output, dim=1) if len(output.shape) > 1 else F.softmax(output.unsqueeze(0), dim=1)
            confidence = probs[0, target_class].item()

        # Backward pass for target class
        self.model.zero_grad()
        if len(output.shape) > 1:
            output[0, target_class].backward()
        else:
            output[target_class].backward()

        # Compute Grad-CAM
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Convert to numpy
        cam_np = cam.squeeze().cpu().numpy()

        return cam_np, target_class, confidence

    def cleanup(self) -> None:
        """Remove hooks and release tensors to prevent memory leaks."""
        self._forward_handle.remove()
        self._backward_handle.remove()
        self.activations = None
        self.gradients = None


def _deprocess_image(
    cam: np.ndarray,
    original_size: tuple[int, int],
) -> np.ndarray:
    """Resize heatmap to original image size and apply colormap."""
    cam_resized = cv2.resize(cam, original_size, interpolation=cv2.INTER_LINEAR)

    # Apply colormap
    cam_uint8 = (cam_resized * 255).astype(np.uint8)

    colormap = getattr(cv2, f"COLORMAP_{GRADCAM_COLORMAP.upper()}", cv2.COLORMAP_JET)
    cam_colored = cv2.applyColorMap(cam_uint8, colormap)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

    return cam_colored


def _blend_heatmap(
    original: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = GRADCAM_ALPHA,
) -> np.ndarray:
    """Blend original image with heatmap."""
    if original.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

    blended = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
    return blended


def generate_gradcam(
    model: nn.Module,
    image_path: str,
    model_type: str,
    predicted_class: int | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate Grad-CAM heatmap for a model's prediction.

    Args:
        model: PyTorch model
        image_path: Path to the input image
        model_type: Model type string (resnet, efficientnet, etc.)
        predicted_class: Class to explain. None = use model's prediction.
        output_path: Where to save the blended heatmap image.

    Returns:
        Dict with gradcam_url, predicted_class, confidence, etc.

    """
    # Load and preprocess image
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        original_size = img.size  # (W, H)
        original_np = np.array(rgb_img)
        input_tensor = TRANSFORM(rgb_img).unsqueeze(0)

    # Create GradCAM instance
    gradcam = GradCAM(model, model_type)

    try:
        # Generate heatmap + get prediction in single forward pass
        cam, pred_idx, confidence = gradcam.generate(input_tensor, predicted_class)

        # Deprocess and blend
        heatmap = _deprocess_image(cam, original_size)
        blended = _blend_heatmap(original_np, heatmap)

        # Save if output_path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, blended_bgr)
            logger.info(f"Grad-CAM saved to {output_path}")

        return {
            "predicted_class": CATEGORIES[pred_idx],
            "predicted_class_idx": pred_idx,
            "confidence": confidence,
            "heatmap_raw": cam,
            "output_path": output_path,
        }

    finally:
        gradcam.cleanup()


def get_gradcam_output_path(media_root: str, image_name: str) -> str:
    """Get output path for Grad-CAM heatmap image."""
    gradcam_dir = os.path.join(media_root, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_name))[0]
    return os.path.join(gradcam_dir, f"{base_name}_gradcam.png")
