"""Definitive diagnostic: test ONNX model output with EXACT inference pipeline preprocessing.

Usage: python scripts/debug_inference.py [image_path]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnxruntime as ort
import numpy as np
from PIL import Image
from retina_app.services.transforms import TRANSFORM_RAW, TRANSFORM
from retina_app.constants import CATEGORIES, MODEL_LABEL_MAP

# Resolve model path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "models", "efficientnet_b0_retinopathy.onnx")

print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")
print(f"CATEGORIES: {CATEGORIES}")
print(f"MODEL_LABEL_MAP['efficientnet_b0']: {MODEL_LABEL_MAP['efficientnet_b0']}")
print(f"Maps identical: {CATEGORIES == MODEL_LABEL_MAP['efficientnet_b0']}")

if not os.path.exists(model_path):
    print("ERROR: ONNX model not found!")
    sys.exit(1)

# Load ONNX model
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Test with an image
image_path = sys.argv[1] if len(sys.argv) > 1 else None
if not image_path:
    import glob
    uploads = (
        glob.glob(os.path.join(project_root, "media", "uploads", "*.jpg"))
        + glob.glob(os.path.join(project_root, "media", "uploads", "*.png"))
        + glob.glob(os.path.join(project_root, "media", "uploads", "*.jpeg"))
    )
    if uploads:
        image_path = uploads[0]
    else:
        print("No test image found. Usage: python scripts/debug_inference.py <image_path>")
        sys.exit(1)

print(f"\nTesting with: {image_path}")

# Open and preprocess EXACTLY like ensemble.py does for efficientnet_b0
with Image.open(image_path) as pil_img:
    image = pil_img.convert("RGB")
    print(f"Image size: {image.size}")

    # Test with TRANSFORM_RAW (what efficientnet_b0 uses in ensemble.py non-TTA path)
    tensor_raw = TRANSFORM_RAW(image).unsqueeze(0).numpy()

    # Test with TRANSFORM (what other models use / TTA uses)
    tensor_norm = TRANSFORM(image).unsqueeze(0).numpy()

    input_name = session.get_inputs()[0].name
    print(f"ONNX input name: {input_name}")
    print(f"Tensor shape (TRANSFORM_RAW): {tensor_raw.shape}")
    print(f"Tensor dtype (TRANSFORM_RAW): {tensor_raw.dtype}")
    print(f"Tensor range (TRANSFORM_RAW): [{tensor_raw.min():.4f}, {tensor_raw.max():.4f}]")
    print(f"Tensor shape (TRANSFORM): {tensor_norm.shape}")
    print(f"Tensor range (TRANSFORM): [{tensor_norm.min():.4f}, {tensor_norm.max():.4f}]")

    # Run with TRANSFORM_RAW
    print("\n" + "=" * 60)
    print("=== With TRANSFORM_RAW (no ImageNet normalization) ===")
    print("=" * 60)
    outputs = session.run(None, {input_name: tensor_raw.astype(np.float32)})
    logits_raw = outputs[0][0]
    exp_logits = np.exp(logits_raw - np.max(logits_raw))
    probs_raw = exp_logits / exp_logits.sum()
    pred_idx_raw = int(np.argmax(probs_raw))
    print(f"Raw logits:    {logits_raw}")
    print(f"Probabilities: {probs_raw}")
    print(f"Sum of probs:  {probs_raw.sum():.6f}")
    print(f"Predicted index: {pred_idx_raw}")
    print(f"Confidence:    {probs_raw[pred_idx_raw]:.4f}")
    print(f"CATEGORIES[{pred_idx_raw}]              = {CATEGORIES[pred_idx_raw]}")
    print(f"MODEL_LABEL_MAP['efficientnet_b0'][{pred_idx_raw}] = {MODEL_LABEL_MAP['efficientnet_b0'][pred_idx_raw]}")
    print()
    for i, (cat, mlm) in enumerate(zip(CATEGORIES, MODEL_LABEL_MAP["efficientnet_b0"])):
        marker = " <-- PREDICTED" if i == pred_idx_raw else ""
        print(f"  Index {i}: CATEGORIES={cat}, MODEL_LABEL_MAP={mlm}, prob={probs_raw[i]:.4f}{marker}")

    # Run with TRANSFORM
    print("\n" + "=" * 60)
    print("=== With TRANSFORM (ImageNet normalization) ===")
    print("=" * 60)
    outputs_norm = session.run(None, {input_name: tensor_norm.astype(np.float32)})
    logits_norm = outputs_norm[0][0]
    exp_logits_n = np.exp(logits_norm - np.max(logits_norm))
    probs_norm = exp_logits_n / exp_logits_n.sum()
    pred_idx_norm = int(np.argmax(probs_norm))
    print(f"Raw logits:    {logits_norm}")
    print(f"Probabilities: {probs_norm}")
    print(f"Sum of probs:  {probs_norm.sum():.6f}")
    print(f"Predicted index: {pred_idx_norm}")
    print(f"Confidence:    {probs_norm[pred_idx_norm]:.4f}")
    print(f"CATEGORIES[{pred_idx_norm}]              = {CATEGORIES[pred_idx_norm]}")
    print(f"MODEL_LABEL_MAP['efficientnet_b0'][{pred_idx_norm}] = {MODEL_LABEL_MAP['efficientnet_b0'][pred_idx_norm]}")
    print()
    for i, (cat, mlm) in enumerate(zip(CATEGORIES, MODEL_LABEL_MAP["efficientnet_b0"])):
        marker = " <-- PREDICTED" if i == pred_idx_norm else ""
        print(f"  Index {i}: CATEGORIES={cat}, MODEL_LABEL_MAP={mlm}, prob={probs_norm[i]:.4f}{marker}")

    # Compare entropy (higher = more uncertain)
    entropy_raw = -np.sum(probs_raw * np.log(probs_raw + 1e-10))
    entropy_norm = -np.sum(probs_norm * np.log(probs_norm + 1e-10))
    print(f"\nEntropy (TRANSFORM_RAW):  {entropy_raw:.4f}")
    print(f"Entropy (TRANSFORM):      {entropy_norm:.4f}")

    # Also test what ensemble.py actually does - the _predict_single_model path
    print("\n" + "=" * 60)
    print("=== Simulating ensemble.py _predict_single_model (non-TTA) ===")
    print("=" * 60)
    import torch
    import torch.nn.functional as F

    # ensemble.py does: TRANSFORM_RAW(image).unsqueeze(0) for efficientnet_b0
    input_tensor = TRANSFORM_RAW(image).unsqueeze(0).numpy().astype(np.float32)
    output = session.run(None, {input_name: input_tensor})[0][0]

    # ensemble.py does F.softmax(output, dim=0)
    output_tensor = torch.tensor(output, dtype=torch.float32)
    probs_ensemble = F.softmax(output_tensor, dim=0).numpy()
    max_idx_ensemble = int(np.argmax(probs_ensemble))
    confidence_ensemble = float(probs_ensemble[max_idx_ensemble])

    print(f"Output from ONNX: {output}")
    print(f"Softmax probs:    {probs_ensemble}")
    print(f"Predicted index:  {max_idx_ensemble}")
    print(f"Confidence:       {confidence_ensemble:.4f}")
    print(f"Label (CATEGORIES): {CATEGORIES[max_idx_ensemble]}")

    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    if entropy_raw < entropy_norm:
        print("TRANSFORM_RAW produces MORE confident predictions -> model was likely trained WITHOUT ImageNet normalization")
        print("ensemble.py correctly uses TRANSFORM_RAW for efficientnet_b0")
    else:
        print("TRANSFORM produces MORE confident predictions -> model may need ImageNet normalization")
        print("ensemble.py may need to use TRANSFORM instead of TRANSFORM_RAW for efficientnet_b0")

    # Check if predictions differ between transforms
    if pred_idx_raw != pred_idx_norm:
        print()
        print("WARNING: Different predictions from different preprocessing!")
        print(f"  TRANSFORM_RAW predicts: {CATEGORIES[pred_idx_raw]} (idx={pred_idx_raw})")
        print(f"  TRANSFORM predicts:     {CATEGORIES[pred_idx_norm]} (idx={pred_idx_norm})")
    else:
        print()
        print(f"Both preprocessing paths agree on: {CATEGORIES[pred_idx_raw]} (idx={pred_idx_raw})")
