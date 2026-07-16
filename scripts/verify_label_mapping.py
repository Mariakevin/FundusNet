"""Verify the corrected MODEL_LABEL_MAP for efficientnet_b0.

Loads the ONNX model, runs inference with TRANSFORM (ImageNet normalization)
on several test images, and prints results to confirm the mapping is correct.

Usage: python scripts/verify_label_mapping.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
import onnxruntime as ort
from PIL import Image

from retina_app.constants import MODEL_LABEL_MAP
from retina_app.services.transforms import TRANSFORM

# Resolve paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "efficientnet_b0_retinopathy.onnx")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "media", "uploads")

# The corrected mapping
LABEL_MAP = MODEL_LABEL_MAP["efficientnet_b0"]
CATEGORIES_ORDER = ["Cataract", "Retina Disease", "Glaucoma", "Healthy"]  # index 0..3


def get_test_images(n: int = 8) -> list[str]:
    """Pick n diverse test images from media/uploads/."""
    all_images = (
        glob.glob(os.path.join(UPLOADS_DIR, "*.jpg"))
        + glob.glob(os.path.join(UPLOADS_DIR, "*.jpeg"))
        + glob.glob(os.path.join(UPLOADS_DIR, "*.png"))
    )
    # Filter out .gitkeep and other non-image files
    all_images = [img for img in all_images if not img.endswith(".gitkeep")]

    if not all_images:
        return []

    # Pick evenly spaced images across the list for diversity
    if len(all_images) <= n:
        return all_images

    indices = np.linspace(0, len(all_images) - 1, n, dtype=int)
    return [all_images[i] for i in indices]


def main():
    print("=" * 80)
    print("LABEL MAPPING VERIFICATION")
    print("=" * 80)
    print(f"\nModel path: {MODEL_PATH}")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")
    print(f"Corrected MODEL_LABEL_MAP: {LABEL_MAP}")
    print(f"  Index 0 -> {LABEL_MAP[0]} (model's 'cataract' class)")
    print(f"  Index 1 -> {LABEL_MAP[1]} (model's 'diabetic_retinopathy' class)")
    print(f"  Index 2 -> {LABEL_MAP[2]} (model's 'glaucoma' class)")
    print(f"  Index 3 -> {LABEL_MAP[3]} (model's 'normal' class)")

    if not os.path.exists(MODEL_PATH):
        print("\nERROR: ONNX model not found!")
        sys.exit(1)

    # Load ONNX model
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"ONNX input name: {input_name}")
    print(f"ONNX input shape: {session.get_inputs()[0].shape}")

    # Get test images
    test_images = get_test_images(n=8)
    if not test_images:
        print("\nERROR: No test images found in media/uploads/")
        sys.exit(1)

    print(f"\nTesting with {len(test_images)} images from media/uploads/")
    print("=" * 80)

    # Results for summary table
    results = []

    for image_path in test_images:
        filename = os.path.basename(image_path)

        with Image.open(image_path) as pil_img:
            image = pil_img.convert("RGB")

            # Use TRANSFORM (ImageNet normalization) — this is what ensemble.py uses
            input_tensor = TRANSFORM(image).unsqueeze(0).numpy().astype(np.float32)

            # Run inference
            outputs = session.run(None, {input_name: input_tensor})
            logits = outputs[0][0]

            # Compute probabilities via softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            label = LABEL_MAP[pred_idx]

            # Guess the disease from filename (heuristic)
            filename_lower = filename.lower()
            if "cataract" in filename_lower:
                expected = "Cataract"
            elif "glaucoma" in filename_lower:
                expected = "Glaucoma"
            elif "retina" in filename_lower or "dr" in filename_lower or "diabetic" in filename_lower:
                expected = "Retina Disease"
            elif "normal" in filename_lower or "healthy" in filename_lower:
                expected = "Healthy"
            else:
                expected = "?"

            match_str = "MATCH" if label == expected else ("UNKNOWN EXPECTED" if expected == "?" else "MISMATCH")

            results.append({
                "filename": filename,
                "label": label,
                "confidence": confidence,
                "pred_idx": pred_idx,
                "expected": expected,
                "match": match_str,
            })

            # Print detailed output for this image
            print(f"\n--- {filename} ---")
            print(f"  Raw logits:      [{', '.join(f'{v:.4f}' for v in logits)}]")
            print(f"  Probabilities:   [{', '.join(f'{v:.4f}' for v in probs)}]")
            print(f"  Predicted index: {pred_idx}")
            print(f"  Confidence:      {confidence:.4f}")
            print(f"  Label (new map): {label}")
            if expected != "?":
                print(f"  Filename hint:   {expected}")
                print(f"  Verdict:         {match_str}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Filename':<40} {'Predicted':<20} {'Idx':<5} {'Conf':<8} {'Expected':<18} {'Status'}")
    print("-" * 100)
    for r in results:
        print(f"{r['filename']:<40} {r['label']:<20} {r['pred_idx']:<5} {r['confidence']:<8.4f} {r['expected']:<18} {r['match']}")

    # Aggregate stats
    known = [r for r in results if r["expected"] != "?"]
    matches = [r for r in known if r["match"] == "MATCH"]
    mismatches = [r for r in known if r["match"] == "MISMATCH"]

    print(f"\nTotal images:       {len(results)}")
    print(f"Known expected:     {len(known)}")
    print(f"Matches:            {len(matches)}")
    print(f"Mismatches:         {len(mismatches)}")

    if mismatches:
        print("\nMISMATCHES (filename vs prediction):")
        for r in mismatches:
            print(f"  {r['filename']}: predicted={r['label']}, expected={r['expected']}")
    else:
        print("\nAll known-label images matched the corrected mapping.")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("The corrected mapping is:")
    print("  Index 0 -> Cataract")
    print("  Index 1 -> Retina Disease (diabetic_retinopathy)")
    print("  Index 2 -> Glaucoma")
    print("  Index 3 -> Healthy (normal)")
    print("=" * 80)


if __name__ == "__main__":
    main()
