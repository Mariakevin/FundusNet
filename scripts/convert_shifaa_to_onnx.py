"""Convert Shifaa pre-trained EfficientNet-B0 to ONNX format.

Downloads the model from HuggingFace, rebuilds the architecture,
loads weights, and exports to ONNX for runtime inference.

Usage:
    python scripts/convert_shifaa_to_onnx.py

Requirements:
    pip install efficientnet-pytorch huggingface-hub onnx torch
"""

import sys
from pathlib import Path

# Resolve project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_PATH = MODELS_DIR / "efficientnet_b0_retinopathy.onnx"

REPO_ID = "Ahmed-Selem/Shifaa-Eye-Disease-EfficientNetB0"
FILENAME = "efficientnet_b0_Eye_Diseases.pth"


def ensure_dependencies():
    """Install required packages if missing."""
    missing = []
    try:
        import efficientnet_pytorch  # noqa: F401
    except ImportError:
        missing.append("efficientnet-pytorch")
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        missing.append("huggingface-hub")
    try:
        import onnx  # noqa: F401
    except ImportError:
        missing.append("onnx")

    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def build_model():
    """Build the EfficientNet-B0 architecture with custom classification head."""
    import torch
    import torch.nn as nn
    from efficientnet_pytorch import EfficientNet

    model = EfficientNet.from_name("efficientnet-b0")
    model._fc = nn.Sequential(
        nn.Linear(model._fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4),
    )
    return model


def download_weights():
    """Download the pre-trained weights from HuggingFace."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    print(f"Downloaded weights to: {path}")
    return path


def load_weights(model, weights_path):
    """Load state dict into the model."""
    import torch

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"Loaded weights from {weights_path}")
    return model


def export_to_onnx(model, output_path):
    """Export model to ONNX format with dynamic batching."""
    import torch

    # CRITICAL: disable memory-efficient swish for ONNX compatibility
    model.set_swish(memory_efficient=False)

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported ONNX model to: {output_path}")


def verify_onnx(output_path):
    """Verify the exported ONNX model is valid."""
    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed")


def main():
    ensure_dependencies()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Shifaa EfficientNet-B0 -> ONNX Conversion")
    print("=" * 60)

    # Step 1: Download weights
    print("\n[1/5] Downloading pre-trained weights...")
    weights_path = download_weights()

    # Step 2: Build architecture
    print("\n[2/5] Building model architecture...")
    model = build_model()

    # Step 3: Load weights
    print("\n[3/5] Loading weights into model...")
    model = load_weights(model, weights_path)

    # Step 4: Export to ONNX
    print("\n[4/5] Exporting to ONNX...")
    export_to_onnx(model, OUTPUT_PATH)

    # Step 5: Verify
    print("\n[5/5] Verifying ONNX model...")
    verify_onnx(OUTPUT_PATH)

    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDone! Output: {OUTPUT_PATH} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
