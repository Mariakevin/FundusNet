"""ONNX export for retina models.

Exports trained models to ONNX format for 3-5x faster inference
via ONNX Runtime or TensorRT.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "retina_project.settings")

try:
    import django

    django.setup()
    from retina_app.constants import CATEGORIES
except Exception:
    CATEGORIES = ["Healthy", "Cataract", "Glaucoma", "Retina Disease"]

try:
    from retina_app.utils import create_model
except ImportError:
    def create_model(model_name, num_classes=4, pretrained=True):
        import timm
        from retina_app.constants import MODEL_NAME_MAP
        timm_name = MODEL_NAME_MAP.get(model_name, model_name)
        return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Export retina models to ONNX")
parser.add_argument("--model", default="efficientnet_v2", help="Model type to export")
parser.add_argument("--checkpoint", help="Path to .pth checkpoint (optional)")
parser.add_argument("--output", help="Output ONNX path (optional)")
parser.add_argument("--input-size", type=int, default=224)
parser.add_argument("--opset", type=int, default=17)
parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch size")
parser.add_argument("--verify", action="store_true", help="Verify ONNX model")
parser.add_argument("--quantize", action="store_true", help="Apply dynamic quantization")
args = parser.parse_args()


def load_model(model_type, checkpoint_path=None, num_classes=4):
    model = create_model(model_type, num_classes)
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def export_to_onnx(model, output_path, input_size=224, opset_version=17, dynamic_axes=False):
    dummy_input = torch.randn(1, 3, input_size, input_size)

    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes_dict,
    )
    print(f"Exported to: {output_path}")
    return output_path


def verify_onnx(onnx_path, input_size=224):
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        dummy = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
        outputs = session.run(None, {"input": dummy})
        print(f"ONNX verification passed: output shape={outputs[0].shape}")
        return True
    except ImportError:
        print("onnxruntime not installed, skipping verification")
        return False
    except Exception as e:
        print(f"ONNX verification failed: {e}")
        return False


def quantize_onnx(input_path, output_path):
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(input_path, output_path, weight_type=QuantType.QUInt8)
        print(f"Quantized model saved to: {output_path}")
        return True
    except ImportError:
        print("onnxruntime not installed, skipping quantization")
        return False
    except Exception as e:
        print(f"Quantization failed: {e}")
        return False


def main():
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = os.path.join(SCRIPT_DIR, "models", f"{args.model}_retinopathy.pth")

    output_path = args.output
    if not output_path:
        output_path = os.path.join(SCRIPT_DIR, "models", f"{args.model}_retinopathy.onnx")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")

    model = load_model(args.model, checkpoint_path, len(CATEGORIES))
    export_to_onnx(model, output_path, args.input_size, args.opset, args.dynamic)

    if args.verify:
        verify_onnx(output_path, args.input_size)

    if args.quantize:
        quant_path = output_path.replace(".onnx", "_quantized.onnx")
        quantize_onnx(output_path, quant_path)

    print("Done!")


if __name__ == "__main__":
    main()
