"""Deterministic transform pipelines for inference and TTA."""

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Shared pre-resize to avoid redundant resize in TTA variants
_PRERESIZE_224 = transforms.Resize((224, 224))
_PRERESIZE_256 = transforms.Resize((256, 256))
_TO_NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])

TRANSFORM = transforms.Compose(
    [
        _PRERESIZE_224,
        _TO_NORM,
    ]
)

# Transform for models trained WITHOUT ImageNet normalization (e.g., efficientnet_b0)
TRANSFORM_RAW = transforms.Compose(
    [
        _PRERESIZE_224,
        transforms.ToTensor(),
    ]
)

# TTA transforms: avoid redundant resize by reusing pre-resize
TRANSFORMS = {
    "standard": TRANSFORM,
    "augmented": transforms.Compose(
        [
            _PRERESIZE_256,
            transforms.Lambda(lambda x: F.hflip(x)),
            transforms.Lambda(lambda x: F.rotate(x, 10)),
            transforms.CenterCrop((224, 224)),
            _TO_NORM,
        ]
    ),
    "rotate90": transforms.Compose(
        [
            _PRERESIZE_256,
            transforms.Lambda(lambda x: F.rotate(x, 90)),
            transforms.CenterCrop((224, 224)),
            _TO_NORM,
        ]
    ),
    "rotate270": transforms.Compose(
        [
            _PRERESIZE_256,
            transforms.Lambda(lambda x: F.rotate(x, 270)),
            transforms.CenterCrop((224, 224)),
            _TO_NORM,
        ]
    ),
    "hflip": transforms.Compose(
        [
            _PRERESIZE_224,
            transforms.Lambda(lambda x: F.hflip(x)),
            _TO_NORM,
        ]
    ),
    "scale_90": transforms.Compose(
        [
            transforms.Resize((int(224 * 1.1), int(224 * 1.1))),
            transforms.CenterCrop((224, 224)),
            _TO_NORM,
        ]
    ),
    "scale_110": transforms.Compose(
        [
            transforms.Resize((int(224 * 1.15), int(224 * 1.15))),
            transforms.CenterCrop((224, 224)),
            _TO_NORM,
        ]
    ),
}

# For TTA: pre-resize image to 256 once, then apply augmentation transforms
# This avoids re-resizing the same image 3 times (augmented, rotate90, rotate270)
PRERESIZE_256 = _PRERESIZE_256
