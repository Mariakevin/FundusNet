"""Advanced data pipeline with mixed augmentation strategies.

Implements:
- RandAugment / TrivialAugment (auto-augmentation)
- MixUp / CutMix / ManifoldMixUp / CutBlur
- Class-balanced sampling with effective number of samples
- Multi-scale training (random resize crop)
- Advanced medical image augmentations
- Test-time augmentation (TTA) with K-Fold ensemble
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF

# ── Medical-Specific Augmentations ───────────────────────────────────────────


class CLAHETransform:
    """Apply Contrast Limited Adaptive Histogram Equalization."""

    def __init__(self, clip_limit: float = 2.0, grid_size: int = 8):
        self.clip_limit = clip_limit
        self.grid_size = grid_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        try:
            import cv2

            arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l_ch, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=(self.grid_size, self.grid_size),
            )
            l_ch = clahe.apply(l_ch)
            lab = cv2.merge([l_ch, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return torch.from_numpy(result.astype(np.float32) / 255.0).permute(2, 0, 1)
        except ImportError:
            return img


class GreenChannelExtract:
    """Extract green channel for blood vessel enhancement."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img[1:2, :, :]


class FundusNoiseInjection:
    """Simulate realistic fundus imaging noise."""

    def __init__(self, std_range: tuple = (0.01, 0.05)):
        self.std_range = std_range

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(img) * std
        return torch.clamp(img + noise, 0, 1)


class FundusBrightnessJitter:
    """Simulate uneven fundus illumination."""

    def __init__(self, intensity: float = 0.2):
        self.intensity = intensity

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        center_x = random.uniform(0.3, 0.7)
        center_y = random.uniform(0.3, 0.7)
        h, w = img.shape[1], img.shape[2]
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, 1, h),
            torch.linspace(0, 1, w),
            indexing="ij",
        )
        dist = ((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2).sqrt()
        brightness = 1.0 + self.intensity * (1 - dist / dist.max())
        return torch.clamp(img * brightness.unsqueeze(0), 0, 1)


# ── Advanced MixUp / CutMix Variants ────────────────────────────────────────


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.8) -> tuple:
    """Standard MixUp: convex combination of two samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> tuple:
    """CutMix: cut a patch from one image and paste on another."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, h, w = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)

    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
    y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (h * w)
    return mixed_x, y, y[index], lam


def cutblur(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> tuple:
    """CutBlur: cut a region and apply different resolution (simulates blur)."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, h, w = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)

    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
    y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)

    mixed_x = x.clone()
    blurred = TF.gaussian_blur(x[index], kernel_size=5)
    mixed_x[:, :, y1:y2, x1:x2] = blurred[:, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (h * w)
    return mixed_x, y, y[index], lam


def manifold_mixup(
    x: torch.Tensor, y: torch.Tensor, model: torch.nn.Module, layer_idx: int = -1, alpha: float = 0.2
) -> tuple:
    """ManifoldMixUp: mix hidden representations instead of input pixels."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    def hook_fn(module, inp, out):
        nonlocal mixed_repr
        if isinstance(out, torch.Tensor):
            mixed_repr = lam * out + (1 - lam) * out[index]

    mixed_repr = None
    layers = list(model.modules())
    target_layer = layers[layer_idx] if abs(layer_idx) < len(layers) else layers[-1]

    handle = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(x)
    handle.remove()

    if mixed_repr is None:
        return x, y, y[index], lam

    return mixed_repr, y, y[index], lam


# ── Data Augmentation Pipelines ──────────────────────────────────────────────


def get_train_augmentation(image_size: int = 224, config: Any = None) -> transforms.Compose:
    """Build training augmentation pipeline from config."""
    aug_list = [
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]

    if config and hasattr(config, "use_trivialaugment") and config.use_trivialaugment:
        aug_list.append(transforms.TrivialAugmentWide())
    elif config and hasattr(config, "use_randaugment") and config.use_randaugment:
        n = getattr(config, "RandAugment_n", 2)
        m = getattr(config, "RandAugment_m", 9)
        aug_list.append(transforms.RandAugment(num_ops=n, magnitude=m))
    else:
        color_jitter = getattr(config, "color_jitter", 0.3) if config else 0.3
        aug_list.extend(
            [
                transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=0.1,
                ),
                transforms.RandomAffine(
                    degrees=getattr(config, "random_rotation", 30) if config else 30,
                    translate=getattr(config, "random_affine_translate", (0.1, 0.1)) if config else (0.1, 0.1),
                ),
            ]
        )

    aug_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    erase_prob = getattr(config, "random_erasing_prob", 0.2) if config else 0.2
    if erase_prob > 0:
        aug_list.append(
            transforms.RandomErasing(
                p=erase_prob,
                scale=getattr(config, "random_erasing_scale", (0.02, 0.15)) if config else (0.02, 0.15),
            )
        )

    return transforms.Compose(aug_list)


def get_val_augmentation(image_size: int = 224) -> transforms.Compose:
    """Standard validation/test augmentation (deterministic)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_tta_augmentations(image_size: int = 224) -> list[transforms.Compose]:
    """Return multiple augmentation pipelines for test-time augmentation."""
    base = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    resize = transforms.Resize((image_size, image_size))

    return [
        transforms.Compose([resize, transforms.ToTensor(), base]),
        transforms.Compose([resize, transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), base]),
        transforms.Compose([resize, transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), base]),
        transforms.Compose([resize, transforms.RandomRotation(15), transforms.ToTensor(), base]),
        transforms.Compose([resize, transforms.ColorJitter(brightness=0.2), transforms.ToTensor(), base]),
        transforms.Compose(
            [resize, transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)), transforms.ToTensor(), base]
        ),
    ]


# ── Class-Balanced Sampling ──────────────────────────────────────────────────


def compute_class_weights_from_dataset(dataset: Dataset) -> list[float]:
    """Compute inverse-frequency class weights."""
    counter = Counter()
    for _, label in dataset:
        counter[label] += 1
    total = sum(counter.values())
    n_classes = max(counter.keys()) + 1
    weights = [total / (n_classes * max(counter.get(i, 1), 1)) for i in range(n_classes)]
    return weights


def compute_effective_num_samples(dataset: Dataset, beta: float = 0.999) -> list[float]:
    """Compute effective number of samples per class (Cui et al., CVPR 2019)."""
    counter = Counter()
    for _, label in dataset:
        counter[label] += 1
    n_classes = max(counter.keys()) + 1
    effective_num = [1.0 - beta ** counter.get(i, 0) for i in range(n_classes)]
    weights = [(1.0 - beta) / max(en, 1e-8) for en in effective_num]
    total_weight = sum(weights)
    return [w / total_weight * n_classes for w in weights]


def create_balanced_sampler(dataset: Dataset, beta: float = 0.999) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler using effective number of samples."""
    counter = Counter()
    for _, label in dataset:
        counter[label] += 1

    n_classes = max(counter.keys()) + 1
    effective_num = [1.0 - beta ** counter.get(i, 0) for i in range(n_classes)]
    weights = [(1.0 - beta) / max(en, 1e-8) for en in effective_num]

    sample_weights = [weights[label] for _, label in dataset]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


# ── MixUp Handler ────────────────────────────────────────────────────────────


class MixUpHandler:
    """Manages MixUp/CutMix/CutBlur strategy selection and application."""

    def __init__(self, config: Any = None):
        self.mixup_alpha = getattr(config, "mixup_alpha", 0.8) if config else 0.8
        self.cutmix_alpha = getattr(config, "cutmix_alpha", 1.0) if config else 1.0
        self.mixup_prob = getattr(config, "mixup_prob", 0.5) if config else 0.5
        self.label_smoothing = getattr(config, "label_smoothing", 0.1) if config else 0.1
        self.n_classes = 4

    def __call__(self, x: torch.Tensor, y: torch.Tensor, model: torch.nn.Module | None = None) -> tuple:
        """Apply random mixup strategy."""
        if random.random() > self.mixup_prob:
            return x, y, y, 1.0

        strategy = random.choice(["mixup", "cutmix", "cutblur"])
        if strategy == "mixup":
            return mixup_data(x, y, self.mixup_alpha)
        elif strategy == "cutmix":
            return cutmix_data(x, y, self.cutmix_alpha)
        else:
            return cutblur(x, y, self.cutmix_alpha)

    def smooth_labels(self, y: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing to one-hot targets."""
        if self.label_smoothing <= 0:
            return y
        one_hot = F.one_hot(y, self.n_classes).float()
        smooth = one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.n_classes
        return smooth
