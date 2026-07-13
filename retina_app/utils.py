"""Shared utilities for retina training and distillation scripts."""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from retina_app.constants import CATEGORIES
except ImportError:
    CATEGORIES = ["Healthy", "Cataract", "Glaucoma", "Retina Disease"]

CLASS_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}


def setup_seed(seed):
    """Set random seeds for reproducibility across all backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


MODEL_NAME_MAP = {
    "swin": "swin_tiny_patch4_window7_224.ms_in22k",
    "maxvit": "maxvit_base_224.sw_in1k",
    "convnext_v2": "convnextv2_base.fcmae_ft_in1k",
    "efficientnet_v2": "efficientnet_v2_m.orig_in21k_ft_in1k",
    "deit": "deit3_base_patch16_224.fb_in22k_ft_in1k",
}


def create_model(model_name, num_classes, pretrained=True):
    """Create model using timm for all architectures."""
    try:
        import timm

        timm_name = MODEL_NAME_MAP.get(model_name, model_name)
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    except ImportError:
        raise ImportError(f"timm is required to create {model_name} models")
    return model


class EMA:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class RetinaDataset(Dataset):
    """Dataset for retina fundus images organized in class subdirectories."""

    def __init__(self, root_dir, transform=None, image_size=224):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.samples = []
        folder_mapping = {
            "1_normal": "Healthy",
            "2_cataract": "Cataract",
            "3_glaucoma": "Glaucoma",
            "4_Diabetic_Retinopathy": "Retina Disease",
            "Healthy": "Healthy",
            "Cataract": "Cataract",
            "Glaucoma": "Glaucoma",
            "Retina Disease": "Retina Disease",
            "normal": "Healthy",
            "cataract": "Cataract",
            "glaucoma": "Glaucoma",
            "diabetic_retinopathy": "Retina Disease",
        }
        seen_folders = set()
        for folder_name, class_name in folder_mapping.items():
            folder_path = os.path.normpath(os.path.join(root_dir, folder_name))
            # Use normcase for case-insensitive deduplication (Windows)
            abs_path = os.path.normcase(os.path.abspath(folder_path))
            if abs_path in seen_folders:
                continue
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                seen_folders.add(abs_path)
                class_idx = CLASS_TO_IDX[class_name]
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")):
                        self.samples.append((os.path.join(folder_path, img_name), class_idx))
        print(f"Loaded {len(self.samples)} images")
        for cat in CATEGORIES:
            count = sum(1 for _, idx in self.samples if idx == CLASS_TO_IDX[cat])
            print(f"  {cat}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return image, label
