"""Structured configuration system for FundusNet training experiments.

Supports YAML config files, CLI overrides, and experiment tracking
with automatic config diffing and versioning.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    dataset_path: str = "retina_dataset"
    image_size: int = 224
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = True


@dataclass
class AugmentationConfig:
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.5
    label_smoothing: float = 0.1
    random_erasing_prob: float = 0.2
    random_erasing_scale: tuple = (0.02, 0.15)
    color_jitter: float = 0.3
    random_rotation: int = 30
    random_affine_translate: tuple = (0.1, 0.1)
    RandAugment_n: int = 2
    RandAugment_m: int = 9
    use_randaugment: bool = False
    use_trivialaugment: bool = False


@dataclass
class ModelConfig:
    name: str = "efficientnet_b0"
    num_classes: int = 4
    pretrained: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1
    label_smoothing: float = 0.1
    ema_decay: float = 0.9999
    use_amp: bool = True
    use_ema: bool = True
    early_stopping_patience: int = 20
    seed: int = 42


@dataclass
class SchedulerConfig:
    name: str = "cosine_warm_restarts"
    T_0: int = 20
    T_mult: int = 2
    T_max: int = 200
    eta_min: float = 1e-6
    warmup_start_factor: float = 0.01
    step_size: int = 30
    gamma: float = 0.1
    plateau_patience: int = 10
    plateau_factor: float = 0.1


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9
    nesterov: bool = True
    use_trac: bool = False


@dataclass
class DistributedConfig:
    backend: str = "nccl"
    find_unused_parameters: bool = False
    sync_bn: bool = True


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "default_experiment"
    description: str = ""
    tags: list = field(default_factory=list)
    output_dir: str = "experiments"
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    folds: int = 5
    enable_wandb: bool = False
    enable_mlflow: bool = False
    wandb_project: str = "fundusnet"
    mlflow_experiment: str = "fundusnet"
    log_interval: int = 10
    val_interval: int = 1

    def hash(self) -> str:
        """Generate a deterministic hash of the config for deduplication."""
        config_str = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """Recursively construct config from a nested dict."""
        cfg = cls()
        for key, value in d.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if isinstance(value, dict) and hasattr(attr, "__dataclass_fields__"):
                    nested = type(attr)(**{k: v for k, v in value.items() if k in fields(attr)})
                    setattr(cfg, key, nested)
                else:
                    setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load config from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required: pip install pyyaml")
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    def diff(self, other: ExperimentConfig) -> dict[str, Any]:
        """Compute the diff between two configs."""
        a = asdict(self)
        b = asdict(other)
        diffs = {}
        _diff_dicts(a, b, diffs, "")
        return diffs

    def save_experiment(self, metrics: dict[str, Any] | None = None) -> str:
        """Save config + optional metrics to experiment directory."""
        exp_dir = Path(self.output_dir) / self.name / self.hash()
        exp_dir.mkdir(parents=True, exist_ok=True)

        self.to_json(exp_dir / "config.json")

        meta = {
            "name": self.name,
            "hash": self.hash(),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tags": self.tags,
        }
        if metrics:
            meta["metrics"] = metrics

        with open(exp_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        return str(exp_dir)


def _diff_dicts(a: dict, b: dict, out: dict, prefix: str) -> None:
    all_keys = set(a.keys()) | set(b.keys())
    for key in sorted(all_keys):
        full_key = f"{prefix}.{key}" if prefix else key
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            _diff_dicts(va, vb, out, full_key)
        elif va != vb:
            out[full_key] = {"old": va, "new": vb}


def load_config(path: str | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load config from file with optional CLI-style overrides.

    Args:
        path: Path to YAML/JSON config file
        overrides: List of "key.subkey=value" strings
    """
    if path and path.endswith(".yaml"):
        cfg = ExperimentConfig.from_yaml(path)
    elif path and path.endswith(".json"):
        cfg = ExperimentConfig.from_json(path)
    else:
        cfg = ExperimentConfig()

    if overrides:
        for override in overrides:
            key, value = override.split("=", 1)
            _set_nested(cfg, key.strip(), _parse_value(value.strip()))

    return cfg


def _set_nested(obj: Any, key: str, value: Any) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _parse_value(s: str) -> Any:
    if s.lower() in ("true", "yes", "1"):
        return True
    if s.lower() in ("false", "no", "0"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s
