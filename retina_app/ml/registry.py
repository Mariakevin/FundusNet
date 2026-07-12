"""Model registry with versioning, artifact management, and experiment comparison.

Provides:
- Centralized model registration and versioning
- Artifact storage (checkpoints, configs, metrics)
- Experiment comparison and leaderboard
- Model promotion (staging → production)
- Automatic model cleanup and pruning
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelArtifact:
    """Represents a single model artifact (checkpoint + metadata)."""

    model_name: str
    version: str
    stage: str = "development"  # development, staging, production, archived
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    checkpoint_path: str = ""
    config_path: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    parent_version: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Result of a training experiment."""

    experiment_name: str
    model_name: str
    config_hash: str
    fold_results: list[dict[str, Any]] = field(default_factory=list)
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def primary_metric(self) -> float:
        return self.mean_metrics.get("val_acc", 0.0)


class ModelRegistry:
    """Central model registry with versioning and artifact management."""

    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts: dict[str, list[ModelArtifact]] = {}
        self._load_registry()

    def _registry_file(self) -> Path:
        return self.registry_dir / "registry.json"

    def _load_registry(self):
        reg_file = self._registry_file()
        if reg_file.exists():
            with open(reg_file) as f:
                data = json.load(f)
            for model_name, artifacts in data.items():
                self._artifacts[model_name] = [ModelArtifact(**a) for a in artifacts]

    def _save_registry(self):
        data = {model_name: [a.to_dict() for a in artifacts] for model_name, artifacts in self._artifacts.items()}
        with open(self._registry_file(), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def register(
        self,
        model_name: str,
        checkpoint_path: str,
        metrics: dict[str, Any] | None = None,
        config_path: str = "",
        tags: list[str] | None = None,
        notes: str = "",
    ) -> ModelArtifact:
        """Register a new model version."""
        version = self._next_version(model_name)
        artifact = ModelArtifact(
            model_name=model_name,
            version=version,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            metrics=metrics or {},
            tags=tags or [],
            notes=notes,
        )

        # Store checkpoint in registry
        dest_dir = self.registry_dir / model_name / version
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / "model.pth"
        if checkpoint_path and os.path.exists(checkpoint_path):
            shutil.copy2(checkpoint_path, dest_path)
            artifact.checkpoint_path = str(dest_path)

        if config_path and os.path.exists(config_path):
            shutil.copy2(config_path, dest_dir / "config.json")
            artifact.config_path = str(dest_dir / "config.json")

        # Save metadata
        with open(dest_dir / "artifact.json", "w") as f:
            json.dump(artifact.to_dict(), f, indent=2, default=str)

        if model_name not in self._artifacts:
            self._artifacts[model_name] = []
        self._artifacts[model_name].append(artifact)
        self._save_registry()

        return artifact

    def _next_version(self, model_name: str) -> str:
        artifacts = self._artifacts.get(model_name, [])
        if not artifacts:
            return "v1.0.0"
        versions = []
        for a in artifacts:
            try:
                v = a.version.lstrip("v").split(".")
                versions.append(tuple(int(x) for x in v))
            except (ValueError, IndexError):
                continue
        if versions:
            latest = max(versions)
            return f"v{latest[0]}.{latest[1]}.{latest[2] + 1}"
        return "v1.0.0"

    def get_latest(self, model_name: str, stage: str = "production") -> ModelArtifact | None:
        """Get the latest model version for a given stage."""
        artifacts = self._artifacts.get(model_name, [])
        staged = [a for a in artifacts if a.stage == stage]
        if staged:
            return max(staged, key=lambda a: a.created_at)
        return artifacts[-1] if artifacts else None

    def promote(self, model_name: str, version: str, target_stage: str) -> ModelArtifact | None:
        """Promote a model to a different stage."""
        for artifact in self._artifacts.get(model_name, []):
            if artifact.version == version:
                artifact.stage = target_stage
                self._save_registry()
                return artifact
        return None

    def list_models(self, model_name: str | None = None) -> list[ModelArtifact]:
        """List all registered models, optionally filtered by name."""
        if model_name:
            return self._artifacts.get(model_name, [])
        return [a for artifacts in self._artifacts.values() for a in artifacts]

    def compare(self, versions: list[str], model_name: str) -> dict[str, Any]:
        """Compare metrics across model versions."""
        results = {}
        for artifact in self._artifacts.get(model_name, []):
            if artifact.version in versions:
                results[artifact.version] = artifact.metrics
        return results

    def delete_version(self, model_name: str, version: str) -> bool:
        """Delete a model version and its artifacts."""
        artifacts = self._artifacts.get(model_name, [])
        target = None
        for a in artifacts:
            if a.version == version:
                target = a
                break
        if target is None:
            return False

        # Remove files
        artifact_dir = self.registry_dir / model_name / version
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)

        artifacts.remove(target)
        self._save_registry()
        return True

    def cleanup(self, keep_latest_n: int = 3) -> int:
        """Remove old versions, keeping only the latest N per model."""
        removed = 0
        for model_name, artifacts in self._artifacts.items():
            if len(artifacts) <= keep_latest_n:
                continue
            sorted_arts = sorted(artifacts, key=lambda a: a.created_at)
            to_remove = sorted_arts[:-keep_latest_n]
            for artifact in to_remove:
                if artifact.stage == "production":
                    continue
                self.delete_version(model_name, artifact.version)
                removed += 1
        return removed

    def leaderboard(self, model_name: str | None = None) -> list[dict]:
        """Get a sorted leaderboard of all experiments."""
        entries = []
        for artifact in self.list_models(model_name):
            entries.append(
                {
                    "model": artifact.model_name,
                    "version": artifact.version,
                    "stage": artifact.stage,
                    "val_acc": artifact.metrics.get("val_acc", 0),
                    "val_loss": artifact.metrics.get("val_loss", float("inf")),
                    "created": artifact.created_at,
                }
            )
        return sorted(entries, key=lambda x: x["val_acc"], reverse=True)


class ExperimentTracker:
    """Track and compare training experiments."""

    def __init__(self, tracker_dir: str = "experiments"):
        self.tracker_dir = Path(tracker_dir)
        self.tracker_dir.mkdir(parents=True, exist_ok=True)

    def log_experiment(
        self,
        name: str,
        config: dict[str, Any],
        fold_results: list[dict[str, Any]],
        duration: float = 0.0,
    ) -> ExperimentResult:
        """Log a complete experiment result."""
        # Compute aggregate metrics
        all_metrics = {}
        for fold in fold_results:
            for k, v in fold.items():
                if isinstance(v, (int, float)):
                    all_metrics.setdefault(k, []).append(v)

        mean_metrics = {k: float(sum(v) / len(v)) for k, v in all_metrics.items()}
        std_metrics = {
            k: float((sum((x - mean_metrics[k]) ** 2 for x in v) / len(v)) ** 0.5) for k, v in all_metrics.items()
        }

        result = ExperimentResult(
            experiment_name=name,
            model_name=config.get("model", {}).get("name", "unknown"),
            config_hash=config.get("hash", ""),
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            duration_seconds=duration,
        )

        # Save experiment
        exp_dir = self.tracker_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_dir / "result.json", "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        return result

    def list_experiments(self) -> list[ExperimentResult]:
        """List all tracked experiments."""
        results = []
        for exp_dir in self.tracker_dir.iterdir():
            result_file = exp_dir / "result.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                results.append(ExperimentResult(**data))
        return sorted(results, key=lambda r: r.primary_metric, reverse=True)

    def compare_experiments(self, names: list[str]) -> dict[str, Any]:
        """Side-by-side comparison of experiments."""
        comparison = {}
        for name in names:
            exp_dir = self.tracker_dir / name
            if (exp_dir / "result.json").exists():
                with open(exp_dir / "result.json") as f:
                    comparison[name] = json.load(f)
        return comparison
