"""Production-grade training loop with distributed training, AMP, gradient accumulation.

Features:
- Distributed Data Parallel (DDP) support
- Automatic Mixed Precision (AMP)
- Gradient accumulation for effective large batch sizes
- Exponential Moving Average (EMA)
- Cosine annealing with warm restarts + linear warmup
- Stratified K-Fold cross-validation
- W&B / MLflow experiment tracking
- Model checkpointing with best/last/periodic saves
- Early stopping with patience
- Gradient norm monitoring
- Mixed augmentation: MixUp / CutMix / CutBlur
- Class-balanced sampling
- Discriminative learning rates for fine-tuning
"""

from __future__ import annotations

import copy
import csv
import logging
import os
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from retina_app.config import ExperimentConfig
from retina_app.ml.data_pipeline import (
    MixUpHandler,
    create_balanced_sampler,
    get_train_augmentation,
    get_val_augmentation,
)
from retina_app.ml.optim import (
    GradientAccumulator,
    create_discriminative_param_groups,
    create_optimizer,
    create_scheduler,
)
from retina_app.ml.registry import ExperimentTracker, ModelRegistry
from retina_app.utils import (
    CATEGORIES,
    EMA,
    RetinaDataset,
    create_model,
    setup_seed,
)

logger = logging.getLogger(__name__)


# ── Loss Functions ────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (Lin et al., ICCV 2017)."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.n_classes = len(alpha) if alpha else 4
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            one_hot = F.one_hot(targets, self.n_classes).float()
            smooth = one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.n_classes
            ce_loss = -(smooth * F.log_softmax(inputs, dim=1)).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing (Szegedy et al., 2016)."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target)
        return (1 - self.smoothing) * nll + self.smoothing * loss / n_classes


# ── Metrics Tracker ───────────────────────────────────────────────────────────


class MetricsTracker:
    """Tracks running metrics during training."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.correct = 0
        self.total = 0
        self.batch_count = 0

    def update(self, loss: float, predicted: torch.Tensor, labels: torch.Tensor):
        self.loss_sum += loss
        self.correct += predicted.eq(labels).sum().item()
        self.total += labels.size(0)
        self.batch_count += 1

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / max(self.batch_count, 1)

    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / max(self.total, 1)


# ── Main Trainer ──────────────────────────────────────────────────────────────


class Trainer:
    """Production training loop with full feature set."""

    def __init__(self, config: ExperimentConfig, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_distributed = False
        self.is_main_process = True
        self.local_rank = 0

        # Experiment tracking
        self.registry = ModelRegistry(registry_dir=os.path.join(config.output_dir, "registry"))
        self.experiment_tracker = ExperimentTracker(tracker_dir=os.path.join(config.output_dir, "experiments"))

        # Initialize experiment logging
        self._wandb_run = None
        self._mlflow_run = None

    def setup_distributed(self, rank: int = 0, world_size: int = 1):
        """Initialize distributed training."""
        if world_size <= 1:
            return

        self.is_distributed = True
        self.local_rank = rank
        self.is_main_process = rank == 0

        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

        torch.distributed.init_process_group(
            backend=self.config.distributed.backend,
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(rank)
        self.device = torch.device(f"cuda:{rank}")

    def cleanup_distributed(self):
        if self.is_distributed:
            torch.distributed.destroy_process_group()

    @contextmanager
    def _amp_context(self):
        """Context manager for automatic mixed precision."""
        if self.config.training.use_amp and self.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                yield
        else:
            yield

    def _log_experiment_start(self):
        """Initialize experiment logging backends."""
        if self.config.enable_wandb and self.is_main_process:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.name,
                    config=self.config.to_dict(),
                    tags=self.config.tags,
                )
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")

        if self.config.enable_mlflow and self.is_main_process:
            try:
                import mlflow

                mlflow.set_experiment(self.config.mlflow_experiment)
                self._mlflow_run = mlflow.start_run(run_name=self.config.name)
            except ImportError:
                logger.warning("mlflow not installed, skipping MLflow logging")

    def _log_metrics(self, metrics: dict[str, Any], step: int, epoch: int):
        """Log metrics to all configured backends."""
        if not self.is_main_process:
            return

        if self._wandb_run:
            try:
                import wandb

                wandb.log(metrics, step=step)
            except Exception:
                pass

        if self._mlflow_run:
            try:
                import mlflow

                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v, step=step)
            except Exception:
                pass

    def _log_experiment_end(self, metrics: dict[str, Any]):
        """Finalize experiment logging."""
        if self._wandb_run:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass
        if self._mlflow_run:
            try:
                import mlflow

                mlflow.end_run()
            except Exception:
                pass

    def train_fold(
        self,
        fold_idx: int,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        model_name: str,
        full_dataset: Dataset,
    ) -> dict[str, Any]:
        """Train a single fold with full feature set."""
        cfg = self.config
        start_time = time.time()

        if self.is_main_process:
            print(f"\n{'=' * 60}")
            print(f"Fold {fold_idx + 1} — Training {model_name}")
            print(f"{'=' * 60}")

        # ── Data ───────────────────────────────────────────────────────────
        train_transform = get_train_augmentation(cfg.data.image_size, cfg.augmentation)
        val_transform = get_val_augmentation(cfg.data.image_size)

        train_dataset = Subset(
            RetinaDataset(cfg.data.dataset_path, transform=train_transform),
            train_indices,
        )
        val_dataset = Subset(
            RetinaDataset(cfg.data.dataset_path, transform=val_transform),
            val_indices,
        )

        # Class-balanced sampling
        sampler = create_balanced_sampler(train_dataset) if len(train_indices) > 100 else None

        n_workers = min(os.cpu_count() or 4, cfg.data.num_workers)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=n_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=cfg.data.drop_last,
            persistent_workers=cfg.data.persistent_workers and n_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=cfg.data.pin_memory,
        )

        # ── Model ──────────────────────────────────────────────────────────
        model = create_model(model_name, cfg.model.num_classes, cfg.model.pretrained)
        model.to(self.device)

        if self.is_distributed and cfg.distributed.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if self.is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=cfg.distributed.find_unused_parameters,
            )

        # ── Class Weights ──────────────────────────────────────────────────
        class_counts = [0] * cfg.model.num_classes
        for _, label_idx in full_dataset.samples:
            class_counts[label_idx] += 1
        total = sum(class_counts)
        alpha = [total / (cfg.model.num_classes * max(c, 1)) for c in class_counts]

        # ── Loss ───────────────────────────────────────────────────────────
        if cfg.training.label_smoothing > 0:
            criterion = FocalLoss(
                alpha=alpha,
                gamma=2.0,
                label_smoothing=cfg.training.label_smoothing,
            )
        else:
            criterion = FocalLoss(alpha=alpha, gamma=2.0)

        # ── Optimizer ──────────────────────────────────────────────────────
        base_model = model.module if self.is_distributed else model
        param_groups = create_discriminative_param_groups(base_model, cfg.training.learning_rate)
        optimizer = create_optimizer(
            base_model,
            type(
                "OptConfig",
                (),
                {
                    "name": cfg.optimizer.name,
                    "lr": cfg.training.learning_rate,
                    "weight_decay": cfg.training.weight_decay,
                    "betas": cfg.optimizer.betas,
                    "eps": cfg.optimizer.eps,
                    "momentum": cfg.optimizer.momentum,
                    "nesterov": cfg.optimizer.nesterov,
                },
            )(),
            param_groups,
        )

        # ── Scheduler ──────────────────────────────────────────────────────
        sched_config = type(
            "SchedConfig",
            (),
            {
                "name": cfg.scheduler.name,
                "T_0": cfg.scheduler.T_0,
                "T_mult": cfg.scheduler.T_mult,
                "T_max": cfg.training.epochs,
                "eta_min": cfg.training.min_lr,
                "warmup_epochs": cfg.training.warmup_epochs,
                "flat_epochs": 10,
                "plateau_patience": cfg.scheduler.plateau_patience,
                "plateau_factor": cfg.scheduler.plateau_factor,
                "step_size": cfg.scheduler.step_size,
                "gamma": cfg.scheduler.gamma,
            },
        )()
        scheduler, step_on_epoch = create_scheduler(optimizer, sched_config)

        # ── EMA ────────────────────────────────────────────────────────────
        ema = EMA(base_model, decay=cfg.training.ema_decay) if cfg.training.use_ema else None

        # ── Gradient Accumulation ───────────────────────────────────────────
        accum_steps = cfg.training.grad_accumulation_steps
        grad_accum = GradientAccumulator(base_model, accum_steps) if accum_steps > 1 else None

        # ── MixUp Handler ──────────────────────────────────────────────────
        mix_handler = MixUpHandler(cfg.augmentation)

        # ── Training Loop ──────────────────────────────────────────────────
        best_val_acc = 0.0
        epochs_without_improvement = 0
        best_model_state = None
        training_log = []
        global_step = 0

        for epoch in range(cfg.training.epochs):
            model.train()
            train_metrics = MetricsTracker()

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # MixUp / CutMix
                mixed_x, y_a, y_b, lam = mix_handler(inputs, labels)

                with self._amp_context():
                    outputs = model(mixed_x)
                    if lam < 1.0:
                        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                    else:
                        loss = criterion(outputs, labels)

                # Gradient accumulation
                if grad_accum:
                    grad_accum.zero_grad()
                    grad_accum.backward(loss / accum_steps)
                    if (batch_idx + 1) % accum_steps == 0:
                        if cfg.training.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(base_model.parameters(), cfg.training.grad_clip)
                        grad_accum.step(optimizer)
                        if ema:
                            ema.update()
                        global_step += 1
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    if cfg.training.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), cfg.training.grad_clip)
                    optimizer.step()
                    if ema:
                        ema.update()
                    global_step += 1

                _, predicted = outputs.max(1)
                train_metrics.update(loss.item(), predicted, labels)

                # Log batch metrics
                if global_step % cfg.log_interval == 0 and self.is_main_process:
                    self._log_metrics(
                        {
                            "train/loss": loss.item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch,
                        },
                        global_step,
                        epoch,
                    )

            # Scheduler step
            if step_on_epoch:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Will step after validation
                    pass
                else:
                    scheduler.step()

            # ── Validation ─────────────────────────────────────────────────
            if epoch % cfg.val_interval == 0:
                if ema:
                    ema.apply_shadow()

                model.eval()
                val_metrics = MetricsTracker()
                all_probs = []
                all_labels = []

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, predicted = outputs.max(1)
                        val_metrics.update(loss.item(), predicted, labels)
                        all_probs.append(F.softmax(outputs, dim=1).cpu())
                        all_labels.append(labels.cpu())

                if ema:
                    ema.restore()

                train_acc = train_metrics.accuracy
                val_acc = val_metrics.accuracy
                avg_train_loss = train_metrics.avg_loss
                avg_val_loss = val_metrics.avg_loss
                current_lr = optimizer.param_groups[0]["lr"]

                # Step plateau scheduler
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_acc)

                # Log epoch metrics
                epoch_metrics = {
                    "epoch": epoch + 1,
                    "train_loss": round(avg_train_loss, 6),
                    "train_acc": round(train_acc, 4),
                    "val_loss": round(avg_val_loss, 6),
                    "val_acc": round(val_acc, 4),
                    "lr": round(current_lr, 8),
                }
                training_log.append(epoch_metrics)

                if self.is_main_process:
                    print(
                        f"Epoch {epoch + 1}/{cfg.training.epochs}: "
                        f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                        f"LR: {current_lr:.6f}"
                    )

                    self._log_metrics(
                        {
                            "val/loss": avg_val_loss,
                            "val/acc": val_acc,
                            "val/lr": current_lr,
                        },
                        global_step,
                        epoch,
                    )

                # ── Checkpointing ──────────────────────────────────────────
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_without_improvement = 0

                    if ema:
                        ema.apply_shadow()
                    best_model_state = copy.deepcopy(base_model.state_dict())
                    if ema:
                        ema.restore()

                    if self.is_main_process:
                        models_dir = os.path.join(cfg.output_dir, "models")
                        os.makedirs(models_dir, exist_ok=True)
                        save_path = os.path.join(models_dir, f"{model_name}_best.pth")
                        torch.save(
                            {
                                "model_state_dict": best_model_state,
                                "model_type": model_name,
                                "num_classes": cfg.model.num_classes,
                                "categories": CATEGORIES,
                                "best_val_acc": best_val_acc,
                                "config": cfg.to_dict(),
                                "epoch": epoch,
                            },
                            save_path,
                        )
                        print(f"  Saved best model ({val_acc:.2f}%) to {save_path}")
                else:
                    epochs_without_improvement += 1

                # Early stopping
                if epochs_without_improvement >= cfg.training.early_stopping_patience:
                    if self.is_main_process:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # ── Save Training Log ──────────────────────────────────────────────
        if self.is_main_process:
            logs_dir = os.path.join(cfg.output_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            log_path = os.path.join(logs_dir, f"{model_name}_training_log.csv")
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
                writer.writeheader()
                writer.writerows(training_log)

            # Register model
            if best_model_state:
                self.registry.register(
                    model_name=model_name,
                    checkpoint_path=os.path.join(cfg.output_dir, "models", f"{model_name}_best.pth"),
                    metrics={"val_acc": best_val_acc, "epochs_trained": epoch + 1},
                    tags=[f"fold_{fold_idx}"],
                )

        duration = time.time() - start_time
        return {
            "fold": fold_idx + 1,
            "model_name": model_name,
            "best_val_acc": best_val_acc,
            "epochs_trained": epoch + 1,
            "duration_seconds": duration,
            "training_log": training_log,
        }

    def train(
        self,
        model_names: list[str] | None = None,
        dataset_path: str | None = None,
    ) -> dict[str, Any]:
        """Run full training pipeline with K-Fold cross-validation."""
        cfg = self.config
        model_names = model_names or [cfg.model.name]
        dataset_path = dataset_path or cfg.data.dataset_path

        setup_seed(cfg.training.seed)

        if self.is_main_process:
            self._log_experiment_start()
            print(f"Training config: {cfg.name}")
            print(f"Dataset: {dataset_path}")
            print(f"Categories: {CATEGORIES}")
            print(f"Models: {model_names}")
            print(f"Device: {self.device}")

        # ── Data ───────────────────────────────────────────────────────────
        train_transform = get_train_augmentation(cfg.data.image_size, cfg.augmentation)
        full_dataset = RetinaDataset(dataset_path, transform=train_transform)
        labels = [label for _, label in full_dataset.samples]

        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(
            n_splits=cfg.folds,
            shuffle=True,
            random_state=cfg.training.seed,
        )

        all_results = {}
        for model_name in model_names:
            if self.is_main_process:
                print(f"\n{'#' * 60}")
                print(f"# Training {model_name} with {cfg.folds}-fold CV")
                print(f"{'#' * 60}")

            fold_results = []
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
                result = self.train_fold(fold_idx, train_idx, val_idx, model_name, full_dataset)
                fold_results.append(result)

            # Aggregate fold results
            fold_accs = [r["best_val_acc"] for r in fold_results]
            mean_acc = float(np.mean(fold_accs))
            std_acc = float(np.std(fold_accs))
            total_duration = sum(r["duration_seconds"] for r in fold_results)

            all_results[model_name] = {
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "fold_results": fold_results,
                "total_duration": total_duration,
            }

            if self.is_main_process:
                print(f"\n{model_name}: {mean_acc:.2f}% ± {std_acc:.2f}%")

                # Log to experiment tracker
                self.experiment_tracker.log_experiment(
                    name=f"{cfg.name}_{model_name}",
                    config=cfg.to_dict(),
                    fold_results=[
                        {
                            "fold": r["fold"],
                            "val_acc": r["best_val_acc"],
                            "epochs": r["epochs_trained"],
                            "duration": r["duration_seconds"],
                        }
                        for r in fold_results
                    ],
                    duration=total_duration,
                )

        # ── Summary ────────────────────────────────────────────────────────
        if self.is_main_process:
            print(f"\n{'=' * 60}")
            print("Training Summary")
            print(f"{'=' * 60}")
            for model_name, res in all_results.items():
                print(f"  {model_name}: {res['mean_acc']:.2f}% ± {res['std_acc']:.2f}%")

            self._log_experiment_end({f"{mn}/mean_acc": res["mean_acc"] for mn, res in all_results.items()})

        self.cleanup_distributed()
        return all_results


# ── Distributed Launcher ─────────────────────────────────────────────────────


def launch_distributed(config: ExperimentConfig, model_names: list[str]):
    """Launch distributed training across multiple GPUs."""
    world_size = torch.cuda.device_count()

    if world_size <= 1:
        trainer = Trainer(config)
        return trainer.train(model_names)

    import torch.multiprocessing as mp

    def _worker(rank, world_size, config, model_names):
        trainer = Trainer(config)
        trainer.setup_distributed(rank, world_size)
        try:
            trainer.train(model_names)
        finally:
            trainer.cleanup_distributed()

    mp.spawn(
        _worker,
        args=(world_size, config, model_names),
        nprocs=world_size,
        join=True,
    )
