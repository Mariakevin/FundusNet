"""Training logger for experiment tracking.

Provides structured logging of training runs including hyperparameters,
per-epoch metrics, and configuration snapshots. Outputs are saved as
JSON and CSV for downstream analysis (figures, tables, reproducibility).

Usage:
    from retina_app.services.training_logger import TrainingLogger

    logger = TrainingLogger(output_dir="logs", run_name="efficientnet_v1")
    logger.log_config({"model": "efficientnet", "lr": 0.001, "epochs": 15})
    for epoch in range(15):
        logger.log_epoch(epoch, train_loss=0.5, val_loss=0.4, train_acc=85.0, val_acc=88.0)
    logger.finalize()
"""

import os
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrainingLogger:
    """Structured training run logger.

    Saves:
        - {run_name}_config.json — hyperparameters and metadata
        - {run_name}_metrics.csv — per-epoch train/val metrics
        - {run_name}_summary.json — final summary with best metrics
    """

    def __init__(self, output_dir: str = "logs", run_name: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
        self.run_name = run_name

        self.config: Dict[str, Any] = {}
        self.epochs: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.best_val_acc = 0.0
        self.best_epoch = -1

        self._config_path = self.output_dir / f"{run_name}_config.json"
        self._csv_path = self.output_dir / f"{run_name}_metrics.csv"
        self._summary_path = self.output_dir / f"{run_name}_summary.json"

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log training configuration / hyperparameters.

        Args:
            config: dict of hyperparameters (model, lr, batch_size, epochs, etc.)
        """
        self.config = config
        self.config["run_name"] = self.run_name
        self.config["timestamp"] = datetime.now().isoformat()
        with open(self._config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

    def log_epoch(self, epoch: int, **metrics) -> None:
        """Log metrics for a single epoch.

        Args:
            epoch: epoch number (0-indexed)
            **metrics: keyword arguments for metrics (train_loss, val_loss, train_acc, val_acc, lr, etc.)
        """
        record = {"epoch": epoch + 1}
        record.update(metrics)
        self.epochs.append(record)

        val_acc = metrics.get("val_acc", 0.0)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch + 1

        self._write_csv()

    def _write_csv(self) -> None:
        """Append new epoch records to CSV."""
        if not self.epochs:
            return
        fieldnames = list(self.epochs[0].keys())
        write_header = not self._csv_path.exists()
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(self.epochs[-1])

    def finalize(self) -> Dict[str, Any]:
        """Write final summary and return it.

        Returns:
            dict with summary statistics
        """
        elapsed = time.time() - self.start_time

        train_losses = [e.get("train_loss", 0) for e in self.epochs]
        val_losses = [e.get("val_loss", 0) for e in self.epochs]
        train_accs = [e.get("train_acc", 0) for e in self.epochs]
        val_accs = [e.get("val_acc", 0) for e in self.epochs]

        summary = {
            "run_name": self.run_name,
            "config": self.config,
            "total_epochs": len(self.epochs),
            "elapsed_seconds": round(elapsed, 2),
            "best_val_acc": round(self.best_val_acc, 4),
            "best_epoch": self.best_epoch,
            "final_train_loss": round(train_losses[-1], 6) if train_losses else None,
            "final_val_loss": round(val_losses[-1], 6) if val_losses else None,
            "final_train_acc": round(train_accs[-1], 4) if train_accs else None,
            "final_val_acc": round(val_accs[-1], 4) if val_accs else None,
            "mean_train_loss": round(float(sum(train_losses) / len(train_losses)), 6) if train_losses else None,
            "mean_val_loss": round(float(sum(val_losses) / len(val_losses)), 6) if val_losses else None,
            "converged": self.best_epoch < len(self.epochs) - 3 if self.epochs else False,
        }

        with open(self._summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def get_training_curves(self):
        """Return training curves data for plotting.

        Returns:
            tuple of (train_losses, val_losses, train_accs, val_accs)
        """
        train_losses = [e.get("train_loss", 0) for e in self.epochs]
        val_losses = [e.get("val_loss", 0) for e in self.epochs]
        train_accs = [e.get("train_acc", 0) for e in self.epochs]
        val_accs = [e.get("val_acc", 0) for e in self.epochs]
        return train_losses, val_losses, train_accs, val_accs
