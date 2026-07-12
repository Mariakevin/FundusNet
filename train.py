"""Train retina classification models — production-grade pipeline.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --overrides training.epochs=100 model.name=swin
    python train.py --models swin maxvit convnext_v2 efficientnet_v2 deit

Features:
- Structured config system (YAML/JSON/CLI)
- Distributed Data Parallel (multi-GPU)
- Automatic Mixed Precision (AMP)
- Gradient accumulation
- Exponential Moving Average (EMA)
- Cosine annealing with warm restarts + linear warmup
- Stratified K-Fold cross-validation
- MixUp / CutMix / CutBlur augmentation
- Class-balanced sampling
- Discriminative learning rates
- W&B / MLflow experiment tracking
- Model registry with versioning
- Early stopping with patience
"""

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from retina_app.config import ExperimentConfig, load_config
from retina_app.ml.trainer import Trainer, launch_distributed
from retina_app.utils import CATEGORIES, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train retina classification models")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON config file",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to train (overrides config)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset path (overrides config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=None,
        help="Number of CV folds (overrides config)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (overrides config)",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Config overrides as key=value pairs",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable exponential moving average",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow logging",
    )
    return parser.parse_args()


def build_config(args) -> ExperimentConfig:
    """Build experiment config from args and config file."""
    if args.config:
        config = load_config(args.config, args.overrides)
    else:
        config = ExperimentConfig()

    # CLI overrides
    if args.name:
        config.name = args.name
    if args.models:
        config.model.name = args.models[0]
    if args.dataset:
        config.data.dataset_path = args.dataset
    if args.output:
        config.output_dir = args.output
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.seed:
        config.training.seed = args.seed
    if args.folds:
        config.folds = args.folds
    if args.no_amp:
        config.training.use_amp = False
    if args.no_ema:
        config.training.use_ema = False
    if args.wandb:
        config.enable_wandb = True
    if args.mlflow:
        config.enable_mlflow = True

    return config


def generate_default_config():
    """Generate a default config YAML file."""
    config = ExperimentConfig()

    try:
        import yaml

        with open("config.yaml", "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
        print("Generated config.yaml with default settings")
    except ImportError:
        config.to_json("config.json")
        print("Generated config.json with default settings (install pyyaml for YAML)")


def main():
    args = parse_args()

    if not args.config and not args.models:
        print("No config or models specified. Generating default config...")
        generate_default_config()
        return

    config = build_config(args)
    setup_seed(config.training.seed)

    print(f"{'=' * 60}")
    print("FundusNet Training Pipeline")
    print(f"{'=' * 60}")
    print(f"Experiment: {config.name}")
    print(f"Dataset: {config.data.dataset_path}")
    print(f"Categories: {CATEGORIES}")
    print(f"Output: {config.output_dir}")
    print(f"Config hash: {config.hash()}")
    print()

    # Save config
    os.makedirs(config.output_dir, exist_ok=True)
    config.to_json(os.path.join(config.output_dir, "config.json"))

    start_time = time.time()

    if args.distributed:
        model_names = args.models or [config.model.name]
        results: dict = launch_distributed(config, model_names)  # type: ignore[assignment]
    else:
        model_names = args.models or [config.model.name]
        trainer = Trainer(config)
        results = trainer.train(model_names, config.data.dataset_path)

    duration = time.time() - start_time
    print(f"\nTotal training time: {duration:.1f}s ({duration / 60:.1f}m)")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Final Results")
    print(f"{'=' * 60}")
    for model_name, res in results.items():
        print(f"  {model_name}: {res['mean_acc']:.2f}% ± {res['std_acc']:.2f}%")
        print(f"    Total duration: {res['total_duration']:.1f}s")
        for fold in res["fold_results"]:
            print(
                f"    Fold {fold['fold']}: {fold['best_val_acc']:.2f}% "
                f"({fold['epochs_trained']} epochs, {fold['duration_seconds']:.1f}s)"
            )

    # Save final results
    import json

    results_path = os.path.join(config.output_dir, "results.json")
    serializable = {}
    for mn, res in results.items():
        serializable[mn] = {
            "mean_acc": res["mean_acc"],
            "std_acc": res["std_acc"],
            "total_duration": res["total_duration"],
            "fold_results": [{k: v for k, v in f.items() if k != "training_log"} for f in res["fold_results"]],
        }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
