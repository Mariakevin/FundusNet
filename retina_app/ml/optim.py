"""Advanced optimizers, schedulers, and learning rate strategies.

Implements:
- Custom optimizers: Lookahead, AdaBelief, LAMB
- Schedulers: CosineAnnealingWarmRestarts, OneCycleLR, FlatCosine, LinearWarmupCosine
- Gradient accumulation wrapper
- Mixed precision training helper
- Learning rate finder
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    OneCycleLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)

# ── Custom Optimizers ────────────────────────────────────────────────────────


class Lookahead(Optimizer):
    """Lookahead optimizer wrapper (Zhang et al., 2019).

    Wraps any base optimizer and maintains slow weights that are
    updated toward the fast weights every k steps.
    """

    def __init__(self, base_optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        self.slow_weights = [p.clone().detach() for group in self.param_groups for p in group["params"]]
        self._step_count = 0

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self._step_count += 1

        if self._step_count % self.k == 0:
            for group_idx, group in enumerate(self.param_groups):
                for param_idx, p in enumerate(group["params"]):
                    if p.grad is None:
                        continue
                    slow_idx = sum(len(g["params"]) for g in self.param_groups[:group_idx]) + param_idx
                    self.slow_weights[slow_idx].add_(self.alpha * (p.data - self.slow_weights[slow_idx]))
                    p.data.copy_(self.slow_weights[slow_idx])

        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict):
        self.base_optimizer.add_param_group(param_group)
        self.slow_weights.extend(p.clone().detach() for p in param_group["params"])

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.base_optimizer.param_groups = value


class AdaBelief(Optimizer):
    """AdaBelief optimizer (Zhuang et al., NeurIPS 2020).

    Adapts step size based on the "belief" in the gradient direction,
    using the exponential moving average of the gradient's deviation.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_deviation = grad - exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(grad_deviation, grad_deviation, value=1 - beta2)

                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ── Learning Rate Schedulers ─────────────────────────────────────────────────


def create_scheduler(
    optimizer: Optimizer,
    config: Any,
    steps_per_epoch: int | None = None,
) -> tuple:
    """Create LR scheduler from config, returning (scheduler, use_step_every_epoch).

    Returns:
        (scheduler, step_on_epoch): tuple of scheduler and whether to call .step() once per epoch
    """
    name = getattr(config, "name", "cosine_warm_restarts")
    warmup_epochs = getattr(config, "warmup_start_factor", 0.01)
    total_epochs = getattr(config, "T_max", 200)

    if name == "cosine_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(config, "T_0", 20),
            T_mult=getattr(config, "T_mult", 2),
            eta_min=getattr(config, "eta_min", 1e-6),
        )
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=getattr(config, "warmup_epochs", 10),
        )
        combined = SequentialLR(
            optimizer,
            [warmup, scheduler],
            milestones=[getattr(config, "warmup_epochs", 10)],
        )
        return combined, True

    elif name == "one_cycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        return OneCycleLR(
            optimizer,
            max_lr=getattr(config, "max_lr", 1e-3),
            epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1e4,
        ), False

    elif name == "flat_cosine":
        flat_epochs = getattr(config, "flat_epochs", 10)
        warmup_epochs = getattr(config, "warmup_epochs", 5)
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        flat = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=flat_epochs)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - flat_epochs - warmup_epochs,
            eta_min=getattr(config, "eta_min", 1e-6),
        )
        combined = SequentialLR(
            optimizer,
            [warmup, flat, cosine],
            milestones=[warmup_epochs, warmup_epochs + flat_epochs],
        )
        return combined, True

    elif name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=getattr(config, "plateau_factor", 0.1),
            patience=getattr(config, "plateau_patience", 10),
            min_lr=getattr(config, "eta_min", 1e-6),
        ), True

    elif name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=getattr(config, "eta_min", 1e-6),
        )
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=getattr(config, "warmup_epochs", 10),
        )
        combined = SequentialLR(
            optimizer,
            [warmup, scheduler],
            milestones=[getattr(config, "warmup_epochs", 10)],
        )
        return combined, True

    elif name == "step":
        return StepLR(
            optimizer,
            step_size=getattr(config, "step_size", 30),
            gamma=getattr(config, "gamma", 0.1),
        ), True

    else:
        raise ValueError(f"Unknown scheduler: {name}")


def create_optimizer(
    model: torch.nn.Module,
    config: Any,
    param_groups: list[dict] | None = None,
) -> Optimizer:
    """Create optimizer from config with optional parameter groups."""
    name = getattr(config, "name", "adamw")
    lr = getattr(config, "lr", 1e-3)
    weight_decay = getattr(config, "weight_decay", 0.05)

    if param_groups is None:
        param_groups = [{"params": model.parameters()}]

    if name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=getattr(config, "betas", (0.9, 0.999)),
            eps=getattr(config, "eps", 1e-8),
            weight_decay=weight_decay,
        )
    elif name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=getattr(config, "momentum", 0.9),
            weight_decay=weight_decay,
            nesterov=getattr(config, "nesterov", True),
        )
    elif name == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=getattr(config, "betas", (0.9, 0.999)),
            eps=getattr(config, "eps", 1e-8),
            weight_decay=weight_decay,
        )
    elif name == "adabelief":
        return AdaBelief(
            param_groups,
            lr=lr,
            betas=getattr(config, "betas", (0.9, 0.999)),
            eps=getattr(config, "eps", 1e-8),
            weight_decay=weight_decay,
        )
    elif name == "lookahead_adamw":
        base = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=getattr(config, "betas", (0.9, 0.999)),
            eps=getattr(config, "eps", 1e-8),
            weight_decay=weight_decay,
        )
        return Lookahead(base, k=5, alpha=0.5)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ── Gradient Accumulation ────────────────────────────────────────────────────


class GradientAccumulator:
    """Wraps model to accumulate gradients over multiple micro-batches."""

    def __init__(self, model: torch.nn.Module, accumulation_steps: int = 4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self._current_step = 0

    def zero_grad(self, set_to_none: bool = True):
        if self._current_step % self.accumulation_steps == 0:
            self.model.zero_grad(set_to_none=set_to_none)

    def backward(self, loss: torch.Tensor) -> None:
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        self._current_step += 1

    def step(self, optimizer: Optimizer) -> bool:
        """Step optimizer if enough gradients accumulated. Returns True if stepped."""
        if self._current_step % self.accumulation_steps == 0:
            optimizer.step()
            return True
        return False

    def state_dict(self):
        return {"current_step": self._current_step}

    def load_state_dict(self, state: dict):
        self._current_step = state["current_step"]


# ── Learning Rate Finder ─────────────────────────────────────────────────────


class LRFinder:
    """Learning rate range test (Smith, 2017).

    Trains the model with exponentially increasing learning rates
    to find the optimal LR range.
    """

    def __init__(self, model, optimizer, criterion, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {"lr": [], "loss": []}

    def range_test(self, loader, start_lr=1e-7, end_lr=10, num_iter=100):
        """Run the LR range test."""
        self.model.train()
        lr_schedule = np.geomspace(start_lr, end_lr, num_iter)

        best_loss = float("inf")
        for idx, lr in enumerate(lr_schedule):
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            try:
                inputs, labels = next(iter(loader))
            except StopIteration:
                break

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.history["lr"].append(lr)
            self.history["loss"].append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
            if loss.item() > 4 * best_loss:
                break

        return self.history

    def plot(self, save_path: str | None = None):
        """Plot the LR vs loss curve."""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(self.history["lr"], self.history["loss"])
            plt.xscale("log")
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.title("Learning Rate Finder")
            plt.grid(True, alpha=0.3)
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        except ImportError:
            pass


# ── Discriminative Learning Rates ────────────────────────────────────────────


def create_discriminative_param_groups(
    model: torch.nn.Module,
    base_lr: float = 1e-3,
    decay_factor: float = 0.1,
) -> list[dict]:
    """Create parameter groups with discriminative learning rates.

    Earlier layers get lower LR, later layers get higher LR.
    Useful for fine-tuning pretrained models.
    """
    layers = []
    for name, _ in model.named_parameters():
        layer_name = name.split(".")[0]
        if layer_name not in layers:
            layers.append(layer_name)

    n_layers = len(layers)
    param_groups = []
    for i, layer_name in enumerate(layers):
        layer_params = [p for n, p in model.named_parameters() if n.startswith(layer_name) and p.requires_grad]
        if layer_params:
            lr_scale = decay_factor ** (n_layers - 1 - i)
            param_groups.append(
                {
                    "params": layer_params,
                    "lr": base_lr * lr_scale,
                    "name": layer_name,
                }
            )

    return param_groups
