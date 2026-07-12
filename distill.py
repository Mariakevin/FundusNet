"""Knowledge distillation — distill ensemble into a single efficient model.

Distills knowledge from multiple teacher models into one student model,
achieving near-ensemble accuracy at a fraction of inference cost.

Uses soft target learning (Hinton et al., 2015) with temperature scaling.
"""

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Knowledge distillation for retina ensemble")
parser.add_argument("--dataset", default=os.path.join(SCRIPT_DIR, "retina_dataset"))
parser.add_argument("--output", default=SCRIPT_DIR)
parser.add_argument("--teacher-models", nargs="+", default=["convnext_v2", "efficientnet_v2", "deit"])
parser.add_argument("--student-model", default="efficientnet_b0")
parser.add_argument("--temperature", type=float, default=4.0)
parser.add_argument("--alpha", type=float, default=0.7, help="Weight for soft loss (1-alpha for hard loss)")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--image-size", type=int, default=224)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--label-smoothing", type=float, default=0.1)
args = parser.parse_args()

from retina_app.utils import (
    CATEGORIES,
    EMA,
    RetinaDataset,
    create_model,
    setup_seed,
)

SEED = args.seed
setup_seed(SEED)

IMAGE_SIZE = args.image_size
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.lr
TEMPERATURE = args.temperature
ALPHA = args.alpha


train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_teacher(model_name, num_classes, device, models_dir):
    """Load a trained teacher model."""
    model = create_model(model_name, num_classes, pretrained=False)
    ckpt_path = os.path.join(models_dir, f"{model_name}_retinopathy.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"  Loaded teacher {model_name} from {ckpt_path}")
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"  WARNING: No checkpoint for {model_name}, using pretrained weights")
    model.to(device)
    model.eval()
    return model


def distillation_loss(student_logits, teacher_logits, true_labels, temperature, alpha, label_smoothing=0.0):
    """Combined hard + soft distillation loss."""
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature**2)

    hard_loss = F.cross_entropy(student_logits, true_labels, label_smoothing=label_smoothing)
    return alpha * soft_loss + (1 - alpha) * hard_loss


def ensemble_soft_targets(teachers, image_tensor, model_weights=None):
    """Compute ensemble soft targets from multiple teachers."""
    all_probs = []
    with torch.no_grad():
        for teacher in teachers:
            output = teacher(image_tensor)
            if isinstance(output, tuple):
                output = output[0]
            probs = F.softmax(output, dim=1)
            all_probs.append(probs)
    if model_weights is not None:
        stacked = torch.stack(all_probs, dim=0)
        weights = torch.tensor(model_weights, device=stacked.device).view(-1, 1, 1)
        weighted = (stacked * weights).sum(dim=0)
        weighted = weighted / weighted.sum(dim=1, keepdim=True)
        return weighted
    else:
        return torch.stack(all_probs).mean(dim=0)


def main():
    os.makedirs(os.path.join(SCRIPT_DIR, "models"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Knowledge distillation on {device}")
    print(f"Teachers: {args.teacher_models}")
    print(f"Student: {args.student_model}")
    print(f"Temperature: {TEMPERATURE}, Alpha: {ALPHA}")

    models_dir = os.path.join(SCRIPT_DIR, "models")
    teachers = [load_teacher(name, len(CATEGORIES), device, models_dir) for name in args.teacher_models]
    teacher_weights = [1.0 / len(teachers)] * len(teachers)

    student = create_model(args.student_model, len(CATEGORIES), pretrained=True).to(device)

    full_dataset = RetinaDataset(args.dataset, transform=train_transform)
    labels = [label for _, label in full_dataset.samples]

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    train_idx, val_idx = list(skf.split(np.zeros(len(labels)), labels))[0]
    train_dataset = Subset(RetinaDataset(args.dataset, transform=train_transform), train_idx)
    val_dataset = Subset(RetinaDataset(args.dataset, transform=val_transform), val_idx)

    n_workers = min(os.cpu_count() or 4, 8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=n_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=n_workers,
    )

    optimizer = optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    ema = EMA(student, decay=0.9999)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        student.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            soft_targets = ensemble_soft_targets(teachers, images, teacher_weights)

            student_logits = student(images)
            teacher_logits = torch.log(soft_targets + 1e-10)

            loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels_batch,
                TEMPERATURE,
                ALPHA,
                args.label_smoothing,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            ema.update()

            train_loss += loss.item()
            _, predicted = student_logits.max(1)
            train_total += labels_batch.size(0)
            train_correct += predicted.eq(labels_batch).sum().item()

        scheduler.step()

        ema.apply_shadow()
        student.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                outputs = student(images)
                _, predicted = outputs.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()
        ema.restore()

        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ema.apply_shadow()
            save_path = os.path.join(SCRIPT_DIR, "models", f"{args.student_model}_distilled.pth")
            torch.save(
                {
                    "model_state_dict": copy.deepcopy(student.state_dict()),
                    "num_classes": len(CATEGORIES),
                    "categories": CATEGORIES,
                    "model_type": args.student_model,
                    "distilled_from": args.teacher_models,
                    "temperature": TEMPERATURE,
                    "alpha": ALPHA,
                    "best_val_acc": best_val_acc,
                },
                save_path,
            )
            ema.restore()
            print(f"  Saved distilled model ({val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    print(f"\nBest distilled student accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
