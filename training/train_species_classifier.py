"""Train a UK species classifier (badger, otter, other) for CameraTrap."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from validate_species_dataset import validate_dataset
from wildlife_classifier import DEFAULT_NORMALIZATION, build_classifier_model


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


class OrderedImageFolder(datasets.ImageFolder):
    """ImageFolder that enforces a fixed class order."""

    def __init__(self, root: str | Path, class_names: Sequence[str], **kwargs):
        self.class_names = list(class_names)
        super().__init__(root=str(root), **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        directory_path = Path(directory)
        available = {entry.name for entry in directory_path.iterdir() if entry.is_dir()}
        missing = [name for name in self.class_names if name not in available]
        if missing:
            raise FileNotFoundError(
                f"Missing class folders in {directory_path}: {missing}. "
                f"Expected classes: {self.class_names}"
            )
        class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        return self.class_names, class_to_idx


def confusion_matrix(num_classes: int, targets: np.ndarray, preds: np.ndarray) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(targets.tolist(), preds.tolist()):
        matrix[int(target), int(pred)] += 1
    return matrix


def classification_metrics_from_confusion(
    matrix: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, object]:
    per_class: Dict[str, Dict[str, float]] = {}
    f1_values = []
    precision_values = []
    recall_values = []

    for class_idx, class_name in enumerate(class_names):
        tp = int(matrix[class_idx, class_idx])
        fp = int(matrix[:, class_idx].sum() - tp)
        fn = int(matrix[class_idx, :].sum() - tp)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)

        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[class_idx, :].sum()),
        }
        f1_values.append(f1)
        precision_values.append(precision)
        recall_values.append(recall)

    accuracy = safe_div(int(np.trace(matrix)), int(matrix.sum()))
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "macro_precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "macro_recall": float(np.mean(recall_values)) if recall_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.numel())
            running_loss += float(loss.item()) * int(targets.size(0))

    return {
        "loss": safe_div(running_loss, total),
        "accuracy": safe_div(correct, total),
    }


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    class_names: Sequence[str],
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_targets: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            running_loss += float(loss.item()) * int(targets.size(0))
            total += int(targets.numel())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    targets_np = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    probs_np = np.concatenate(all_probs) if all_probs else np.empty((0, len(class_names)), dtype=np.float32)
    preds_np = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)

    matrix = confusion_matrix(len(class_names), targets_np, preds_np)
    metrics = classification_metrics_from_confusion(matrix, class_names)
    metrics["loss"] = safe_div(running_loss, total)

    return {
        "metrics": metrics,
        "targets": targets_np,
        "probs": probs_np,
        "preds": preds_np,
    }


def compute_class_weights(targets: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(targets), minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    total = float(np.sum(counts))
    weights = total / (float(num_classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_protected_alerts(
    probs: np.ndarray,
    targets: np.ndarray,
    class_names: Sequence[str],
    protected_classes: Sequence[str],
    threshold: float,
) -> Dict[str, object]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    protected_indices = [class_to_idx[name] for name in protected_classes]
    unknown_index = len(protected_indices)
    alert_labels = list(protected_classes) + ["unknown"]

    top_idx = probs.argmax(axis=1)
    top_conf = probs.max(axis=1)

    mapped_targets = np.full(shape=(len(targets),), fill_value=unknown_index, dtype=np.int64)
    for idx_position, cls_idx in enumerate(protected_indices):
        mapped_targets[targets == cls_idx] = idx_position

    mapped_preds = np.full(shape=(len(top_idx),), fill_value=unknown_index, dtype=np.int64)
    for idx_position, cls_idx in enumerate(protected_indices):
        mask = (top_idx == cls_idx) & (top_conf >= threshold)
        mapped_preds[mask] = idx_position

    matrix = confusion_matrix(len(alert_labels), mapped_targets, mapped_preds)
    metrics = classification_metrics_from_confusion(matrix, alert_labels)

    protected_precision = {}
    protected_recall = {}
    for class_name in protected_classes:
        protected_precision[class_name] = metrics["per_class"][class_name]["precision"]
        protected_recall[class_name] = metrics["per_class"][class_name]["recall"]

    metrics["protected_precision"] = protected_precision
    metrics["protected_recall"] = protected_recall
    metrics["threshold"] = float(threshold)
    metrics["labels"] = alert_labels
    return metrics


def calibrate_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    class_names: Sequence[str],
    protected_classes: Sequence[str],
    precision_target: float,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> Dict[str, object]:
    thresholds = np.linspace(threshold_min, threshold_max, threshold_steps)
    candidates: List[Dict[str, object]] = []
    qualified: List[Dict[str, object]] = []

    for threshold in thresholds:
        metrics = evaluate_protected_alerts(
            probs=probs,
            targets=targets,
            class_names=class_names,
            protected_classes=protected_classes,
            threshold=float(threshold),
        )
        precision_values = [metrics["protected_precision"][name] for name in protected_classes]
        recall_values = [metrics["protected_recall"][name] for name in protected_classes]
        mean_precision = float(np.mean(precision_values)) if precision_values else 0.0
        mean_recall = float(np.mean(recall_values)) if recall_values else 0.0

        row = {
            "threshold": float(threshold),
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "protected_precision": metrics["protected_precision"],
            "protected_recall": metrics["protected_recall"],
        }
        candidates.append(row)

        if all(value >= precision_target for value in precision_values):
            qualified.append(row)

    if qualified:
        best = max(qualified, key=lambda item: (item["mean_recall"], item["mean_precision"], -item["threshold"]))
        strategy = "precision_constrained"
    else:
        best = max(candidates, key=lambda item: (item["mean_precision"], item["mean_recall"], -item["threshold"]))
        strategy = "best_available_precision"

    return {
        "recommended_threshold": float(best["threshold"]),
        "strategy": strategy,
        "precision_target": float(precision_target),
        "candidates": candidates,
    }


def sanitize_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    return json.loads(json.dumps(metrics))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train UK species classifier for CameraTrap.")
    parser.add_argument("--data-root", default="dataset_uk_species", help="Dataset root with train/val/test folders")
    parser.add_argument("--classes", default="badger,otter,other", help="Comma-separated class names")
    parser.add_argument("--protected-classes", default="badger,otter", help="Protected target classes")
    parser.add_argument("--arch", default="mobilenet_v3_small", choices=["mobilenet_v3_small"], help="Backbone architecture")
    parser.add_argument("--input-size", type=int, default=224, help="Model input image size")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained initialization")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda")
    parser.add_argument("--output", default="wildlife_model_badger_otter.pth", help="Output checkpoint path")
    parser.add_argument("--precision-target", type=float, default=0.85, help="Minimum precision target for protected classes")
    parser.add_argument("--threshold-min", type=float, default=0.30, help="Minimum threshold for calibration")
    parser.add_argument("--threshold-max", type=float, default=0.95, help="Maximum threshold for calibration")
    parser.add_argument("--threshold-steps", type=int, default=66, help="Number of candidate thresholds")
    parser.add_argument("--min-train", type=int, default=400, help="Min images/class required in train")
    parser.add_argument("--min-val", type=int, default=100, help="Min images/class required in val")
    parser.add_argument("--min-test", type=int, default=100, help="Min images/class required in test")
    parser.add_argument("--skip-dataset-validation", action="store_true", help="Skip dataset validation checks")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class-weighted cross-entropy")
    parser.add_argument("--imbalance-ratio-threshold", type=float, default=1.20, help="Apply class weights when max/min count exceeds this ratio")
    return parser


def resolve_device(raw_value: str) -> torch.device:
    if raw_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_value)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    class_names = parse_csv(args.classes)
    protected_classes = parse_csv(args.protected_classes)
    unknown_protected = [name for name in protected_classes if name not in class_names]
    if unknown_protected:
        parser.error(f"Protected classes not in classes list: {unknown_protected}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Training device: {device}")

    if not args.skip_dataset_validation:
        validation = validate_dataset(
            data_root=args.data_root,
            required_classes=class_names,
            expected_splits=("train", "val", "test"),
            min_counts={
                "train": args.min_train,
                "val": args.min_val,
                "test": args.min_test,
            },
            check_corrupt=True,
        )
        if not validation.ok:
            print(json.dumps(validation.to_dict(), indent=2))
            print("Dataset validation failed. Fix dataset issues or run with --skip-dataset-validation.")
            return 1

    normalization = DEFAULT_NORMALIZATION
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((args.input_size, args.input_size), scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization["mean"], std=normalization["std"]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization["mean"], std=normalization["std"]),
    ])

    data_root = Path(args.data_root)
    train_dataset = OrderedImageFolder(data_root / "train", class_names=class_names, transform=train_transform)
    val_dataset = OrderedImageFolder(data_root / "val", class_names=class_names, transform=eval_transform)
    test_dataset = OrderedImageFolder(data_root / "test", class_names=class_names, transform=eval_transform)

    if len(train_dataset) == 0:
        print("No training data found.")
        return 1
    if len(val_dataset) == 0:
        print("No validation data found.")
        return 1
    if len(test_dataset) == 0:
        print("No test data found.")
        return 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_classifier_model(
        args.arch,
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
    ).to(device)
    class_counts = np.bincount(np.array(train_dataset.targets), minlength=len(class_names))
    ratio = float(class_counts.max()) / float(max(1, class_counts.min()))

    class_weights = None
    if not args.no_class_weights and ratio > args.imbalance_ratio_threshold:
        class_weights = compute_class_weights(train_dataset.targets, num_classes=len(class_names)).to(device)
        print(f"Using class weights: {class_weights.tolist()} (imbalance ratio={ratio:.3f})")
    else:
        print(f"Class weights disabled (imbalance ratio={ratio:.3f})")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_macro_f1 = -1.0
    best_state_dict = None
    history = []

    for epoch_idx in range(args.epochs):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_result = evaluate_split(model, val_loader, criterion, class_names, device)
        val_metrics = val_result["metrics"]

        history_entry = {
            "epoch": epoch_idx + 1,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(history_entry)
        print(
            f"Epoch {epoch_idx + 1:03d}/{args.epochs:03d} "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is None:
        print("Training did not produce a valid checkpoint.")
        return 1

    model.load_state_dict(best_state_dict)

    val_result = evaluate_split(model, val_loader, criterion, class_names, device)
    test_result = evaluate_split(model, test_loader, criterion, class_names, device)

    threshold_calibration = calibrate_threshold(
        probs=val_result["probs"],
        targets=val_result["targets"],
        class_names=class_names,
        protected_classes=protected_classes,
        precision_target=args.precision_target,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_steps=args.threshold_steps,
    )
    chosen_threshold = threshold_calibration["recommended_threshold"]

    val_alert_metrics = evaluate_protected_alerts(
        probs=val_result["probs"],
        targets=val_result["targets"],
        class_names=class_names,
        protected_classes=protected_classes,
        threshold=chosen_threshold,
    )
    test_alert_metrics = evaluate_protected_alerts(
        probs=test_result["probs"],
        targets=test_result["targets"],
        class_names=class_names,
        protected_classes=protected_classes,
        threshold=chosen_threshold,
    )

    metrics = {
        "class_names": class_names,
        "protected_classes": protected_classes,
        "history": history,
        "val": val_result["metrics"],
        "test": test_result["metrics"],
        "threshold_calibration": threshold_calibration,
        "recommended_clf_conf_threshold": chosen_threshold,
        "val_protected_alerts": val_alert_metrics,
        "test_protected_alerts": test_alert_metrics,
        "train_class_counts": {class_names[idx]: int(count) for idx, count in enumerate(class_counts.tolist())},
    }
    metrics = sanitize_metrics(metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state_dict,
        "class_names": class_names,
        "input_size": int(args.input_size),
        "arch": args.arch,
        "normalization": normalization,
        "metrics": metrics,
    }
    torch.save(checkpoint, output_path)

    metrics_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved best checkpoint to: {output_path}")
    print(f"Saved metrics report to: {metrics_path}")
    print(f"Recommended classifier confidence threshold: {chosen_threshold:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
