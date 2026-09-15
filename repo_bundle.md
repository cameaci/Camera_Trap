# CameraTrap repo bundle

Olusturulma: 2026-09-14 17:23 (yerel saat)
Dizin: C:\Python Scripts\CameraTrap
```text
python komutu: python
Python 3.13.0

--- ilgili paketler ---
(eslesen paket yok)

--- bagimlilik dosyalari ---
YOK: requirements.txt
YOK: pyproject.toml
YOK: environment.yml
YOK: setup.py
```text
.\app.ipynb
.\app.py
.\benchmark_species_classifier.py
.\bundle_repo.ps1
.\bundle_repo.ps1.md
.\fetch_uk_open_data.py
.\model.py
.\new_model_test.py
.\species_postprocessing.py
.\speciesnet_adapter.py
.\train_species_classifier.py
.\validate_species_dataset.py
.\wildlife_classifier.py
.\wildlife_model_badger_otter.pth.metrics.json
.\.vscode\settings.json
.\demo_data\imgs\image_recognition_file.json
.\demo_data\imgs\results.json
.\demo_data\imgs\results.xlsx
.\demo_data\imgs\graphs\activity-patterns\hour-of-day\combined.html
.\demo_data\imgs\graphs\activity-patterns\hour-of-day\class-specific\animal.html
.\demo_data\imgs\graphs\activity-patterns\hour-of-day\class-specific\person.html
.\demo_data\imgs\graphs\activity-patterns\month-of-year\combined.html
.\demo_data\imgs\graphs\activity-patterns\month-of-year\class-specific\animal.html
.\demo_data\imgs\graphs\activity-patterns\month-of-year\class-specific\person.html
.\demo_data\imgs\graphs\bar-charts\grouped-by-year\combined-multi-layer.html
.\demo_data\imgs\graphs\bar-charts\grouped-by-year\combined-single-layer.html
.\demo_data\imgs\graphs\bar-charts\grouped-by-year\class-specific\animal.html
.\demo_data\imgs\graphs\bar-charts\grouped-by-year\class-specific\person.html
.\demo_data\imgs\graphs\pie-charts\distribution-detections.html
.\demo_data\imgs\graphs\pie-charts\distribution-files.html
.\demo_data\imgs\graphs\temporal-heatmaps\grouped-by-year\absolute\temporal-heatmap.html
.\demo_data\imgs\graphs\temporal-heatmaps\grouped-by-year\relative\temporal-heatmap.html
.\reports\dataset_validation.json
.\tests\test_fetch_uk_open_data.py
.\tests\test_species_postprocessing.py
.\tests\test_speciesnet_adapter.py
.\tests\test_validate_species_dataset.py
.\tests\test_wildlife_classifier_adapter.py

### train_species_classifier.py

```python
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


### wildlife_classifier.py

```python
"""Utilities for loading and running wildlife species classification models."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

from species_postprocessing import format_species_detection_label

DEFAULT_INPUT_SIZE = 224
DEFAULT_NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
__all__ = [
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_NORMALIZATION",
    "build_classifier_model",
    "WildlifeClassificationAdapter",
    "format_species_detection_label",
]


def build_classifier_model(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Build a supported backbone configured for a class count."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture '{arch}'. Supported: mobilenet_v3_small")


def _as_class_name_map(class_names: object) -> Dict[int, str]:
    if isinstance(class_names, Mapping):
        parsed: Dict[int, str] = {}
        for key, value in class_names.items():
            parsed[int(key)] = str(value)
        return parsed
    if isinstance(class_names, Sequence) and not isinstance(class_names, (str, bytes)):
        return {idx: str(name) for idx, name in enumerate(class_names)}
    return {}


def _infer_num_classes_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> int:
    for key in ("classifier.3.weight", "fc.weight", "head.weight"):
        weights = state_dict.get(key)
        if isinstance(weights, torch.Tensor) and weights.ndim >= 2:
            return int(weights.shape[0])
    # fallback: choose first 2D tensor output channels
    for weights in state_dict.values():
        if isinstance(weights, torch.Tensor) and weights.ndim >= 2:
            return int(weights.shape[0])
    return 0


def _normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    if keys and all(key.startswith("module.") for key in keys):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return dict(state_dict)


class WildlifeClassificationAdapter:
    """Adapter that loads legacy or checkpoint-based models for inference."""

    def __init__(self, weights_path: str, device: str | torch.device):
        self.device = torch.device(device)
        self.arch = None
        self.metrics = {}
        self.checkpoint = {}
        self.supports_batch = False  # Single-image only for now.

        loaded = self._torch_load_compatible(weights_path)
        self.model, metadata = self._load_model_and_metadata(loaded)
        self.checkpoint = metadata
        self.arch = metadata.get("arch")
        self.metrics = metadata.get("metrics", {})

        normalization = metadata.get("normalization", DEFAULT_NORMALIZATION)
        mean = normalization.get("mean", DEFAULT_NORMALIZATION["mean"])
        std = normalization.get("std", DEFAULT_NORMALIZATION["std"])
        input_size = metadata.get("input_size", DEFAULT_INPUT_SIZE)
        if isinstance(input_size, Sequence) and not isinstance(input_size, (str, bytes)):
            input_size = int(input_size[0])
        else:
            input_size = int(input_size)

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.CLASS_NAMES = self._resolve_class_names(metadata)

    def _torch_load_compatible(self, weights_path: str):
        try:
            return torch.load(weights_path, map_location=self.device)
        except Exception as exc:
            message = str(exc)
            # PyTorch >=2.6 defaults to weights_only=True. Retry for trusted legacy full-model checkpoints.
            if "weights_only" in message.lower() or "unsupported global" in message.lower():
                return torch.load(weights_path, map_location=self.device, weights_only=False)
            raise

    def _load_model_and_metadata(self, loaded: object) -> tuple[nn.Module, Dict[str, object]]:
        if isinstance(loaded, Mapping) and "state_dict" in loaded:
            checkpoint = dict(loaded)
            state_dict = _normalize_state_dict_keys(checkpoint["state_dict"])
            class_names = _as_class_name_map(checkpoint.get("class_names"))
            num_classes = len(class_names) or _infer_num_classes_from_state_dict(state_dict)
            if num_classes <= 0:
                raise ValueError("Unable to infer class count from checkpoint state_dict")

            arch = str(checkpoint.get("arch", "mobilenet_v3_small"))
            model = build_classifier_model(arch=arch, num_classes=num_classes)
            model.load_state_dict(state_dict, strict=True)
            model = model.to(self.device)
            model.eval()
            checkpoint["class_names"] = class_names or {idx: f"Species {idx}" for idx in range(num_classes)}
            return model, checkpoint

        if hasattr(loaded, "to"):
            loaded = loaded.to(self.device)
        if hasattr(loaded, "eval"):
            loaded.eval()
        return loaded, {}

    def _resolve_class_names(self, metadata: Mapping[str, object]) -> Dict[int, str]:
        metadata_names = _as_class_name_map(metadata.get("class_names"))
        if metadata_names:
            return metadata_names

        model_names = _as_class_name_map(getattr(self.model, "CLASS_NAMES", None))
        if model_names:
            return model_names

        num_classes = self._infer_num_classes()
        if num_classes:
            return {idx: f"Species {idx}" for idx in range(num_classes)}
        return {}

    def _infer_num_classes(self) -> int:
        try:
            with torch.no_grad():
                sample = torch.zeros(1, 3, DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE, device=self.device)
                output = self.model(sample)
            if output.ndim >= 2:
                return int(output.shape[-1])
        except Exception:
            return 0
        return 0

    def _prepare_tensor(self, image: Image.Image | np.ndarray) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL Image or numpy array for classification")
        return self.transform(image).unsqueeze(0).to(self.device)

    def _format_label(self, class_idx: int) -> str:
        if self.CLASS_NAMES:
            return self.CLASS_NAMES.get(class_idx, f"Species {class_idx}")
        return f"Species {class_idx}"

    def single_image_classification(self, image: Image.Image | np.ndarray) -> Dict[str, object]:
        tensor = self._prepare_tensor(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, class_idx = torch.max(probs, dim=1)
        idx = int(class_idx.item())
        return {
            "prediction": self._format_label(idx),
            "confidence": float(confidence.item()),
            "class_id": idx,
        }


### speciesnet_adapter.py

```python
"""Adapter for running SpeciesNet as a crop classification model."""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

DEFAULT_SPECIESNET_MODEL = "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"
DEFAULT_SPECIESNET_CACHE_PARTS = (
    "models",
    "google",
    "speciesnet",
    "pyTorch",
    "v4.0.2a",
    "1",
)


def _is_speciesnet_model_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "info.json").is_file()
        and any(path.glob("*.pt"))
        and any(path.glob("*.labels.*.txt"))
    )


def find_cached_speciesnet_model() -> Path | None:
    """Find the default SpeciesNet model in KaggleHub's local cache."""
    cache_roots = []
    kagglehub_cache = os.environ.get("KAGGLEHUB_CACHE")
    if kagglehub_cache:
        cache_roots.append(Path(kagglehub_cache))
    cache_roots.append(Path.home() / ".cache" / "kagglehub")

    for cache_root in cache_roots:
        candidate = cache_root.joinpath(*DEFAULT_SPECIESNET_CACHE_PARTS)
        if _is_speciesnet_model_dir(candidate):
            return candidate
    return None


def resolve_speciesnet_model_name(model_name: str | None) -> str:
    """Prefer the local cached default model to avoid unnecessary downloads."""
    if model_name and model_name != DEFAULT_SPECIESNET_MODEL:
        return model_name

    cached_model = find_cached_speciesnet_model()
    if cached_model is not None:
        return str(cached_model)
    return model_name or DEFAULT_SPECIESNET_MODEL


def format_speciesnet_label(raw_label: object) -> str:
    """Convert SpeciesNet taxonomy strings into compact display labels."""
    label = str(raw_label or "").strip()
    if not label:
        return "Unknown"

    parts = [part.strip() for part in label.split(";")]
    if len(parts) == 1:
        return label

    ignored = {"", "no cv result"}
    for part in reversed(parts[1:]):
        if part.lower() not in ignored:
            return part
    return "Unknown"


class SpeciesNetClassificationAdapter:
    """Small adapter matching the app's single-image classifier contract."""

    supports_batch = False

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        classifier_cls: type | None = None,
    ) -> None:
        if classifier_cls is None:
            try:
                speciesnet = import_module("speciesnet")
            except ImportError as exc:
                raise ImportError(
                    "SpeciesNet is not installed. Install it with `pip install speciesnet` "
                    "to use the SpeciesNet classifier."
                ) from exc

            classifier_cls = getattr(speciesnet, "SpeciesNetClassifier")
            default_model = getattr(speciesnet, "DEFAULT_MODEL", DEFAULT_SPECIESNET_MODEL)
        else:
            default_model = DEFAULT_SPECIESNET_MODEL

        requested_model_name = model_name or default_model
        self.model_name = resolve_speciesnet_model_name(requested_model_name)
        self.device = device
        try:
            self.classifier = classifier_cls(model_name=self.model_name, device=device)
        except Exception as exc:
            message = str(exc)
            if "CERTIFICATE_VERIFY_FAILED" in message or "SSLCertVerificationError" in message:
                raise RuntimeError(
                    "SpeciesNet model could not be downloaded because SSL certificate "
                    "verification failed. Use a local SpeciesNet model folder in the "
                    "Weights / Model Path field, or fix the Kaggle/HuggingFace SSL "
                    "certificate chain in this Python environment."
                ) from exc
            raise
        self.CLASS_NAMES = {
            idx: format_speciesnet_label(label)
            for idx, label in getattr(self.classifier, "labels", {}).items()
        }
        self._label_to_idx = {
            label: idx for idx, label in getattr(self.classifier, "labels", {}).items()
        }

    def _prepare_image(self, image: Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL Image or numpy array for classification")
        return image.convert("RGB")

    def single_image_classification(self, image: Image.Image | np.ndarray) -> Dict[str, Any]:
        pil_image = self._prepare_image(image)
        preprocessed = self.classifier.preprocess(pil_image, bboxes=None)
        result = self.classifier.predict("<memory>", preprocessed)
        classifications = result.get("classifications", {}) if isinstance(result, dict) else {}
        classes = classifications.get("classes") or []
        scores = classifications.get("scores") or []

        if not classes or not scores:
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "class_id": -1,
                "raw_prediction": None,
            }

        raw_prediction = classes[0]
        return {
            "prediction": format_speciesnet_label(raw_prediction),
            "confidence": float(scores[0]),
            "class_id": int(self._label_to_idx.get(raw_prediction, -1)),
            "raw_prediction": raw_prediction,
        }


### model.py

```python
"""Gradio interface combining MegaDetector with a custom classification model."""

import torch
import numpy as np
import gradio as gr
import supervision as sv
from PIL import Image
from ultralytics import YOLO

from wildlife_classifier import (
    WildlifeClassificationAdapter,
    format_species_detection_label,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Annotators for visualising detections
BOX_ANNOTATOR = sv.BoxAnnotator(thickness=4)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=4, text_scale=2)

DETECTION_MODEL_PATH = "MegaDetectorV6.pt"
CLASSIFICATION_MODEL_PATH = "wildlife_model.pth"

print("Loading MegaDetector model...")
detection_model = YOLO(DETECTION_MODEL_PATH)
print("MegaDetector loaded successfully.")

print("Loading classification model...")
classification_model = WildlifeClassificationAdapter(CLASSIFICATION_MODEL_PATH, device=DEVICE)
print("Classification model loaded successfully.")

def classify_species(image: Image.Image) -> dict:
    """Predict species for a crop."""
    return classification_model.single_image_classification(image)


def single_image_detection(
    input_img: Image.Image,
    det_conf_thres: float = 0.25,
    clf_conf_thres: float = 0.70,
) -> Image.Image:
    """Run detection and classification on a single image."""
    input_img_np = np.array(input_img)
    raw_result = detection_model(input_img_np)[0]
    detections = sv.Detections.from_ultralytics(raw_result)

    if len(detections) == 0:
        return input_img

    mask = detections.confidence > det_conf_thres
    filtered_detections = detections[mask]

    if len(filtered_detections) == 0:
        return input_img

    labels = []
    for bbox, confidence in zip(filtered_detections.xyxy, filtered_detections.confidence):
        x_min, y_min, x_max, y_max = map(int, bbox)
        crop = input_img.crop((x_min, y_min, x_max, y_max))
        species = classify_species(crop)
        labels.append(format_species_detection_label(species, conf_threshold=clf_conf_thres))

    annotated_scene = LABEL_ANNOTATOR.annotate(
        scene=BOX_ANNOTATOR.annotate(scene=input_img_np.copy(), detections=filtered_detections),
        detections=filtered_detections,
        labels=labels,
    )
    return Image.fromarray(annotated_scene)


with gr.Blocks() as demo:
    gr.Markdown("# MegaDetector + Classification Model")

    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload image")
        conf_threshold = gr.Slider(0, 1, value=0.25, label="Detection confidence threshold")
        clf_conf_threshold = gr.Slider(0, 1, value=0.70, label="Classification confidence threshold")

    img_output = gr.Image(label="Annotated image")
    detect_button = gr.Button("Detect animals")

    detect_button.click(
        single_image_detection,
        inputs=[img_input, conf_threshold, clf_conf_threshold],
        outputs=img_output,
    )


if __name__ == "__main__":
    demo.launch(share=True)



### benchmark_species_classifier.py

```python
"""Benchmark single-crop classifier inference latency on CPU/GPU."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
from PIL import Image

from wildlife_classifier import WildlifeClassificationAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark species classifier latency.")
    parser.add_argument("--weights", default="wildlife_model_badger_otter.pth", help="Model checkpoint path")
    parser.add_argument("--device", default="cpu", help="Device for inference")
    parser.add_argument("--image", default="", help="Optional image path for benchmark input")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=200, help="Measured iterations")
    return parser


def load_image(path: str) -> Image.Image:
    if path:
        return Image.open(path).convert("RGB")
    array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(array)


def main() -> int:
    args = build_parser().parse_args()
    model = WildlifeClassificationAdapter(weights_path=args.weights, device=args.device)
    image = load_image(args.image)

    for _ in range(args.warmup):
        model.single_image_classification(image)

    timings_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        model.single_image_classification(image)
        end = time.perf_counter()
        timings_ms.append((end - start) * 1000.0)

    median_ms = statistics.median(timings_ms)
    p95_ms = sorted(timings_ms)[int(0.95 * (len(timings_ms) - 1))]
    mean_ms = statistics.fmean(timings_ms)

    print(f"median_ms={median_ms:.2f}")
    print(f"p95_ms={p95_ms:.2f}")
    print(f"mean_ms={mean_ms:.2f}")
    print("pass_under_80ms=" + str(median_ms < 80.0).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



### species_postprocessing.py

```python
"""Post-processing helpers for species classification output."""

from __future__ import annotations

from typing import Iterable, Mapping

DEFAULT_OTHER_ALIASES = ("other", "unknown", "background", "blank", "empty")


def normalize_species_label(label: str | None) -> str:
    """Normalize a label for robust comparisons."""
    if label is None:
        return ""
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


def _normalize_other_aliases(other_aliases: Iterable[str] | None) -> set[str]:
    aliases = other_aliases or DEFAULT_OTHER_ALIASES
    return {normalize_species_label(alias) for alias in aliases}


def is_unknown_prediction(
    prediction: str | None,
    confidence: float,
    conf_threshold: float,
    other_aliases: Iterable[str] | None = None,
) -> bool:
    """Decide whether a classification should be suppressed as unknown."""
    if confidence < conf_threshold:
        return True
    return normalize_species_label(prediction) in _normalize_other_aliases(other_aliases)


def format_species_detection_label(
    result: Mapping[str, object] | None,
    conf_threshold: float,
    other_aliases: Iterable[str] | None = None,
) -> str:
    """Format a model prediction for detection overlays."""
    if not isinstance(result, Mapping):
        return "Unknown"

    prediction = str(result.get("prediction", "Unknown"))
    confidence = float(result.get("confidence", 0.0))
    if is_unknown_prediction(prediction, confidence, conf_threshold, other_aliases=other_aliases):
        return "Unknown"
    return f"{prediction} {confidence:.2f}"


### new_model_test.py

```python
import argparse
from PIL import Image

from wildlife_classifier import WildlifeClassificationAdapter

DEVICE = "cpu"
MODEL_PATH = "wildlife_model.pth"


def load_model():
    return WildlifeClassificationAdapter(weights_path=MODEL_PATH, device=DEVICE)


def classify_image(model, image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    return model.single_image_classification(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a single image using wildlife_model.pth")
    parser.add_argument("image", help="Path to the image to classify")
    args = parser.parse_args()

    model = load_model()
    result = classify_image(model, args.image)
    print(f"Predicted class id: {result['class_id']}")
    print(f"Predicted label: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()


### fetch_uk_open_data.py

```python
"""Fetch UK open-licensed wildlife images from iNaturalist into dataset_uk_species."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from urllib import error as url_error
from urllib import parse, request

from PIL import Image, UnidentifiedImageError

INAT_API_BASE = "https://api.inaturalist.org/v1"
SPLITS = ("train", "val", "test")
DEFAULT_LICENSES = ("cc0", "cc-by", "cc-by-nc")
SIZE_TOKEN_RE = re.compile(r"/(square|small|medium|large|original)\.")

DEFAULT_CLASS_TAXA = {
    "badger": [("Meles meles", 855297)],
    "otter": [("Lutra lutra", 41850)],
    "other": [
        ("Vulpes vulpes", 42069),
        ("Capreolus capreolus", 42184),
        ("Erinaceus europaeus", 43042),
        ("Lepus europaeus", 43128),
        ("Sciurus carolinensis", 46017),
    ],
}


@dataclass
class PhotoRecord:
    class_name: str
    species: str
    observation_id: int
    photo_id: int
    license_code: str
    url: str


@dataclass
class ObservationRecord:
    class_name: str
    species: str
    observation_id: int
    photos: List[PhotoRecord]


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str, expected_len: int, label: str) -> List[int]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != expected_len:
        raise ValueError(f"{label} expects {expected_len} comma-separated integers")
    try:
        return [int(item) for item in parts]
    except ValueError as exc:
        raise ValueError(f"{label} contains non-integer values") from exc


def api_get_json(url: str, timeout: float = 30.0, retries: int = 5) -> Dict[str, object]:
    backoff = 1.5
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (url_error.URLError, url_error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_seconds = backoff ** attempt
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed API request after {retries} attempts: {url}") from last_error


def download_bytes(url: str, timeout: float = 30.0, retries: int = 4) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = request.Request(url, headers={"User-Agent": "CameraTrapUK/1.0"})
            with request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (url_error.URLError, url_error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(1.2 ** attempt)
    raise RuntimeError(f"Failed to download image after {retries} attempts: {url}") from last_error


def normalize_photo_url(url: str, preferred_size: str = "large") -> str:
    return SIZE_TOKEN_RE.sub(f"/{preferred_size}.", url)


def image_extension_from_format(fmt: str | None) -> str:
    mapping = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "BMP": ".bmp",
    }
    return mapping.get((fmt or "").upper(), ".jpg")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_dataset_dirs(root: Path) -> None:
    for split in SPLITS:
        for class_name in DEFAULT_CLASS_TAXA:
            (root / split / class_name).mkdir(parents=True, exist_ok=True)


def count_images(root: Path) -> Dict[str, Dict[str, int]]:
    counts = {split: {class_name: 0 for class_name in DEFAULT_CLASS_TAXA} for split in SPLITS}
    for split in SPLITS:
        for class_name in DEFAULT_CLASS_TAXA:
            class_dir = root / split / class_name
            if class_dir.exists():
                counts[split][class_name] = len([path for path in class_dir.glob("*") if path.is_file()])
    return counts


def load_existing_manifest(manifest_path: Path) -> tuple[list[dict[str, str]], set[str], Dict[str, Dict[str, int]]]:
    rows: list[dict[str, str]] = []
    hashes: set[str] = set()
    counts = {split: {class_name: 0 for class_name in DEFAULT_CLASS_TAXA} for split in SPLITS}
    if not manifest_path.exists():
        return rows, hashes, counts

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            sha_value = row.get("sha256", "")
            if sha_value:
                hashes.add(sha_value)
            split = row.get("split", "")
            class_name = row.get("class_name", "")
            if split in counts and class_name in counts[split]:
                counts[split][class_name] += 1
    return rows, hashes, counts


def fetch_observations_for_taxon(
    taxon_id: int,
    place_id: int,
    licenses_csv: str,
    per_page: int,
    max_observations: int,
    request_sleep: float,
) -> List[dict]:
    observations: List[dict] = []
    page = 1

    while True:
        params = {
            "taxon_id": taxon_id,
            "place_id": place_id,
            "quality_grade": "research",
            "photos": "true",
            "photo_license": licenses_csv,
            "per_page": per_page,
            "page": page,
            "order": "desc",
            "order_by": "created_at",
        }
        url = f"{INAT_API_BASE}/observations?{parse.urlencode(params)}"
        try:
            payload = api_get_json(url)
        except RuntimeError as exc:
            if observations:
                print(
                    f"WARNING: API paging interrupted for taxon_id={taxon_id} page={page}. "
                    f"Proceeding with {len(observations)} observations collected so far. Error: {exc}"
                )
                break
            raise
        results = payload.get("results", [])
        if not results:
            break

        observations.extend(results)
        if max_observations > 0 and len(observations) >= max_observations:
            observations = observations[:max_observations]
            break

        if len(results) < per_page:
            break

        page += 1
        if request_sleep > 0:
            time.sleep(request_sleep)

    return observations


def build_observation_records(
    class_name: str,
    species_name: str,
    observations: Sequence[dict],
    allowed_licenses: set[str],
) -> List[ObservationRecord]:
    grouped: List[ObservationRecord] = []
    for obs in observations:
        observation_id = int(obs.get("id"))
        photos = []
        for photo in obs.get("photos", []):
            raw_license = (photo.get("license_code") or "").strip().lower()
            if raw_license and raw_license not in allowed_licenses:
                continue

            photo_id = int(photo.get("id"))
            url = photo.get("original_url") or photo.get("url")
            if not url:
                continue
            url = normalize_photo_url(url, preferred_size="large")
            photos.append(
                PhotoRecord(
                    class_name=class_name,
                    species=species_name,
                    observation_id=observation_id,
                    photo_id=photo_id,
                    license_code=raw_license or "unknown",
                    url=url,
                )
            )

        if photos:
            grouped.append(
                ObservationRecord(
                    class_name=class_name,
                    species=species_name,
                    observation_id=observation_id,
                    photos=photos,
                )
            )
    return grouped


def interleave_observations(records: Sequence[ObservationRecord], rng: random.Random) -> List[ObservationRecord]:
    by_species: Dict[str, List[ObservationRecord]] = {}
    for rec in records:
        by_species.setdefault(rec.species, []).append(rec)

    for species in by_species:
        rng.shuffle(by_species[species])

    species_order = sorted(by_species.keys())
    interleaved: List[ObservationRecord] = []
    while True:
        pushed = 0
        for species in species_order:
            bucket = by_species[species]
            if bucket:
                interleaved.append(bucket.pop())
                pushed += 1
        if pushed == 0:
            break
    return interleaved


def choose_target_split(class_counts: Dict[str, int], split_targets: Dict[str, int]) -> str | None:
    candidates = [split for split in SPLITS if class_counts[split] < split_targets[split]]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda split: (
            (split_targets[split] - class_counts[split]) / max(1, split_targets[split]),
            split_targets[split] - class_counts[split],
        ),
    )


def validate_image_content(content: bytes, min_short_edge: int) -> tuple[bool, str, str]:
    try:
        with Image.open(BytesIO(content)) as img:
            width, height = img.size
            fmt = img.format or "JPEG"
            if min(width, height) < min_short_edge:
                return False, "", fmt
            return True, image_extension_from_format(fmt), fmt
    except (UnidentifiedImageError, OSError, ValueError):
        return False, "", ""


def write_manifest(manifest_path: Path, rows: Sequence[dict[str, str]]) -> None:
    fieldnames = ["class_name", "source", "observation_id", "photo_id", "license", "url", "sha256", "split"]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch UK open data for badger/otter classifier training.")
    parser.add_argument("--out", default="dataset_uk_species", help="Output dataset root")
    parser.add_argument("--licenses", default="cc0,cc-by,cc-by-nc", help="Comma-separated iNat photo licenses")
    parser.add_argument("--uk-place-id", type=int, default=6857, help="iNaturalist place ID for UK")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--request-sleep", type=float, default=0.03, help="Sleep between API pages (seconds)")
    parser.add_argument("--per-page", type=int, default=200, help="iNat API per_page value")
    parser.add_argument("--max-observations-per-species", type=int, default=0, help="Optional cap per species; 0 means unlimited")
    parser.add_argument("--targets", default="600,150,150", help="Split targets as train,val,test counts per class")
    parser.add_argument("--min-short-edge", type=int, default=224, help="Minimum short-edge pixels")
    parser.add_argument("--reset", action="store_true", help="Delete existing dataset images and manifest before fetching")
    parser.add_argument("--manifest", default="", help="Manifest path; default is <out>/manifest.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)

    out_root = Path(args.out)
    manifest_path = Path(args.manifest) if args.manifest else out_root / "manifest.csv"
    split_values = parse_int_csv(args.targets, expected_len=3, label="--targets")
    split_targets = {split: split_values[idx] for idx, split in enumerate(SPLITS)}
    licenses = [item.lower() for item in parse_csv(args.licenses)]
    allowed_licenses = set(licenses)

    if args.reset:
        for split in SPLITS:
            split_dir = out_root / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
        if manifest_path.exists():
            manifest_path.unlink()

    ensure_dataset_dirs(out_root)
    existing_rows, seen_hashes, existing_counts = load_existing_manifest(manifest_path)
    manifest_rows: list[dict[str, str]] = list(existing_rows)

    counts = {
        class_name: {split: existing_counts[split][class_name] for split in SPLITS}
        for class_name in DEFAULT_CLASS_TAXA
    }

    print("Starting fetch with settings:")
    print(
        json.dumps(
            {
                "out": str(out_root),
                "licenses": licenses,
                "uk_place_id": args.uk_place_id,
                "targets": split_targets,
                "min_short_edge": args.min_short_edge,
                "reset": args.reset,
            },
            indent=2,
        )
    )

    class_observations: Dict[str, List[ObservationRecord]] = {name: [] for name in DEFAULT_CLASS_TAXA}
    for class_name, taxa in DEFAULT_CLASS_TAXA.items():
        for species_name, taxon_id in taxa:
            observations = fetch_observations_for_taxon(
                taxon_id=taxon_id,
                place_id=args.uk_place_id,
                licenses_csv=",".join(licenses),
                per_page=args.per_page,
                max_observations=args.max_observations_per_species,
                request_sleep=args.request_sleep,
            )
            grouped = build_observation_records(
                class_name=class_name,
                species_name=species_name,
                observations=observations,
                allowed_licenses=allowed_licenses,
            )
            class_observations[class_name].extend(grouped)
            photo_total = sum(len(item.photos) for item in grouped)
            print(
                f"Collected {len(grouped)} observations / {photo_total} photos "
                f"for class={class_name} species={species_name}"
            )

    for class_name in class_observations:
        class_observations[class_name] = interleave_observations(class_observations[class_name], rng=rng)

    downloaded = 0
    skipped_duplicates = 0
    skipped_quality = 0
    skipped_download_errors = 0

    for class_name, observations in class_observations.items():
        rng.shuffle(observations)
        for obs in observations:
            split = choose_target_split(counts[class_name], split_targets=split_targets)
            if split is None:
                break

            class_dir = out_root / split / class_name
            saved_from_observation = 0
            for photo in obs.photos:
                if counts[class_name][split] >= split_targets[split]:
                    break

                try:
                    content = download_bytes(photo.url)
                except RuntimeError:
                    skipped_download_errors += 1
                    continue

                digest = sha256_bytes(content)
                if digest in seen_hashes:
                    skipped_duplicates += 1
                    continue

                ok, ext, _fmt = validate_image_content(content, min_short_edge=args.min_short_edge)
                if not ok:
                    skipped_quality += 1
                    continue

                file_name = f"{photo.observation_id}_{photo.photo_id}_{digest[:12]}{ext}"
                output_path = class_dir / file_name
                output_path.write_bytes(content)

                seen_hashes.add(digest)
                downloaded += 1
                saved_from_observation += 1
                counts[class_name][split] += 1

                manifest_rows.append(
                    {
                        "class_name": class_name,
                        "source": "inaturalist",
                        "observation_id": str(photo.observation_id),
                        "photo_id": str(photo.photo_id),
                        "license": photo.license_code,
                        "url": photo.url,
                        "sha256": digest,
                        "split": split,
                    }
                )

            if saved_from_observation == 0:
                continue

    write_manifest(manifest_path, manifest_rows)

    print("Fetch complete.")
    print(f"Downloaded new images: {downloaded}")
    print(f"Skipped duplicate images: {skipped_duplicates}")
    print(f"Skipped low-quality/corrupt images: {skipped_quality}")
    print(f"Skipped download errors: {skipped_download_errors}")
    print("Per-class split counts:")
    print(json.dumps(counts, indent=2))

    unmet = []
    for class_name in DEFAULT_CLASS_TAXA:
        for split in SPLITS:
            if counts[class_name][split] < split_targets[split]:
                unmet.append(f"{class_name}/{split}: {counts[class_name][split]} < {split_targets[split]}")

    if unmet:
        print("WARNING: Some split targets were not met.")
        for row in unmet:
            print(f"- {row}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


### app.py

```python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Gradio Demo for image detection"""

import os
import torch
import shutil
import cv2
import supervision as sv
import gradio as gr
from zipfile import ZipFile
from torch.utils.data import DataLoader
import numpy as np
import ast


def patch_yolov5_scale_coords():
    """Keep PytorchWildlife 1.2.0 compatible with newer yolov5 packages."""
    try:
        from yolov5.utils import general as yolov5_general
    except Exception:
        return

    if not hasattr(yolov5_general, "scale_coords") and hasattr(yolov5_general, "scale_boxes"):
        yolov5_general.scale_coords = yolov5_general.scale_boxes


patch_yolov5_scale_coords()

from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.data import datasets as pw_data
from wildlife_classifier import (
    WildlifeClassificationAdapter,
    format_species_detection_label,
)
from speciesnet_adapter import SpeciesNetClassificationAdapter

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Annotators
dot_annotator = sv.DotAnnotator(radius=6)
box_annotator = sv.BoxAnnotator(thickness=4)

# Ensure temp folder exists
os.makedirs(os.path.join("..", "temp"), exist_ok=True)

# Global models
detection_model = None
classification_model = None

WILDLIFE_MODEL_PATH = "wildlife_model.pth"


def short_error(exc):
    return str(exc).splitlines()[0]


def detection_class_name(det_id):
    class_names = getattr(detection_model, "CLASS_NAMES", {})
    try:
        det_id = int(det_id)
    except (TypeError, ValueError):
        return ""

    if isinstance(class_names, dict):
        return str(class_names.get(det_id, class_names.get(str(det_id), "")))
    if isinstance(class_names, (list, tuple)) and 0 <= det_id < len(class_names):
        return str(class_names[det_id])
    return ""


def is_animal_detection(det_id, default_label):
    candidates = [
        detection_class_name(det_id),
        str(default_label).split()[0] if default_label is not None else "",
    ]
    return any(candidate.strip().lower() == "animal" for candidate in candidates)


def load_models(det, version, clf, wpath=None, wclass=None):
    global detection_model, classification_model
    classifier_status = "No classifier"

    # Load detection model
    if det != "None":
        if det == "HerdNet General":
            detection_model = pw_detection.HerdNet(device=DEVICE)
        elif det == "HerdNet Ennedi":
            detection_model = pw_detection.HerdNet(device=DEVICE, version="ennedi")
        else:
            detection_model = pw_detection.__dict__[det](
                device=DEVICE, pretrained=True, version=version
            )
    else:
        detection_model = None
        return "NO MODEL LOADED!!"

    # Load classification model
    if clf != "None":
        if clf == "CustomWeights":
            try:
                if not wpath or not wclass:
                    raise ValueError("CustomWeights requires both weights path and class mapping.")
                wclass = ast.literal_eval(wclass)
                classification_model = pw_classification.__dict__[clf](
                    weights=wpath, class_names=wclass, device=DEVICE
                )
                classifier_status = "CustomWeights"
            except Exception as e:
                print(f"CustomWeights could not be loaded: {e}")
                classification_model = None
                classifier_status = f"CustomWeights failed: {short_error(e)}"
        elif clf == "WildlifeModel":
            try:
                weights_path = wpath.strip() if isinstance(wpath, str) and wpath.strip() else WILDLIFE_MODEL_PATH
                classification_model = WildlifeClassificationAdapter(
                    weights_path=weights_path,
                    device=DEVICE,
                )
                print(f"Wildlife model successfully loaded from {weights_path}.")
                recommended_thr = classification_model.metrics.get("recommended_clf_conf_threshold")
                if isinstance(recommended_thr, (float, int)):
                    print(f"Recommended classification threshold: {float(recommended_thr):.2f}")
                classifier_status = f"WildlifeModel ({weights_path})"
            except Exception as e:
                print(f"Wildlife model could not be loaded: {e}")
                classification_model = None
                classifier_status = f"WildlifeModel failed: {short_error(e)}"
        elif clf == "SpeciesNet":
            try:
                model_name = wpath.strip() if isinstance(wpath, str) and wpath.strip() else None
                classification_model = SpeciesNetClassificationAdapter(
                    model_name=model_name,
                    device=DEVICE,
                )
                print(f"SpeciesNet successfully loaded from {classification_model.model_name}.")
                classifier_status = f"SpeciesNet ({classification_model.model_name})"
            except Exception as e:
                print(f"SpeciesNet could not be loaded: {e}")
                classification_model = None
                classifier_status = f"SpeciesNet failed: {short_error(e)}"
        else:
            try:
                classification_model = pw_classification.__dict__[clf](
                    device=DEVICE, pretrained=True
                )
                classifier_status = clf
            except Exception as e:
                print(f"{clf} could not be loaded: {e}")
                classification_model = None
                classifier_status = f"{clf} failed: {short_error(e)}"
    else:
        classification_model = None

    return f"Loaded Detector: {det}. Version: {version}. Classifier: {classifier_status}"

def single_image_detection(input_img, det_conf_thres, clf_conf_thres, img_index=None):
    img = np.array(input_img)
    # Choose annotator & run detection
    if "HerdNet" in detection_model.__class__.__name__:
        annotator = dot_annotator
        results_det = detection_model.single_image_detection(
            img, img_path=img_index,
            det_conf_thres=det_conf_thres,
            clf_conf_thres=clf_conf_thres
        )
    else:
        annotator = box_annotator
        results_det = detection_model.single_image_detection(
            img, img_path=img_index,
            det_conf_thres=det_conf_thres
        )

    default_labels = list(results_det["labels"])
    if classification_model and hasattr(classification_model, "single_image_classification"):
        labels = []
        for xyxy, det_id, default_label in zip(
            results_det["detections"].xyxy,
            results_det["detections"].class_id,
            default_labels,
        ):
            if is_animal_detection(det_id, default_label):
                crop = sv.crop_image(image=img, xyxy=xyxy)
                try:
                    res_clf = classification_model.single_image_classification(crop)
                    labels.append(
                        format_species_detection_label(
                            res_clf,
                            conf_threshold=clf_conf_thres,
                        )
                    )
                except Exception as e:
                    print(f"Classification failed for detection {det_id}: {e}")
                    labels.append(default_label)
            else:
                labels.append(default_label)
    else:
        labels = default_labels

    annotated = annotator.annotate(scene=img, detections=results_det["detections"])
    return draw_labels_within_frame(annotated, results_det["detections"], labels)


def draw_labels_within_frame(scene, detections, labels):
    """Draw labels while clamping them to image boundaries."""
    if len(labels) == 0:
        return scene

    annotated = scene.copy()
    img_h, img_w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = max(0.45, min(0.80, img_w / 1400.0))
    text_thickness = 1 if img_w < 1600 else 2
    padding = 4
    max_chars = 22

    for xyxy, label in zip(detections.xyxy, labels):
        x_min, y_min, _, _ = map(int, xyxy)
        text = str(label)
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."

        (text_w, text_h), baseline = cv2.getTextSize(text, font, text_scale, text_thickness)
        box_w = text_w + (padding * 2)
        box_h = text_h + baseline + (padding * 2)

        x_box = min(max(0, x_min), max(0, img_w - box_w - 1))
        y_box = y_min - box_h - 4
        if y_box < 0:
            y_box = y_min + 4
        y_box = min(max(0, y_box), max(0, img_h - box_h - 1))

        top_left = (x_box, y_box)
        bottom_right = (x_box + box_w, y_box + box_h)
        cv2.rectangle(annotated, top_left, bottom_right, (255, 64, 160), thickness=-1)

        text_x = x_box + padding
        text_y = y_box + padding + text_h
        cv2.putText(
            annotated,
            text,
            (text_x, text_y),
            font,
            text_scale,
            (0, 0, 0),
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated

def batch_detection(zip_file, timelapse, det_conf_thres):
    extract_path = os.path.join("..","temp","zip_upload")
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path)

    json_save = os.path.join(extract_path, "results.json")
    with ZipFile(zip_file.name) as z:
        z.extractall(extract_path)

    files = os.listdir(extract_path)
    tgt = os.path.join(extract_path, files[0]) if len(files)==1 and os.path.isdir(os.path.join(extract_path, files[0])) else extract_path

    if "HerdNet" in detection_model.__class__.__name__:
        det_res = detection_model.batch_image_detection(tgt, batch_size=1, det_conf_thres=det_conf_thres, id_strip=tgt)
    else:
        det_res = detection_model.batch_image_detection(tgt, batch_size=16, det_conf_thres=det_conf_thres, id_strip=tgt)

    classifier_supports_batch = (
        classification_model
        and hasattr(classification_model, "batch_image_classification")
        and callable(getattr(classification_model, "batch_image_classification"))
    )

    if classifier_supports_batch:
        dataset = pw_data.DetectionCrops(
            det_res,
            transform=pw_trans.Classification_Inference_Transform(target_size=224),
            path_head=tgt
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
        clf_res = classification_model.batch_image_classification(loader, id_strip=tgt)
        clf_categories = getattr(classification_model, "CLASS_NAMES", {})

        if timelapse:
            json_save = json_save.replace(".json","_timelapse.json")
            pw_utils.save_detection_classification_timelapse_json(
                det_results=det_res, clf_results=clf_res,
                det_categories=detection_model.CLASS_NAMES,
                clf_categories=clf_categories,
                output_path=json_save
            )
        else:
            pw_utils.save_detection_classification_json(
                det_results=det_res, clf_results=clf_res,
                det_categories=detection_model.CLASS_NAMES,
                clf_categories=clf_categories,
                output_path=json_save
            )
    else:
        if classification_model and not classifier_supports_batch:
            print("Loaded classification model does not support batch processing; skipping classification.")
        if timelapse:
            json_save = json_save.replace(".json","_timelapse.json")
            pw_utils.save_detection_timelapse_json(det_res, json_save, categories=detection_model.CLASS_NAMES)
        elif "HerdNet" in detection_model.__class__.__name__:
            pw_utils.save_detection_json_as_dots(det_res, json_save, categories=detection_model.CLASS_NAMES)
        else:
            pw_utils.save_detection_json(det_res, json_save, categories=detection_model.CLASS_NAMES)

    return json_save

def batch_path_detection(tgt_folder_path, det_conf_thres):
    json_save = os.path.join(tgt_folder_path, "results.json")
    det_res = detection_model.batch_image_detection(tgt_folder_path, det_conf_thres=det_conf_thres, id_strip=tgt_folder_path)
    if "HerdNet" in detection_model.__class__.__name__:
        pw_utils.save_detection_json_as_dots(det_res, json_save, categories=detection_model.CLASS_NAMES)
    else:
        pw_utils.save_detection_json(det_res, json_save, categories=detection_model.CLASS_NAMES)
    return json_save


def video_detection(video, det_conf_thres, clf_conf_thres, target_fps, codec):
    def cb(frame, i):
        return single_image_detection(frame, det_conf_thres, clf_conf_thres, img_index=i)
    out_path = os.path.join("..","temp","video_detection.mp4")
    pw_utils.process_video(source_path=video, target_path=out_path,
                           callback=cb, target_fps=int(target_fps), codec=codec)
    return out_path

with gr.Blocks() as demo:
    gr.Markdown("# Pytorch-Wildlife Demo.")
    with gr.Row():
        det_drop = gr.Dropdown(
            ["None","MegaDetectorV5","MegaDetectorV6","HerdNet General","HerdNet Ennedi"],
            label="Detection model", value="None"
        )
        det_version = gr.Dropdown(
            ["None"], label="Model version", value="None"
        )

    with gr.Column():
        clf_drop = gr.Dropdown(
            ["None","AI4GOpossum","AI4GAmazonRainforest","AI4GSnapshotSerengeti","CustomWeights","WildlifeModel","SpeciesNet"],
            label="Classification model", interactive=True, visible=False, value="None"
        )
        custom_weights_path = gr.Textbox(
            label="Weights / Model Path",
            visible=False,
            placeholder="./weights/my_weight.pt, wildlife_model_badger_otter.pth, or kaggle:google/speciesnet/pyTorch/v4.0.2a/1",
        )
        custom_weights_class = gr.Textbox(label="Custom Weights Class", visible=False, placeholder="{1:'ocelot',2:'cow',3:'bear'}")
        load_but = gr.Button("Load Models!")
        load_out = gr.Text("NO MODEL LOADED!!", label="Loaded models:")

    def update_ui_elements(det_model):
        if det_model=="MegaDetectorV6":
            return (
                gr.Dropdown(
                    choices=["MDV6-yolov9-c","MDV6-yolov9-e","MDV6-yolov10-c","MDV6-yolov10-e","MDV6-rtdetr-c"],
                    label="Model version", value="MDV6-yolov9-e"
                ),
                gr.update(visible=True)
            )
        elif det_model=="MegaDetectorV5":
            return (
                gr.Dropdown(choices=["a","b"], label="Model version", value="a"),
                gr.update(visible=True)
            )
        else:
            return (
                gr.Dropdown(choices=["None"], label="Model version", value="None"),
                gr.update(value="None", visible=False)
            )

    det_drop.change(update_ui_elements, det_drop, [det_version, clf_drop])

    def toggle_textboxes(model):
        if model=="CustomWeights":
            return gr.update(visible=True), gr.update(visible=True)
        if model in {"WildlifeModel", "SpeciesNet"}:
            return gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=False)

    clf_drop.change(toggle_textboxes, clf_drop, [custom_weights_path, custom_weights_class])

    with gr.Tab("Single Image Process"):
        with gr.Row():
            with gr.Column():
                sgl_in = gr.Image(type="pil")
                sgl_conf_sl_det = gr.Slider(0,1,label="Detection Confidence Threshold",value=0.2)
                sgl_conf_sl_clf = gr.Slider(0,1,label="Classification Confidence Threshold",value=0.7)
            sgl_out = gr.Image()
        sgl_but = gr.Button("Detect Animals!")

    with gr.Tab("Folder Separation"):
        with gr.Row():
            with gr.Column():
                inp_path = gr.Textbox(label="Input path", placeholder="./data/")
                out_path = gr.Textbox(label="Output path", placeholder="./output/")
                bth_conf_fs = gr.Slider(0,1,label="Detection Confidence Threshold",value=0.2)
                process_btn = gr.Button("Process Files")
                bth_out2 = gr.File(label="Detection Results JSON.")
                process_but = gr.Button("Separate files")
            process_btn.click(batch_path_detection, inputs=[inp_path,bth_conf_fs], outputs=bth_out2)
            process_but.click(pw_utils.detection_folder_separation,
                              inputs=[bth_out2,inp_path,out_path,bth_conf_fs],
                              outputs=out_path)

    with gr.Tab("Batch Image Process"):
        with gr.Row():
            with gr.Column():
                bth_in = gr.File(label="Upload zip file.")
                chck_timelapse = gr.Checkbox(label="Generate timelapse JSON", visible=False)
                bth_conf_sl = gr.Slider(0,1,label="Detection Confidence Threshold",value=0.2)
            bth_out = gr.File(label="Detection Results JSON.")
        bth_but = gr.Button("Detect Animals!")
        bth_but.click(batch_detection, inputs=[bth_in,chck_timelapse,bth_conf_sl], outputs=bth_out)

    with gr.Tab("Single Video Process"):
        with gr.Row():
            with gr.Column():
                vid_in = gr.Video()
                vid_conf_sl_det = gr.Slider(0,1,label="Detection Confidence Threshold",value=0.2)
                vid_conf_sl_clf = gr.Slider(0,1,label="Classification Confidence Threshold",value=0.7)
                vid_fr = gr.Dropdown([5,10,30],label="Output video framerate",value=30)
                vid_enc = gr.Dropdown(["mp4v","avc1"],label="Video encoder",value="mp4v")
            vid_out = gr.Video()
        vid_but = gr.Button("Detect Animals!")
        vid_but.click(video_detection,
                      inputs=[vid_in,vid_conf_sl_det,vid_conf_sl_clf,vid_fr,vid_enc],
                      outputs=vid_out)

    # Show timelapse checkbox only when detection model is not HerdNet
    det_drop.change(
        lambda m: gr.update(visible=True) if "HerdNet" not in m else gr.update(visible=False),
        det_drop, [chck_timelapse]
    )

    load_but.click(
        load_models,
        inputs=[det_drop, det_version, clf_drop, custom_weights_path, custom_weights_class],
        outputs=load_out
    )
    sgl_but.click(
        single_image_detection,
        inputs=[sgl_in, sgl_conf_sl_det, sgl_conf_sl_clf],
        outputs=sgl_out
    )

# Launch without share (no frpc download)
if __name__ == "__main__":
    demo.launch(share=False)

```json
{
  "class_names": [
    "badger",
    "otter",
    "other"
  ],
  "protected_classes": [
    "badger",
    "otter"
  ],
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.6160574057367113,
      "train_accuracy": 0.7477777777777778,
      "val_loss": 1.0362212737392758,
      "val_accuracy": 0.7088888888888889,
      "val_macro_f1": 0.6996922766795314
    },
    {
      "epoch": 2,
      "train_loss": 0.33564683821466235,
      "train_accuracy": 0.88,
      "val_loss": 0.857354204904744,
      "val_accuracy": 0.7555555555555555,
      "val_macro_f1": 0.7555834755275268
    },
    {
      "epoch": 3,
      "train_loss": 0.2358652087052663,
      "train_accuracy": 0.9088888888888889,
      "val_loss": 1.399649069679306,
      "val_accuracy": 0.6155555555555555,
      "val_macro_f1": 0.5900925729630414
    },
    {
      "epoch": 4,
      "train_loss": 0.21405862390995026,
      "train_accuracy": 0.9233333333333333,
      "val_loss": 0.8323440135491547,
      "val_accuracy": 0.7911111111111111,
      "val_macro_f1": 0.7902095267297291
    },
    {
      "epoch": 5,
      "train_loss": 0.18191443065802257,
      "train_accuracy": 0.9361111111111111,
      "val_loss": 1.8353064099947611,
      "val_accuracy": 0.5777777777777777,
      "val_macro_f1": 0.5582566020781575
    },
    {
      "epoch": 6,
      "train_loss": 0.15067814668019613,
      "train_accuracy": 0.9461111111111111,
      "val_loss": 0.895414263139812,
      "val_accuracy": 0.7755555555555556,
      "val_macro_f1": 0.7755177726192221
    },
    {
      "epoch": 7,
      "train_loss": 0.13393938932153915,
      "train_accuracy": 0.9516666666666667,
      "val_loss": 0.9438581675953305,
      "val_accuracy": 0.7755555555555556,
      "val_macro_f1": 0.7727914852443155
    },
    {
      "epoch": 8,
      "train_loss": 0.15360993249548807,
      "train_accuracy": 0.9477777777777778,
      "val_loss": 0.9890402804479083,
      "val_accuracy": 0.7333333333333333,
      "val_macro_f1": 0.734168083972298
    },
    {
      "epoch": 9,
      "train_loss": 0.12951564672092597,
      "train_accuracy": 0.9527777777777777,
      "val_loss": 0.8586659197840426,
      "val_accuracy": 0.7155555555555555,
      "val_macro_f1": 0.7156730443058629
    },
    {
      "epoch": 10,
      "train_loss": 0.08241021553675333,
      "train_accuracy": 0.9711111111111111,
      "val_loss": 1.0344887667232054,
      "val_accuracy": 0.8088888888888889,
      "val_macro_f1": 0.8051729129797346
    },
    {
      "epoch": 11,
      "train_loss": 0.0663977604607741,
      "train_accuracy": 0.9744444444444444,
      "val_loss": 0.9402235433790419,
      "val_accuracy": 0.7977777777777778,
      "val_macro_f1": 0.7982322886445609
    },
    {
      "epoch": 12,
      "train_loss": 0.10183024727221993,
      "train_accuracy": 0.9644444444444444,
      "val_loss": 0.7384178255627759,
      "val_accuracy": 0.7777777777777778,
      "val_macro_f1": 0.7768464076921787
    },
    {
      "epoch": 13,
      "train_loss": 0.0801923600004779,
      "train_accuracy": 0.9727777777777777,
      "val_loss": 0.8403355153653279,
      "val_accuracy": 0.8066666666666666,
      "val_macro_f1": 0.8063962170018647
    },
    {
      "epoch": 14,
      "train_loss": 0.13405492826882337,
      "train_accuracy": 0.9533333333333334,
      "val_loss": 1.3810001115262922,
      "val_accuracy": 0.74,
      "val_macro_f1": 0.7367360161163355
    },
    {
      "epoch": 15,
      "train_loss": 0.07916151005774737,
      "train_accuracy": 0.9738888888888889,
      "val_loss": 1.0277823601710487,
      "val_accuracy": 0.7533333333333333,
      "val_macro_f1": 0.7529439356211008
    },
    {
      "epoch": 16,
      "train_loss": 0.10289710526665052,
      "train_accuracy": 0.9633333333333334,
      "val_loss": 1.0149892131684142,
      "val_accuracy": 0.7622222222222222,
      "val_macro_f1": 0.757828668439601
    },
    {
      "epoch": 17,
      "train_loss": 0.09687759137815899,
      "train_accuracy": 0.9661111111111111,
      "val_loss": 0.7524456463712785,
      "val_accuracy": 0.8,
      "val_macro_f1": 0.7982023118684151
    },
    {
      "epoch": 18,
      "train_loss": 0.049787297683457535,
      "train_accuracy": 0.9833333333333333,
      "val_loss": 0.9547708294113868,
      "val_accuracy": 0.7666666666666667,
      "val_macro_f1": 0.764466383752208
    },
    {
      "epoch": 19,
      "train_loss": 0.05953237006337279,
      "train_accuracy": 0.98,
      "val_loss": 0.8782244878796498,
      "val_accuracy": 0.8155555555555556,
      "val_macro_f1": 0.8147714387912545
    },
    {
      "epoch": 20,
      "train_loss": 0.03943180157492558,
      "train_accuracy": 0.9855555555555555,
      "val_loss": 0.704294839434054,
      "val_accuracy": 0.8133333333333334,
      "val_macro_f1": 0.8131643229682446
    },
    {
      "epoch": 21,
      "train_loss": 0.031221856972099178,
      "train_accuracy": 0.99,
      "val_loss": 1.093777867419461,
      "val_accuracy": 0.7688888888888888,
      "val_macro_f1": 0.7682201008086939
    },
    {
      "epoch": 22,
      "train_loss": 0.047382485018008284,
      "train_accuracy": 0.9844444444444445,
      "val_loss": 0.8492364401287361,
      "val_accuracy": 0.8266666666666667,
      "val_macro_f1": 0.8271153846153846
    },
    {
      "epoch": 23,
      "train_loss": 0.09015533473342657,
      "train_accuracy": 0.9727777777777777,
      "val_loss": 0.9624174689960071,
      "val_accuracy": 0.7622222222222222,
      "val_macro_f1": 0.7615062897833225
    },
    {
      "epoch": 24,
      "train_loss": 0.05440158608473009,
      "train_accuracy": 0.9827777777777778,
      "val_loss": 0.9973731504546272,
      "val_accuracy": 0.78,
      "val_macro_f1": 0.7792040454481746
    },
    {
      "epoch": 25,
      "train_loss": 0.04977361223246488,
      "train_accuracy": 0.9822222222222222,
      "val_loss": 0.8712971927722295,
      "val_accuracy": 0.8133333333333334,
      "val_macro_f1": 0.8124397822035684
    },
    {
      "epoch": 26,
      "train_loss": 0.06071143476292491,
      "train_accuracy": 0.9833333333333333,
      "val_loss": 0.8159345576498244,
      "val_accuracy": 0.7888888888888889,
      "val_macro_f1": 0.7879727989699018
    },
    {
      "epoch": 27,
      "train_loss": 0.05420496744414171,
      "train_accuracy": 0.9794444444444445,
      "val_loss": 1.149332213655693,
      "val_accuracy": 0.8022222222222222,
      "val_macro_f1": 0.8017233077982963
    },
    {
      "epoch": 28,
      "train_loss": 0.057464405941621714,
      "train_accuracy": 0.9788888888888889,
      "val_loss": 0.81989846309026,
      "val_accuracy": 0.7933333333333333,
      "val_macro_f1": 0.7932770343843009
    },
    {
      "epoch": 29,
      "train_loss": 0.05194110080185864,
      "train_accuracy": 0.985,
      "val_loss": 1.048538908428616,
      "val_accuracy": 0.8,
      "val_macro_f1": 0.7993858803399391
    },
    {
      "epoch": 30,
      "train_loss": 0.10218092448388537,
      "train_accuracy": 0.9694444444444444,
      "val_loss": 1.127093861123754,
      "val_accuracy": 0.7666666666666667,
      "val_macro_f1": 0.7642146959189614
    }
  ],
  "val": {
    "accuracy": 0.8266666666666667,
    "macro_f1": 0.8271153846153846,
    "macro_precision": 0.829364823760959,
    "macro_recall": 0.8266666666666667,
    "per_class": {
      "badger": {
        "precision": 0.8913043478260869,
        "recall": 0.82,
        "f1": 0.8541666666666666,
        "support": 150
      },
      "otter": {
        "precision": 0.8066666666666666,
        "recall": 0.8066666666666666,
        "f1": 0.8066666666666665,
        "support": 150
      },
      "other": {
        "precision": 0.7901234567901234,
        "recall": 0.8533333333333334,
        "f1": 0.8205128205128205,
        "support": 150
      }
    },
    "confusion_matrix": [
      [
        123,
        12,
        15
      ],
      [
        10,
        121,
        19
      ],
      [
        5,
        17,
        128
      ]
    ],
    "loss": 0.8492364401287361
  },
  "test": {
    "accuracy": 0.7555555555555555,
    "macro_f1": 0.7559535888863967,
    "macro_precision": 0.7577266983723577,
    "macro_recall": 0.7555555555555555,
    "per_class": {
      "badger": {
        "precision": 0.8115942028985508,
        "recall": 0.7466666666666667,
        "f1": 0.7777777777777779,
        "support": 150
      },
      "otter": {
        "precision": 0.7295597484276729,
        "recall": 0.7733333333333333,
        "f1": 0.750809061488673,
        "support": 150
      },
      "other": {
        "precision": 0.7320261437908496,
        "recall": 0.7466666666666667,
        "f1": 0.7392739273927392,
        "support": 150
      }
    },
    "confusion_matrix": [
      [
        112,
        25,
        13
      ],
      [
        6,
        116,
        28
      ],
      [
        20,
        18,
        112
      ]
    ],
    "loss": 1.0528578612953425
  },
  "threshold_calibration": {
    "recommended_threshold": 0.7899999999999999,
    "strategy": "precision_constrained",
    "precision_target": 0.85,
    "candidates": [
      {
        "threshold": 0.3,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.31,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.32,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.32999999999999996,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.33999999999999997,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.35,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.36,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.37,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.38,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.38999999999999996,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.39999999999999997,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.41,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.42,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.42999999999999994,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.43999999999999995,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.44999999999999996,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.45999999999999996,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.47,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.48,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.49,
        "mean_precision": 0.8489855072463768,
        "mean_recall": 0.8133333333333332,
        "protected_precision": {
          "badger": 0.8913043478260869,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.82,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.49999999999999994,
        "mean_precision": 0.8485888077858881,
        "mean_recall": 0.81,
        "protected_precision": {
          "badger": 0.8905109489051095,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.8133333333333334,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.51,
        "mean_precision": 0.8485888077858881,
        "mean_recall": 0.81,
        "protected_precision": {
          "badger": 0.8905109489051095,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.8133333333333334,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.52,
        "mean_precision": 0.8485888077858881,
        "mean_recall": 0.81,
        "protected_precision": {
          "badger": 0.8905109489051095,
          "otter": 0.8066666666666666
        },
        "protected_recall": {
          "badger": 0.8133333333333334,
          "otter": 0.8066666666666666
        }
      },
      {
        "threshold": 0.5299999999999999,
        "mean_precision": 0.850258346581876,
        "mean_recall": 0.8033333333333333,
        "protected_precision": {
          "badger": 0.8897058823529411,
          "otter": 0.8108108108108109
        },
        "protected_recall": {
          "badger": 0.8066666666666666,
          "otter": 0.8
        }
      },
      {
        "threshold": 0.5399999999999999,
        "mean_precision": 0.8498498498498499,
        "mean_recall": 0.8,
        "protected_precision": {
          "badger": 0.8888888888888888,
          "otter": 0.8108108108108109
        },
        "protected_recall": {
          "badger": 0.8,
          "otter": 0.8
        }
      },
      {
        "threshold": 0.5499999999999999,
        "mean_precision": 0.8526077097505669,
        "mean_recall": 0.8,
        "protected_precision": {
          "badger": 0.8888888888888888,
          "otter": 0.8163265306122449
        },
        "protected_recall": {
          "badger": 0.8,
          "otter": 0.8
        }
      },
      {
        "threshold": 0.5599999999999999,
        "mean_precision": 0.8526077097505669,
        "mean_recall": 0.8,
        "protected_precision": {
          "badger": 0.8888888888888888,
          "otter": 0.8163265306122449
        },
        "protected_recall": {
          "badger": 0.8,
          "otter": 0.8
        }
      },
      {
        "threshold": 0.57,
        "mean_precision": 0.8526077097505669,
        "mean_recall": 0.8,
        "protected_precision": {
          "badger": 0.8888888888888888,
          "otter": 0.8163265306122449
        },
        "protected_recall": {
          "badger": 0.8,
          "otter": 0.8
        }
      },
      {
        "threshold": 0.58,
        "mean_precision": 0.8526077097505669,
        "mean_recall": 0.8,
        "protected_precision": {
          "badger": 0.8888888888888888,
          "otter": 0.8163265306122449
        },
        "protected_recall": {
          "badger": 0.8,
          "otter": 0.8
        }
      },
      {
        "threshold": 0.59,
        "mean_precision": 0.8519786910197868,
        "mean_recall": 0.7966666666666666,
        "protected_precision": {
          "badger": 0.8888888888888888,
          "otter": 0.815068493150685
        },
        "protected_recall": {
          "badger": 0.8,
          "otter": 0.7933333333333333
        }
      },
      {
        "threshold": 0.5999999999999999,
        "mean_precision": 0.8511432691317334,
        "mean_recall": 0.79,
        "protected_precision": {
          "badger": 0.8872180451127819,
          "otter": 0.815068493150685
        },
        "protected_recall": {
          "badger": 0.7866666666666666,
          "otter": 0.7933333333333333
        }
      },
      {
        "threshold": 0.6099999999999999,
        "mean_precision": 0.8561964351438036,
        "mean_recall": 0.7866666666666666,
        "protected_precision": {
          "badger": 0.8872180451127819,
          "otter": 0.8251748251748252
        },
        "protected_recall": {
          "badger": 0.7866666666666666,
          "otter": 0.7866666666666666
        }
      },
      {
        "threshold": 0.6199999999999999,
        "mean_precision": 0.8624626547161758,
        "mean_recall": 0.7866666666666666,
        "protected_precision": {
          "badger": 0.8939393939393939,
          "otter": 0.8309859154929577
        },
        "protected_recall": {
          "badger": 0.7866666666666666,
          "otter": 0.7866666666666666
        }
      },
      {
        "threshold": 0.6299999999999999,
        "mean_precision": 0.8620578432426621,
        "mean_recall": 0.7833333333333333,
        "protected_precision": {
          "badger": 0.8931297709923665,
          "otter": 0.8309859154929577
        },
        "protected_recall": {
          "badger": 0.78,
          "otter": 0.7866666666666666
        }
      },
      {
        "threshold": 0.6399999999999999,
        "mean_precision": 0.8654929577464789,
        "mean_recall": 0.7833333333333333,
        "protected_precision": {
          "badger": 0.9,
          "otter": 0.8309859154929577
        },
        "protected_recall": {
          "badger": 0.78,
          "otter": 0.7866666666666666
        }
      },
      {
        "threshold": 0.6499999999999999,
        "mean_precision": 0.8680521194128319,
        "mean_recall": 0.78,
        "protected_precision": {
          "badger": 0.8992248062015504,
          "otter": 0.8368794326241135
        },
        "protected_recall": {
          "badger": 0.7733333333333333,
          "otter": 0.7866666666666666
        }
      },
      {
        "threshold": 0.6599999999999999,
        "mean_precision": 0.867469545957918,
        "mean_recall": 0.7766666666666666,
        "protected_precision": {
          "badger": 0.8992248062015504,
          "otter": 0.8357142857142857
        },
        "protected_recall": {
          "badger": 0.7733333333333333,
          "otter": 0.78
        }
      },
      {
        "threshold": 0.6699999999999999,
        "mean_precision": 0.8668785901511349,
        "mean_recall": 0.7733333333333333,
        "protected_precision": {
          "badger": 0.8992248062015504,
          "otter": 0.8345323741007195
        },
        "protected_recall": {
          "badger": 0.7733333333333333,
          "otter": 0.7733333333333333
        }
      },
      {
        "threshold": 0.6799999999999999,
        "mean_precision": 0.8656788854630582,
        "mean_recall": 0.7633333333333333,
        "protected_precision": {
          "badger": 0.8968253968253969,
          "otter": 0.8345323741007195
        },
        "protected_recall": {
          "badger": 0.7533333333333333,
          "otter": 0.7733333333333333
        }
      },
      {
        "threshold": 0.69,
        "mean_precision": 0.8652661870503597,
        "mean_recall": 0.76,
        "protected_precision": {
          "badger": 0.896,
          "otter": 0.8345323741007195
        },
        "protected_recall": {
          "badger": 0.7466666666666667,
          "otter": 0.7733333333333333
        }
      },
      {
        "threshold": 0.7,
        "mean_precision": 0.8652661870503597,
        "mean_recall": 0.76,
        "protected_precision": {
          "badger": 0.896,
          "otter": 0.8345323741007195
        },
        "protected_recall": {
          "badger": 0.7466666666666667,
          "otter": 0.7733333333333333
        }
      },
      {
        "threshold": 0.71,
        "mean_precision": 0.8652661870503597,
        "mean_recall": 0.76,
        "protected_precision": {
          "badger": 0.896,
          "otter": 0.8345323741007195
        },
        "protected_recall": {
          "badger": 0.7466666666666667,
          "otter": 0.7733333333333333
        }
      },
      {
        "threshold": 0.72,
        "mean_precision": 0.8678861788617886,
        "mean_recall": 0.7533333333333334,
        "protected_precision": {
          "badger": 0.9024390243902439,
          "otter": 0.8333333333333334
        },
        "protected_recall": {
          "badger": 0.74,
          "otter": 0.7666666666666667
        }
      },
      {
        "threshold": 0.73,
        "mean_precision": 0.8740356798457087,
        "mean_recall": 0.75,
        "protected_precision": {
          "badger": 0.9098360655737705,
          "otter": 0.8382352941176471
        },
        "protected_recall": {
          "badger": 0.74,
          "otter": 0.76
        }
      },
      {
        "threshold": 0.74,
        "mean_precision": 0.8771402550091074,
        "mean_recall": 0.75,
        "protected_precision": {
          "badger": 0.9098360655737705,
          "otter": 0.8444444444444444
        },
        "protected_recall": {
          "badger": 0.74,
          "otter": 0.76
        }
      },
      {
        "threshold": 0.75,
        "mean_precision": 0.8797300628620732,
        "mean_recall": 0.7466666666666666,
        "protected_precision": {
          "badger": 0.9098360655737705,
          "otter": 0.849624060150376
        },
        "protected_recall": {
          "badger": 0.74,
          "otter": 0.7533333333333333
        }
      },
      {
        "threshold": 0.7599999999999999,
        "mean_precision": 0.8831453634085213,
        "mean_recall": 0.7433333333333333,
        "protected_precision": {
          "badger": 0.9166666666666666,
          "otter": 0.849624060150376
        },
        "protected_recall": {
          "badger": 0.7333333333333333,
          "otter": 0.7533333333333333
        }
      },
      {
        "threshold": 0.7699999999999999,
        "mean_precision": 0.886427298192004,
        "mean_recall": 0.74,
        "protected_precision": {
          "badger": 0.9243697478991597,
          "otter": 0.8484848484848485
        },
        "protected_recall": {
          "badger": 0.7333333333333333,
          "otter": 0.7466666666666667
        }
      },
      {
        "threshold": 0.7799999999999999,
        "mean_precision": 0.8858489960869844,
        "mean_recall": 0.7366666666666666,
        "protected_precision": {
          "badger": 0.9243697478991597,
          "otter": 0.8473282442748091
        },
        "protected_recall": {
          "badger": 0.7333333333333333,
          "otter": 0.74
        }
      },
      {
        "threshold": 0.7899999999999999,
        "mean_precision": 0.8891079508726567,
        "mean_recall": 0.7366666666666666,
        "protected_precision": {
          "badger": 0.9243697478991597,
          "otter": 0.8538461538461538
        },
        "protected_recall": {
          "badger": 0.7333333333333333,
          "otter": 0.74
        }
      },
      {
        "threshold": 0.7999999999999999,
        "mean_precision": 0.892458284062541,
        "mean_recall": 0.7333333333333333,
        "protected_precision": {
          "badger": 0.9322033898305084,
          "otter": 0.8527131782945736
        },
        "protected_recall": {
          "badger": 0.7333333333333333,
          "otter": 0.7333333333333333
        }
      },
      {
        "threshold": 0.8099999999999998,
        "mean_precision": 0.892458284062541,
        "mean_recall": 0.7333333333333333,
        "protected_precision": {
          "badger": 0.9322033898305084,
          "otter": 0.8527131782945736
        },
        "protected_recall": {
          "badger": 0.7333333333333333,
          "otter": 0.7333333333333333
        }
      },
      {
        "threshold": 0.8199999999999998,
        "mean_precision": 0.8907048695184288,
        "mean_recall": 0.7233333333333334,
        "protected_precision": {
          "badger": 0.9322033898305084,
          "otter": 0.8492063492063492
        },
        "protected_recall": {
          "badger": 0.7333333333333333,
          "otter": 0.7133333333333334
        }
      },
      {
        "threshold": 0.8299999999999998,
        "mean_precision": 0.8904151404151404,
        "mean_recall": 0.72,
        "protected_precision": {
          "badger": 0.9316239316239316,
          "otter": 0.8492063492063492
        },
        "protected_recall": {
          "badger": 0.7266666666666667,
          "otter": 0.7133333333333334
        }
      },
      {
        "threshold": 0.8399999999999999,
        "mean_precision": 0.8901204159824849,
        "mean_recall": 0.7166666666666667,
        "protected_precision": {
          "badger": 0.9310344827586207,
          "otter": 0.8492063492063492
        },
        "protected_recall": {
          "badger": 0.72,
          "otter": 0.7133333333333334
        }
      },
      {
        "threshold": 0.8499999999999999,
        "mean_precision": 0.8901204159824849,
        "mean_recall": 0.7166666666666667,
        "protected_precision": {
          "badger": 0.9310344827586207,
          "otter": 0.8492063492063492
        },
        "protected_recall": {
          "badger": 0.72,
          "otter": 0.7133333333333334
        }
      },
      {
        "threshold": 0.8599999999999999,
        "mean_precision": 0.8892173913043477,
        "mean_recall": 0.71,
        "protected_precision": {
          "badger": 0.9304347826086956,
          "otter": 0.848
        },
        "protected_recall": {
          "badger": 0.7133333333333334,
          "otter": 0.7066666666666667
        }
      },
      {
        "threshold": 0.8699999999999999,
        "mean_precision": 0.8926367461430575,
        "mean_recall": 0.71,
        "protected_precision": {
          "badger": 0.9304347826086956,
          "otter": 0.8548387096774194
        },
        "protected_recall": {
          "badger": 0.7133333333333334,
          "otter": 0.7066666666666667
        }
      },
      {
        "threshold": 0.8799999999999999,
        "mean_precision": 0.896127513906718,
        "mean_recall": 0.7066666666666667,
        "protected_precision": {
          "badger": 0.9385964912280702,
          "otter": 0.8536585365853658
        },
        "protected_recall": {
          "badger": 0.7133333333333334,
          "otter": 0.7
        }
      },
      {
        "threshold": 0.8899999999999999,
        "mean_precision": 0.89375,
        "mean_recall": 0.69,
        "protected_precision": {
          "badger": 0.9375,
          "otter": 0.85
        },
        "protected_recall": {
          "badger": 0.7,
          "otter": 0.68
        }
      },
      {
        "threshold": 0.8999999999999999,
        "mean_precision": 0.89375,
        "mean_recall": 0.69,
        "protected_precision": {
          "badger": 0.9375,
          "otter": 0.85
        },
        "protected_recall": {
          "badger": 0.7,
          "otter": 0.68
        }
      },
      {
        "threshold": 0.9099999999999999,
        "mean_precision": 0.892478813559322,
        "mean_recall": 0.6833333333333333,
        "protected_precision": {
          "badger": 0.9375,
          "otter": 0.847457627118644
        },
        "protected_recall": {
          "badger": 0.7,
          "otter": 0.6666666666666666
        }
      },
      {
        "threshold": 0.9199999999999999,
        "mean_precision": 0.9039968652037618,
        "mean_recall": 0.6799999999999999,
        "protected_precision": {
          "badger": 0.9545454545454546,
          "otter": 0.853448275862069
        },
        "protected_recall": {
          "badger": 0.7,
          "otter": 0.66
        }
      },
      {
        "threshold": 0.9299999999999999,
        "mean_precision": 0.9025020508613617,
        "mean_recall": 0.6633333333333333,
        "protected_precision": {
          "badger": 0.9528301886792453,
          "otter": 0.8521739130434782
        },
        "protected_recall": {
          "badger": 0.6733333333333333,
          "otter": 0.6533333333333333
        }
      },
      {
        "threshold": 0.94,
        "mean_precision": 0.9016290726817042,
        "mean_recall": 0.6566666666666666,
        "protected_precision": {
          "badger": 0.9523809523809523,
          "otter": 0.8508771929824561
        },
        "protected_recall": {
          "badger": 0.6666666666666666,
          "otter": 0.6466666666666666
        }
      },
      {
        "threshold": 0.95,
        "mean_precision": 0.9049767808639271,
        "mean_recall": 0.6433333333333333,
        "protected_precision": {
          "badger": 0.9603960396039604,
          "otter": 0.8495575221238938
        },
        "protected_recall": {
          "badger": 0.6466666666666666,
          "otter": 0.64
        }
      }
    ]
  },
  "recommended_clf_conf_threshold": 0.7899999999999999,
  "val_protected_alerts": {
    "accuracy": 0.7911111111111111,
    "macro_f1": 0.793310592752972,
    "macro_precision": 0.8166192309300299,
    "macro_recall": 0.791111111111111,
    "per_class": {
      "badger": {
        "precision": 0.9243697478991597,
        "recall": 0.7333333333333333,
        "f1": 0.8178438661710038,
        "support": 150
      },
      "otter": {
        "precision": 0.8538461538461538,
        "recall": 0.74,
        "f1": 0.7928571428571429,
        "support": 150
      },
      "unknown": {
        "precision": 0.6716417910447762,
        "recall": 0.9,
        "f1": 0.7692307692307693,
        "support": 150
      }
    },
    "confusion_matrix": [
      [
        110,
        6,
        34
      ],
      [
        7,
        111,
        32
      ],
      [
        2,
        13,
        135
      ]
    ],
    "protected_precision": {
      "badger": 0.9243697478991597,
      "otter": 0.8538461538461538
    },
    "protected_recall": {
      "badger": 0.7333333333333333,
      "otter": 0.74
    },
    "threshold": 0.7899999999999999,
    "labels": [
      "badger",
      "otter",
      "unknown"
    ]
  },
  "test_protected_alerts": {
    "accuracy": 0.7466666666666667,
    "macro_f1": 0.7475426628173846,
    "macro_precision": 0.7606489691674598,
    "macro_recall": 0.7466666666666667,
    "per_class": {
      "badger": {
        "precision": 0.8512396694214877,
        "recall": 0.6866666666666666,
        "f1": 0.7601476014760148,
        "support": 150
      },
      "otter": {
        "precision": 0.7676056338028169,
        "recall": 0.7266666666666667,
        "f1": 0.7465753424657535,
        "support": 150
      },
      "unknown": {
        "precision": 0.6631016042780749,
        "recall": 0.8266666666666667,
        "f1": 0.7359050445103856,
        "support": 150
      }
    },
    "confusion_matrix": [
      [
        103,
        20,
        27
      ],
      [
        5,
        109,
        36
      ],
      [
        13,
        13,
        124
      ]
    ],
    "protected_precision": {
      "badger": 0.8512396694214877,
      "otter": 0.7676056338028169
    },
    "protected_recall": {
      "badger": 0.6866666666666666,
      "otter": 0.7266666666666667
    },
    "threshold": 0.7899999999999999,
    "labels": [
      "badger",
      "otter",
      "unknown"
    ]
  },
  "train_class_counts": {
    "badger": 600,
    "otter": 600,
    "other": 600
  }
}
```text
torch kurulu degil - bu bolum atlandi
```text
dataset_uk_species dizini yok - veri henuz hazirlanmamis

--- demo_data ---
  1db0b447-39ba-4301-b2d6-d104d312847a_crop1_person.JPG
  1db0b447-39ba-4301-b2d6-d104d312847a_crop2_person.JPG
  1db0b447-39ba-4301-b2d6-d104d312847a_crop3_person.JPG
  1db0b447-39ba-4301-b2d6-d104d312847a_crop4_person.JPG
  1db0b447-39ba-4301-b2d6-d104d312847a_crop5_person.JPG
  1db0b447-39ba-4301-b2d6-d104d312847a.JPG
  AdobeStock_139807715.jpeg
  AdobeStock_167520209_crop1_animal.jpeg
  AdobeStock_167520209.jpeg
  AdobeStock_224942592_crop1_animal.jpeg
  AdobeStock_224942592_crop2_animal.jpeg
  AdobeStock_224942592.jpeg
  AdobeStock_225788868.jpg
  AdobeStock_227824000.jpeg
  AdobeStock_254560450.jpeg
  AdobeStock_271171992.jpeg
  AdobeStock_297069027_crop1_animal.jpeg
  AdobeStock_297069027.jpeg
  AdobeStock_310204443.jpeg
  AdobeStock_423921874_crop1_animal.jpeg

--- reports ---
  dataset_validation.json
