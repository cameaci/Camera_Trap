"""Validate dataset integrity for UK species classification."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ValidationResult:
    """Collects validation findings and summary information."""

    counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duplicates_across_splits: List[str] = field(default_factory=list)
    checked_images: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "counts": self.counts,
            "errors": self.errors,
            "warnings": self.warnings,
            "duplicates_across_splits": self.duplicates_across_splits,
            "checked_images": self.checked_images,
        }


def _iter_images(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("**/*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_image_readable(path: Path) -> Tuple[bool, str]:
    try:
        with Image.open(path) as img:
            img.verify()
        # reopen to ensure file can be decoded, not only header-validated
        with Image.open(path) as img:
            img.convert("RGB")
        return True, ""
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, str(exc)


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_dataset(
    data_root: str | Path,
    required_classes: Sequence[str],
    expected_splits: Sequence[str] = ("train", "val", "test"),
    min_counts: Dict[str, int] | None = None,
    check_corrupt: bool = True,
) -> ValidationResult:
    """Validate required layout, minimum counts, corrupt files, and split leakage."""
    data_root = Path(data_root)
    min_counts = min_counts or {}
    result = ValidationResult()

    if not data_root.exists():
        result.errors.append(f"Dataset root does not exist: {data_root}")
        return result

    content_hashes: Dict[str, Tuple[str, Path]] = {}

    for split in expected_splits:
        split_dir = data_root / split
        if not split_dir.exists():
            result.errors.append(f"Missing split directory: {split_dir}")
            continue

        result.counts[split] = {}
        for class_name in required_classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                result.errors.append(f"Missing class directory: {class_dir}")
                result.counts[split][class_name] = 0
                continue

            images = list(_iter_images(class_dir))
            count = len(images)
            result.counts[split][class_name] = count

            min_required = min_counts.get(split)
            if min_required is not None and count < min_required:
                result.errors.append(
                    f"{split}/{class_name} has {count} images; expected at least {min_required}"
                )

            for image_path in images:
                result.checked_images += 1
                if check_corrupt:
                    ok, message = _check_image_readable(image_path)
                    if not ok:
                        result.errors.append(f"Unreadable image: {image_path} ({message})")
                        continue

                digest = _sha256(image_path)
                seen = content_hashes.get(digest)
                if seen is not None:
                    seen_split, seen_path = seen
                    if seen_split != split:
                        duplicate_message = (
                            "Duplicate image content across splits: "
                            f"{seen_path} ({seen_split}) == {image_path} ({split})"
                        )
                        result.duplicates_across_splits.append(duplicate_message)
                        result.errors.append(duplicate_message)
                else:
                    content_hashes[digest] = (split, image_path)

    for split in expected_splits:
        if split in result.counts:
            total = sum(result.counts[split].values())
            if total == 0:
                result.warnings.append(f"Split has no images: {split}")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate UK species dataset integrity.")
    parser.add_argument("--data-root", default="dataset_uk_species", help="Dataset root path")
    parser.add_argument(
        "--classes",
        default="badger,otter,other",
        help="Comma-separated required class names",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated split names",
    )
    parser.add_argument("--min-train", type=int, default=400, help="Minimum images per class in train split")
    parser.add_argument("--min-val", type=int, default=100, help="Minimum images per class in val split")
    parser.add_argument("--min-test", type=int, default=100, help="Minimum images per class in test split")
    parser.add_argument(
        "--skip-corrupt-check",
        action="store_true",
        help="Skip opening/decoding images for corruption checks",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional path to write JSON summary",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    required_classes = _parse_csv(args.classes)
    expected_splits = _parse_csv(args.splits)
    min_counts = {
        "train": args.min_train,
        "val": args.min_val,
        "test": args.min_test,
    }

    result = validate_dataset(
        data_root=args.data_root,
        required_classes=required_classes,
        expected_splits=expected_splits,
        min_counts=min_counts,
        check_corrupt=not args.skip_corrupt_check,
    )

    print("Dataset validation summary:")
    print(json.dumps(result.to_dict(), indent=2))

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {output_path}")

    if result.ok:
        print("Validation passed.")
        return 0

    print("Validation failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

