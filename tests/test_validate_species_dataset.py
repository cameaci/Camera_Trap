import tempfile
import unittest
from pathlib import Path

from PIL import Image

from validate_species_dataset import validate_dataset


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


class TestValidateSpeciesDataset(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.classes = ["badger", "otter", "other"]
        self.splits = ["train", "val", "test"]

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_valid_layout(self):
        for split_idx, split in enumerate(self.splits):
            for class_idx, class_name in enumerate(self.classes):
                _write_image(
                    self.root / split / class_name / f"{class_name}_{split}.jpg",
                    color=(split_idx * 40 + class_idx * 10 + 1, 20, 30),
                )

    def test_missing_class_directory_fails(self):
        self._create_valid_layout()
        for file_path in (self.root / "val" / "otter").glob("*"):
            file_path.unlink()
        (self.root / "val" / "otter").rmdir()
        result = validate_dataset(
            data_root=self.root,
            required_classes=self.classes,
            expected_splits=self.splits,
            min_counts={"train": 1, "val": 1, "test": 1},
            check_corrupt=False,
        )
        self.assertFalse(result.ok)
        self.assertTrue(any("Missing class directory" in error for error in result.errors))

    def test_duplicate_across_splits_fails(self):
        self._create_valid_layout()
        duplicate = self.root / "train" / "badger" / "dup.jpg"
        _write_image(duplicate, color=(100, 100, 100))
        copied = self.root / "val" / "badger" / "dup_copy.jpg"
        copied.write_bytes(duplicate.read_bytes())

        result = validate_dataset(
            data_root=self.root,
            required_classes=self.classes,
            expected_splits=self.splits,
            min_counts={"train": 1, "val": 1, "test": 1},
            check_corrupt=False,
        )
        self.assertFalse(result.ok)
        self.assertGreater(len(result.duplicates_across_splits), 0)

    def test_valid_dataset_passes(self):
        for split_idx, split in enumerate(self.splits):
            for class_idx, class_name in enumerate(self.classes):
                _write_image(
                    self.root / split / class_name / f"{class_name}_{split}.jpg",
                    color=(class_idx * 50 + split_idx * 7 + 3, 20, 30),
                )

        result = validate_dataset(
            data_root=self.root,
            required_classes=self.classes,
            expected_splits=self.splits,
            min_counts={"train": 1, "val": 1, "test": 1},
            check_corrupt=True,
        )
        self.assertTrue(result.ok, msg=result.errors)


if __name__ == "__main__":
    unittest.main()
