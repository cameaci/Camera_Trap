import tempfile
import unittest
from pathlib import Path

from PIL import Image

try:
    import torch
    import torch.nn as nn

    from wildlife_classifier import (
        DEFAULT_NORMALIZATION,
        WildlifeClassificationAdapter,
        build_classifier_model,
    )

    TORCH_AVAILABLE = True
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    TORCH_AVAILABLE = False
    IMPORT_ERROR = exc


if TORCH_AVAILABLE:
    class TinyLegacyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(3, 3)
            self.CLASS_NAMES = ["badger", "otter", "other"]

        def forward(self, x):
            x = self.pool(x).view(x.size(0), -1)
            return self.fc(x)


@unittest.skipUnless(TORCH_AVAILABLE, f"torch/torchvision not available: {IMPORT_ERROR}")
class TestWildlifeClassifierAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.image = Image.new("RGB", (224, 224), color=(123, 45, 67))

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_loads_legacy_full_model(self):
        model = TinyLegacyModel()
        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.copy_(torch.tensor([2.0, 0.0, -2.0]))

        model_path = self.root / "legacy_model.pth"
        torch.save(model, model_path)

        adapter = WildlifeClassificationAdapter(str(model_path), device="cpu")
        result = adapter.single_image_classification(self.image)

        self.assertEqual(set(result.keys()), {"prediction", "confidence", "class_id"})
        self.assertEqual(result["prediction"], "badger")
        self.assertGreaterEqual(result["confidence"], 0.0)

    def test_loads_checkpoint_dict_and_metadata(self):
        class_names = ["badger", "otter", "other"]
        model = build_classifier_model("mobilenet_v3_small", num_classes=len(class_names))
        checkpoint = {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "input_size": 224,
            "arch": "mobilenet_v3_small",
            "normalization": DEFAULT_NORMALIZATION,
            "metrics": {"val": {"macro_f1": 0.85}},
        }

        checkpoint_path = self.root / "checkpoint_model.pth"
        torch.save(checkpoint, checkpoint_path)

        adapter = WildlifeClassificationAdapter(str(checkpoint_path), device="cpu")
        self.assertEqual(adapter.CLASS_NAMES[0], "badger")
        self.assertEqual(adapter.CLASS_NAMES[2], "other")
        self.assertEqual(adapter.arch, "mobilenet_v3_small")
        self.assertIn("val", adapter.metrics)

        result = adapter.single_image_classification(self.image)
        self.assertEqual(set(result.keys()), {"prediction", "confidence", "class_id"})


if __name__ == "__main__":
    unittest.main()
