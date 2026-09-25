import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from speciesnet_adapter import (
    DEFAULT_SPECIESNET_MODEL,
    SpeciesNetClassificationAdapter,
    format_speciesnet_label,
    resolve_speciesnet_model_name,
)


class FakeSpeciesNetClassifier:
    labels = {
        0: "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank",
        1: "uuid;mammalia;carnivora;mustelidae;meles;meles;Eurasian badger",
    }

    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device

    def preprocess(self, img, bboxes=None):
        if bboxes is not None:
            raise AssertionError("Crop adapter should not pass bboxes")
        return img

    def predict(self, filepath, img):
        return {
            "filepath": filepath,
            "classifications": {
                "classes": [self.labels[1], self.labels[0]],
                "scores": [0.91, 0.02],
            },
        }


class TestSpeciesNetAdapter(unittest.TestCase):
    def test_formats_taxonomy_label(self):
        self.assertEqual(
            format_speciesnet_label("uuid;mammalia;carnivora;mustelidae;lutra;lutra;Eurasian otter"),
            "Eurasian otter",
        )
        self.assertEqual(format_speciesnet_label("uuid;;;;;;blank"), "blank")
        self.assertEqual(format_speciesnet_label("animal"), "animal")

    def test_single_image_classification(self):
        adapter = SpeciesNetClassificationAdapter(
            model_name="local-speciesnet",
            device="cpu",
            classifier_cls=FakeSpeciesNetClassifier,
        )
        result = adapter.single_image_classification(Image.new("RGB", (64, 64)))

        self.assertEqual(result["prediction"], "Eurasian badger")
        self.assertAlmostEqual(result["confidence"], 0.91)
        self.assertEqual(result["class_id"], 1)
        self.assertIn("raw_prediction", result)

    def test_default_model_prefers_cache(self):
        cached_path = Path("cached/speciesnet")
        with mock.patch("speciesnet_adapter.find_cached_speciesnet_model", return_value=cached_path):
            self.assertEqual(resolve_speciesnet_model_name(None), str(cached_path))
            self.assertEqual(resolve_speciesnet_model_name(DEFAULT_SPECIESNET_MODEL), str(cached_path))

    def test_explicit_non_default_model_is_preserved(self):
        with mock.patch("speciesnet_adapter.find_cached_speciesnet_model") as find_cached:
            self.assertEqual(resolve_speciesnet_model_name("C:/models/speciesnet"), "C:/models/speciesnet")
            find_cached.assert_not_called()


if __name__ == "__main__":
    unittest.main()
