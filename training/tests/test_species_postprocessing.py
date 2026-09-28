import unittest

from species_postprocessing import (
    format_species_detection_label,
    is_unknown_prediction,
    normalize_species_label,
)


class TestSpeciesPostprocessing(unittest.TestCase):
    def test_normalize_species_label(self):
        self.assertEqual(normalize_species_label(" Otter "), "otter")
        self.assertEqual(normalize_species_label("BADGER-SETT"), "badger sett")
        self.assertEqual(normalize_species_label(None), "")

    def test_other_is_unknown(self):
        self.assertTrue(is_unknown_prediction("other", confidence=0.99, conf_threshold=0.7))
        self.assertTrue(is_unknown_prediction("background", confidence=0.99, conf_threshold=0.7))

    def test_low_confidence_is_unknown(self):
        self.assertTrue(is_unknown_prediction("otter", confidence=0.4, conf_threshold=0.7))

    def test_format_species_detection_label(self):
        self.assertEqual(
            format_species_detection_label(
                {"prediction": "badger", "confidence": 0.92},
                conf_threshold=0.70,
            ),
            "badger 0.92",
        )
        self.assertEqual(
            format_species_detection_label(
                {"prediction": "other", "confidence": 0.99},
                conf_threshold=0.70,
            ),
            "Unknown",
        )
        self.assertEqual(
            format_species_detection_label(
                {"prediction": "otter", "confidence": 0.61},
                conf_threshold=0.70,
            ),
            "Unknown",
        )


if __name__ == "__main__":
    unittest.main()

