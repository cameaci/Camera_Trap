import unittest
from io import BytesIO

from PIL import Image

from fetch_uk_open_data import (
    ObservationRecord,
    PhotoRecord,
    choose_target_split,
    interleave_observations,
    normalize_photo_url,
    parse_int_csv,
    validate_image_content,
)


def _make_photo(observation_id: int, photo_id: int) -> PhotoRecord:
    return PhotoRecord(
        class_name="other",
        species="species",
        observation_id=observation_id,
        photo_id=photo_id,
        license_code="cc-by",
        url="https://example.org/photo.jpg",
    )


class TestFetchUkOpenData(unittest.TestCase):
    def test_parse_int_csv(self):
        self.assertEqual(parse_int_csv("600,150,150", 3, "--targets"), [600, 150, 150])
        with self.assertRaises(ValueError):
            parse_int_csv("600,150", 3, "--targets")

    def test_choose_target_split(self):
        counts = {"train": 100, "val": 10, "test": 10}
        targets = {"train": 600, "val": 150, "test": 150}
        self.assertEqual(choose_target_split(counts, targets), "val")

        full = {"train": 600, "val": 150, "test": 150}
        self.assertIsNone(choose_target_split(full, targets))

    def test_interleave_observations(self):
        records = [
            ObservationRecord("other", "fox", 1, [_make_photo(1, 11)]),
            ObservationRecord("other", "fox", 2, [_make_photo(2, 22)]),
            ObservationRecord("other", "deer", 3, [_make_photo(3, 33)]),
            ObservationRecord("other", "deer", 4, [_make_photo(4, 44)]),
        ]
        interleaved = interleave_observations(records, rng=__import__("random").Random(42))
        self.assertEqual(len(interleaved), 4)
        species_sequence = [item.species for item in interleaved]
        self.assertGreaterEqual(species_sequence.count("fox"), 2)
        self.assertGreaterEqual(species_sequence.count("deer"), 2)

    def test_normalize_photo_url(self):
        url = "https://static.inaturalist.org/photos/123/square.jpg"
        self.assertEqual(
            normalize_photo_url(url, preferred_size="large"),
            "https://static.inaturalist.org/photos/123/large.jpg",
        )
        s3_url = "https://inaturalist-open-data.s3.amazonaws.com/photos/123/square.jpg"
        self.assertEqual(
            normalize_photo_url(s3_url, preferred_size="large"),
            "https://inaturalist-open-data.s3.amazonaws.com/photos/123/large.jpg",
        )

    def test_validate_image_content(self):
        img = Image.new("RGB", (300, 240), color=(1, 2, 3))
        bio = BytesIO()
        img.save(bio, format="JPEG")
        ok, ext, _ = validate_image_content(bio.getvalue(), min_short_edge=224)
        self.assertTrue(ok)
        self.assertEqual(ext, ".jpg")

        small = Image.new("RGB", (100, 100), color=(1, 2, 3))
        bio2 = BytesIO()
        small.save(bio2, format="JPEG")
        ok2, _, _ = validate_image_content(bio2.getvalue(), min_short_edge=224)
        self.assertFalse(ok2)


if __name__ == "__main__":
    unittest.main()
