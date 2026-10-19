import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import robustness_check as rc  # noqa: E402


class HashTests(unittest.TestCase):
    def test_sha256_matches_hashlib(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.pt"
            path.write_bytes(b"abc" * 1000)
            self.assertEqual(rc.sha256(path), hashlib.sha256(b"abc" * 1000).hexdigest())


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.image = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)

    def test_every_condition_keeps_shape_and_type(self):
        rng = np.random.default_rng(0)
        for name, make in rc.CONDITIONS.items():
            out = make(self.image, rng)
            self.assertEqual(out.shape, self.image.shape, name)
            self.assertEqual(out.dtype, np.uint8, name)

    def test_corruptions_change_the_image_in_the_expected_direction(self):
        rng = np.random.default_rng(0)
        c = rc.CONDITIONS
        self.assertTrue(np.array_equal(c["clean"](self.image, rng), self.image))
        self.assertLess(c["darker_0.6"](self.image, rng).mean(), self.image.mean())
        self.assertGreater(c["brighter_1.4"](self.image, rng).mean(), self.image.mean())
        self.assertLess(c["blur_sigma_3"](self.image, rng).std(), self.image.std())

    def test_dataset_keeps_labels_and_never_touches_the_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src"
            (source / "val" / "images").mkdir(parents=True)
            (source / "val" / "labels").mkdir(parents=True)
            (source / "test" / "images").mkdir(parents=True)
            cv2.imwrite(str(source / "val" / "images" / "a.jpg"), self.image)
            (source / "val" / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
            before = (source / "val" / "images" / "a.jpg").read_bytes()
            config = rc.perturbed_dataset(source, Path(tmp) / "out", "blur_sigma_1.5")
            self.assertTrue(config.is_file())
            out = Path(tmp) / "out" / "val"
            self.assertTrue((out / "images" / "a.png").is_file())
            self.assertEqual((out / "labels" / "a.txt").read_text(), "0 0.5 0.5 0.1 0.1\n")
            self.assertEqual((source / "val" / "images" / "a.jpg").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
