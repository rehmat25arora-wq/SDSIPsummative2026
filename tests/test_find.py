import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from imagehash import ImageHash
from FINd import FINDHasher
from FINd_optimised import FINDHasherOptimised

# Use a small fixed set of images for testing
TEST_IMAGES = [
    "meme_images/0000_12268686.jpg",
    "meme_images/0000_12270286.jpg",
    "meme_images/0000_12270966.jpg",
    "meme_images/0000_12271226.jpg",
    "meme_images/0000_12272214.jpg",
]

class TestFINDHasherOptimised(unittest.TestCase):

    def setUp(self):
        """Runs before every test -- create both hashers once."""
        self.original = FINDHasher()
        self.optimised = FINDHasherOptimised()

    def test_hash_correctness(self):
        """Optimised version must produce identical hashes to original."""
        for path in TEST_IMAGES:
            with self.subTest(image=path):
                h_orig = str(self.original.fromFile(path))
                h_opt  = str(self.optimised.fromFile(path))
                self.assertEqual(h_orig, h_opt,
                    f"Hash mismatch for {path}:\n"
                    f"  original : {h_orig}\n"
                    f"  optimised: {h_opt}")

    def test_hash_type(self):
        """fromFile() must return an ImageHash object."""
        h = self.optimised.fromFile(TEST_IMAGES[0])
        self.assertIsInstance(h, ImageHash)

    def test_self_distance(self):
        """Hamming distance between a hash and itself must be zero."""
        h = self.optimised.fromFile(TEST_IMAGES[0])
        self.assertEqual(h - h, 0)

    def test_hash_length(self):
        """FINd produces a 256-bit hash -- verify the length."""
        h = self.optimised.fromFile(TEST_IMAGES[0])
        self.assertEqual(h.hash.size, 256)
    def test_fromFiles_matches_fromFile(self):
        """Batch hashing should match individual hashing exactly."""
        batch = self.optimised.fromFiles(TEST_IMAGES, n_workers=2)

        for path in TEST_IMAGES:
            with self.subTest(image=path):
                single = str(self.optimised.fromFile(path))
                self.assertEqual(batch[path], single)

if __name__ == "__main__":
    unittest.main()
