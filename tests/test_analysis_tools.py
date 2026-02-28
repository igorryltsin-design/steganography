import unittest

import numpy as np
from PIL import Image

from core.analysis import compute_change_heatmap, run_attack_suite, run_mode_benchmark
from core.stego import encode_text_into_image


class AnalysisToolsTests(unittest.TestCase):
    def setUp(self):
        h, w = 96, 96
        yy, xx = np.mgrid[0:h, 0:w]
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[..., 0] = (xx * 3 + yy) % 256
        arr[..., 1] = (yy * 5 + 20) % 256
        arr[..., 2] = ((xx + yy) * 2) % 256
        self.image = Image.fromarray(arr, mode="RGB")
        self.message = "analysis-tools-demo"
        self.password = "pwd"
        self.bits = 2
        self.method = "sequential"
        self.encoded = encode_text_into_image(
            self.image,
            self.message,
            self.password,
            self.bits,
            self.method,
        )

    def test_heatmap_returns_rgb_image_with_same_size(self):
        heat = compute_change_heatmap(self.image, self.encoded)
        self.assertEqual(heat.mode, "RGB")
        self.assertEqual(heat.size, self.image.size)

    def test_attack_suite_returns_rows(self):
        rows = run_attack_suite(
            self.encoded,
            self.message,
            self.password,
            self.bits,
            self.method,
        )
        self.assertGreaterEqual(len(rows), 5)
        names = {row["id"] for row in rows}
        self.assertIn("baseline", names)

    def test_mode_benchmark_returns_all_combinations(self):
        rows = run_mode_benchmark(
            self.image,
            self.message,
            self.password,
            bits_options=(1, 2),
            methods=("sequential", "interleaved"),
        )
        self.assertEqual(len(rows), 4)
        ok_count = sum(1 for row in rows if row["fit"])
        self.assertGreater(ok_count, 0)


if __name__ == "__main__":
    unittest.main()

