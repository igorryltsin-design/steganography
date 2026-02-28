import unittest

import numpy as np
from PIL import Image

from core.visual_analysis import (
    build_analysis_preview,
    compute_delta_map,
    compute_hotspot_grid,
    compute_visual_stats,
    probe_pixel,
)


class VisualAnalysisTests(unittest.TestCase):
    def setUp(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        arr[..., 0] = 100
        arr[..., 1] = 120
        arr[..., 2] = 140
        self.original = Image.fromarray(arr, mode="RGB")
        mod = arr.copy()
        mod[3, 4] = [101, 120, 141]
        mod[7, 2] = [100, 123, 140]
        self.modified = Image.fromarray(mod, mode="RGB")

    def test_probe_pixel_reports_channel_bits(self):
        info = probe_pixel(self.original, self.modified, 4, 3)
        self.assertEqual(info["x"], 4)
        self.assertEqual(info["y"], 3)
        self.assertTrue(info["changed"])
        self.assertEqual(info["before"], (100, 120, 140))
        self.assertEqual(info["after"], (101, 120, 141))
        self.assertEqual(info["channels"][0]["before_bits"], "01100100")
        self.assertEqual(info["channels"][0]["after_bits"], "01100101")

    def test_hotspot_grid_is_normalized(self):
        delta = compute_delta_map(self.original, self.modified, threshold=0)
        grid = compute_hotspot_grid(delta, rows=4, cols=5)
        self.assertEqual(grid.shape, (4, 5))
        self.assertGreaterEqual(float(grid.min()), 0.0)
        self.assertLessEqual(float(grid.max()), 1.0)

    def test_threshold_filters_small_changes(self):
        delta = compute_delta_map(self.original, self.modified, threshold=2)
        self.assertEqual(int(delta[3, 4]), 0)
        self.assertEqual(int(delta[7, 2]), 3)

    def test_build_preview_returns_stats(self):
        preview, delta, stats, hotspot = build_analysis_preview(
            self.original,
            self.modified,
            mode="heatmap",
            threshold=0,
        )
        self.assertEqual(preview.mode, "RGB")
        self.assertEqual(delta.shape, (10, 10))
        self.assertGreater(stats.changed_pct, 0.0)
        self.assertEqual(hotspot.shape, (12, 12))

    def test_compute_visual_stats_for_empty_map(self):
        stats = compute_visual_stats(np.zeros((4, 4), dtype=np.uint8), threshold=3)
        self.assertEqual(stats.changed_pct, 0.0)
        self.assertEqual(stats.max_delta, 0)
        self.assertEqual(stats.threshold, 3)


if __name__ == "__main__":
    unittest.main()
