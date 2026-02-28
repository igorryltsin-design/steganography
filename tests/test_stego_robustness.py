import io
import unittest

import numpy as np
from PIL import Image

from core.stego import decode_text_from_image, encode_text_into_image


class StegoRobustnessTests(unittest.TestCase):
    def setUp(self):
        h, w = 128, 128
        yy, xx = np.mgrid[0:h, 0:w]
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[..., 0] = (xx * 2 + yy) % 256
        arr[..., 1] = (yy * 3 + 17) % 256
        arr[..., 2] = ((xx + yy) * 5) % 256
        self.source = Image.fromarray(arr, mode="RGB")
        self.message = "robustness-check-message"
        self.password = "pass-42"
        self.bits = 2
        self.method = "interleaved"
        self.encoded = encode_text_into_image(
            self.source,
            self.message,
            self.password,
            self.bits,
            self.method,
        )

    def assert_decode_corrupted(self, transformed: Image.Image):
        try:
            decoded = decode_text_from_image(transformed, self.password, self.bits, self.method)
        except Exception:
            return
        self.assertNotEqual(decoded, self.message)

    def test_png_roundtrip_preserves_message(self):
        buf = io.BytesIO()
        self.encoded.save(buf, format="PNG")
        buf.seek(0)
        restored = Image.open(buf).convert("RGB")
        decoded = decode_text_from_image(restored, self.password, self.bits, self.method)
        self.assertEqual(decoded, self.message)

    def test_jpeg_recompression_corrupts_message(self):
        buf = io.BytesIO()
        self.encoded.save(buf, format="JPEG", quality=35, subsampling=2)
        buf.seek(0)
        restored = Image.open(buf).convert("RGB")
        self.assert_decode_corrupted(restored)

    def test_resize_corrupts_message(self):
        resized = self.encoded.resize((96, 96), Image.Resampling.BICUBIC)
        restored = resized.resize(self.encoded.size, Image.Resampling.BICUBIC)
        self.assert_decode_corrupted(restored)

    def test_additive_noise_corrupts_message(self):
        rng = np.random.default_rng(123)
        arr = np.array(self.encoded, dtype=np.int16)
        noise = rng.integers(-30, 31, size=arr.shape, dtype=np.int16)
        mask = rng.random((arr.shape[0], arr.shape[1], 1)) < 0.12
        noisy = np.where(mask, arr + noise, arr)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        restored = Image.fromarray(noisy, mode="RGB")
        self.assert_decode_corrupted(restored)


if __name__ == "__main__":
    unittest.main()

