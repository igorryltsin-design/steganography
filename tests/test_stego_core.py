import unittest

from PIL import Image

from core.stego import (
    decode_text_from_image,
    encode_text_into_image,
    max_message_bytes,
)


class StegoCoreTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (128, 128), (93, 127, 149))

    def test_roundtrip_for_all_modes_and_bits(self):
        message = "Привет! Проверка round-trip."
        password = "demo-pass"
        for bits in (1, 2, 3):
            for method in ("sequential", "interleaved"):
                with self.subTest(bits=bits, method=method):
                    encoded = encode_text_into_image(self.image, message, password, bits, method)
                    decoded = decode_text_from_image(encoded, password, bits, method)
                    self.assertEqual(decoded, message)

    def test_capacity_check_raises(self):
        tiny = Image.new("RGB", (8, 8), (40, 80, 120))
        bits = 1
        method = "sequential"
        capacity = max_message_bytes(tiny.size, bits)
        too_large = "x" * (capacity + 1)
        with self.assertRaises(ValueError):
            encode_text_into_image(tiny, too_large, "", bits, method)

    def test_decode_from_clean_image_raises(self):
        with self.assertRaises(ValueError):
            decode_text_from_image(self.image, "", 1, "sequential")


if __name__ == "__main__":
    unittest.main()
