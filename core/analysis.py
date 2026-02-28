from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image, ImageFilter
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from .stego import decode_text_from_image, encode_text_into_image, max_message_bytes
from .visual_analysis import compute_delta_map, compute_heatmap


def compute_change_heatmap(original: Image.Image, modified: Image.Image) -> Image.Image:
    delta_map = compute_delta_map(original, modified, threshold=0)
    return Image.fromarray(compute_heatmap(delta_map), mode="RGB")


def run_attack_suite(
    encoded_image: Image.Image,
    expected_text: str,
    password: str,
    bits_per_channel: int,
    method: str,
) -> List[Dict[str, Any]]:
    suite = [
        ("baseline", "Без атаки", lambda img: img.copy()),
        ("jpeg_q35", "JPEG q=35 (с потерями)", lambda img: _jpeg_roundtrip(img, quality=35)),
        ("resize_70", "Изменение размера 70% + восстановление", lambda img: _resize_restore(img, scale=0.7)),
        ("noise_12", "Шум 12%", lambda img: _add_noise(img, amount=0.12, amplitude=24, seed=123)),
        ("blur_1", "Гауссово размытие r=1", lambda img: img.filter(ImageFilter.GaussianBlur(radius=1.0))),
    ]

    results: List[Dict[str, Any]] = []
    for attack_id, attack_name, transform in suite:
        transformed = transform(encoded_image.convert("RGB"))
        ok = False
        error = None
        extracted = ""
        try:
            extracted = decode_text_from_image(transformed, password, bits_per_channel, method)
            ok = extracted == expected_text
        except Exception as exc:
            error = str(exc)
        results.append(
            {
                "id": attack_id,
                "name": attack_name,
                "success": bool(ok),
                "error": error,
                "preview_text": _safe_preview_text(extracted),
            }
        )
    return results


def run_mode_benchmark(
    original_image: Image.Image,
    message_text: str,
    password: str,
    bits_options: Sequence[int] = (1, 2, 3),
    methods: Sequence[str] = ("sequential", "interleaved"),
) -> List[Dict[str, Any]]:
    image = original_image.convert("RGB")
    payload_bytes = len(message_text.encode("utf-8"))
    results: List[Dict[str, Any]] = []
    for method in methods:
        for bits in bits_options:
            capacity = max_message_bytes(image.size, int(bits))
            if payload_bytes > capacity:
                results.append(
                    {
                        "method": method,
                        "bits": int(bits),
                        "fit": False,
                        "decode_ok": False,
                        "psnr_db": None,
                        "mse": None,
                        "ssim": None,
                        "capacity": int(capacity),
                        "usage_ratio": None,
                        "error": "сообщение не помещается в контейнер",
                    }
                )
                continue
            try:
                encoded = encode_text_into_image(image, message_text, password, int(bits), method)
                decoded = decode_text_from_image(encoded, password, int(bits), method)
                arr_o = np.array(image)
                arr_e = np.array(encoded)
                results.append(
                    {
                        "method": method,
                        "bits": int(bits),
                        "fit": True,
                        "decode_ok": decoded == message_text,
                        "psnr_db": float(psnr(arr_o, arr_e, data_range=255)),
                        "mse": float(mse(arr_o, arr_e)),
                        "ssim": float(ssim(arr_o, arr_e, multichannel=True, data_range=255, channel_axis=-1)),
                        "capacity": int(capacity),
                        "usage_ratio": float(payload_bytes) / float(capacity) if capacity else None,
                        "error": None,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "method": method,
                        "bits": int(bits),
                        "fit": True,
                        "decode_ok": False,
                        "psnr_db": None,
                        "mse": None,
                        "ssim": None,
                        "capacity": int(capacity),
                        "usage_ratio": float(payload_bytes) / float(capacity) if capacity else None,
                        "error": str(exc),
                    }
                )
    return results


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=int(quality), subsampling=2)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _resize_restore(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size
    nw = max(8, int(w * scale))
    nh = max(8, int(h * scale))
    tmp = image.resize((nw, nh), Image.Resampling.BICUBIC)
    return tmp.resize((w, h), Image.Resampling.BICUBIC)


def _add_noise(image: Image.Image, amount: float, amplitude: int, seed: int) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.int16)
    rng = np.random.default_rng(seed)
    noise = rng.integers(-amplitude, amplitude + 1, size=arr.shape, dtype=np.int16)
    mask = rng.random((arr.shape[0], arr.shape[1], 1)) < float(amount)
    out = np.where(mask, arr + noise, arr)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _safe_preview_text(text: str, limit: int = 90) -> str:
    if not text:
        return ""
    cleaned = "".join(ch for ch in text if ch.isprintable() and ch not in "\r\n\t")
    cleaned = cleaned.replace("�", "")
    if not cleaned:
        return "нечитаемый текст"
    letters = sum(ch.isalnum() for ch in cleaned)
    ratio = letters / max(1, len(cleaned))
    if ratio < 0.35:
        return "нечитаемый текст"
    return cleaned[:limit]
