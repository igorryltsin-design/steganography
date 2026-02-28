from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
from PIL import Image


def chi_square_lsb_test(img: Image.Image | None) -> Dict[str, float | int | bool]:
    if img is None:
        return {"chi2": 0.0, "zero_bits": 0, "one_bits": 0, "suspicious": False}

    counts = {"0": 0, "1": 0}
    pixels = np.array(img.convert("RGB"), dtype=np.uint8)
    for channel in range(3):
        lsb = pixels[:, :, channel] & 1
        counts["0"] += int(np.sum(lsb == 0))
        counts["1"] += int(np.sum(lsb == 1))

    total = counts["0"] + counts["1"]
    expected = max(1.0, total / 2.0)
    chi2 = ((counts["0"] - expected) ** 2 / expected) + ((counts["1"] - expected) ** 2 / expected)

    return {
        "chi2": float(chi2),
        "zero_bits": int(counts["0"]),
        "one_bits": int(counts["1"]),
        "suspicious": bool(chi2 > 10.0),
    }


def evaluate_risk(usage_ratio: float, stats_orig: Mapping[str, float], stats_enc: Mapping[str, float]) -> Tuple[str, str]:
    usage_ratio = max(0.0, min(1.0, float(usage_ratio)))

    z0 = float(stats_orig.get("zero_bits", 0))
    o0 = float(stats_orig.get("one_bits", 0))
    z1 = float(stats_enc.get("zero_bits", 0))
    o1 = float(stats_enc.get("one_bits", 0))
    t0 = max(1.0, z0 + o0)
    t1 = max(1.0, z1 + o1)

    imbalance_orig = abs(z0 - o0) / t0
    imbalance_stego = abs(z1 - o1) / t1
    delta_imbalance = max(0.0, imbalance_stego - imbalance_orig)

    chi0 = float(stats_orig.get("chi2", 0.0))
    chi1 = float(stats_enc.get("chi2", 0.0))
    chi_shift = abs(chi1 - chi0) / max(abs(chi0), 1e-9)

    suspicious = bool(stats_enc.get("suspicious", False))
    if usage_ratio >= 0.40 and (delta_imbalance >= 0.02 or chi_shift >= 0.20):
        return (
            "HIGH",
            f"высокая загрузка контейнера ({usage_ratio * 100:.1f}%) и заметный сдвиг LSB-статистики",
        )
    if usage_ratio >= 0.18 or (usage_ratio >= 0.08 and (delta_imbalance >= 0.01 or chi_shift >= 0.08)) or (
        usage_ratio >= 0.12 and suspicious
    ):
        return (
            "MEDIUM",
            f"средняя загрузка ({usage_ratio * 100:.1f}%) или умеренное отклонение LSB-распределения",
        )
    return (
        "LOW",
        f"низкая загрузка ({usage_ratio * 100:.1f}%) и минимальные изменения LSB-статистики",
    )
