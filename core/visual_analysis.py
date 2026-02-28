from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    cv2 = None


@dataclass(frozen=True)
class VisualStats:
    changed_pct: float
    mean_delta: float
    max_delta: int
    hotspot_score: float
    threshold: int


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def compute_delta_map(original: Image.Image, modified: Image.Image, threshold: int = 0) -> np.ndarray:
    orig = _rgb_array(original).astype(np.int16)
    mod = _rgb_array(modified).astype(np.int16)
    delta = np.abs(mod - orig).max(axis=2).astype(np.uint8)
    thr = max(0, min(int(threshold), 255))
    if thr > 0:
        delta = np.where(delta >= thr, delta, 0).astype(np.uint8)
    return delta


def compute_visual_stats(delta_map: np.ndarray, threshold: int = 0) -> VisualStats:
    if delta_map.size == 0:
        return VisualStats(0.0, 0.0, 0, 0.0, int(threshold))
    changed = delta_map > 0
    changed_pct = float(np.mean(changed) * 100.0)
    mean_delta = float(np.mean(delta_map))
    max_delta = int(np.max(delta_map))
    hotspot_score = float(np.percentile(delta_map.astype(np.float32), 95)) if changed.any() else 0.0
    return VisualStats(
        changed_pct=changed_pct,
        mean_delta=mean_delta,
        max_delta=max_delta,
        hotspot_score=hotspot_score,
        threshold=int(threshold),
    )


def compute_heatmap(delta_map: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    if delta_map.ndim != 2:
        raise ValueError("delta_map должен быть двумерной картой")
    normalized = _normalize_uint8(delta_map)
    if cv2 is not None:
        cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET) if colormap == "turbo" else cv2.COLORMAP_JET
        bgr = cv2.applyColorMap(normalized, cmap)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.uint8)
    return _fallback_heatmap(normalized)


def compute_hotspot_grid(delta_map: np.ndarray, rows: int = 12, cols: int = 12) -> np.ndarray:
    if delta_map.ndim != 2:
        raise ValueError("delta_map должен быть двумерной картой")
    h, w = delta_map.shape
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    grid = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        y0 = int(round(r * h / rows))
        y1 = int(round((r + 1) * h / rows))
        for c in range(cols):
            x0 = int(round(c * w / cols))
            x1 = int(round((c + 1) * w / cols))
            tile = delta_map[y0:y1, x0:x1]
            grid[r, c] = float(tile.mean()) if tile.size else 0.0
    max_val = float(grid.max()) if grid.size else 0.0
    if max_val > 0:
        grid /= max_val
    return grid


def probe_pixel(original: Image.Image, modified: Image.Image, x: int, y: int) -> Dict[str, Any]:
    orig = _rgb_array(original)
    mod = _rgb_array(modified)
    h, w = orig.shape[:2]
    if w == 0 or h == 0:
        raise ValueError("Пустое изображение")
    px = max(0, min(int(x), w - 1))
    py = max(0, min(int(y), h - 1))
    before = tuple(int(v) for v in orig[py, px])
    after = tuple(int(v) for v in mod[py, px])
    delta = tuple(abs(a - b) for a, b in zip(before, after))
    max_delta = max(delta)
    intensity = int(round((max_delta / 255.0) * 100.0))
    channels = []
    for name, b, a in zip(("R", "G", "B"), before, after):
        channels.append(
            {
                "name": name,
                "before": int(b),
                "after": int(a),
                "delta": int(a - b),
                "before_bits": f"{b:08b}",
                "after_bits": f"{a:08b}",
                "before_lsb": int(b & 1),
                "after_lsb": int(a & 1),
            }
        )
    return {
        "x": px,
        "y": py,
        "before": before,
        "after": after,
        "delta": delta,
        "changed": bool(any(v != 0 for v in delta)),
        "intensity": intensity,
        "channels": channels,
    }


def build_analysis_preview(
    original: Image.Image,
    modified: Image.Image,
    mode: str,
    threshold: int = 0,
    split_ratio: float = 0.5,
    amplify: int = 20,
) -> Tuple[Image.Image, np.ndarray, VisualStats, np.ndarray]:
    orig = original.convert("RGB")
    mod = modified.convert("RGB")
    delta_map = compute_delta_map(orig, mod, threshold=threshold)
    stats = compute_visual_stats(delta_map, threshold=threshold)
    hotspot = compute_hotspot_grid(delta_map)

    if mode == "heatmap":
        preview = Image.fromarray(compute_heatmap(delta_map), mode="RGB")
    elif mode == "amplify20":
        preview = _build_amplified_delta(delta_map, amplify=max(1, int(amplify)))
    elif mode == "blend":
        ratio = max(0.0, min(1.0, float(split_ratio)))
        preview = Image.blend(orig, mod, ratio)
    else:
        preview = _build_split_preview(orig, mod, split_ratio)
    return preview, delta_map, stats, hotspot


def _build_split_preview(original: Image.Image, modified: Image.Image, split_ratio: float) -> Image.Image:
    ratio = max(0.0, min(1.0, float(split_ratio)))
    split_x = int(original.width * ratio)
    out = modified.copy()
    if split_x > 0:
        out.paste(original.crop((0, 0, split_x, original.height)), (0, 0))
    if 0 < split_x < out.width:
        line_w = max(2, out.width // 420)
        arr = np.array(out, dtype=np.uint8)
        x0 = max(1, min(out.width - 2, split_x))
        arr[:, max(0, x0 - 1 - line_w // 2):min(out.width, x0 + 1 + line_w // 2), :] = np.array([8, 22, 35], dtype=np.uint8)
        arr[:, max(0, x0 - line_w // 2):min(out.width, x0 + line_w // 2 + 1), :] = np.array([255, 209, 102], dtype=np.uint8)
        out = Image.fromarray(arr, mode="RGB")
    return out


def _build_amplified_delta(delta_map: np.ndarray, amplify: int = 20) -> Image.Image:
    amplified = np.clip(delta_map.astype(np.float32) * float(amplify), 0.0, 255.0)
    max_amp = float(amplified.max()) if amplified.size else 0.0
    scaled = (amplified / max_amp * 255.0) if max_amp > 0 else amplified
    mag = scaled.astype(np.float32) / 255.0
    r = np.clip((mag ** 0.7) * 255.0, 0.0, 255.0).astype(np.uint8)
    g = np.clip((mag ** 1.0) * 210.0 + 6.0, 0.0, 255.0).astype(np.uint8)
    b = np.clip((mag ** 1.6) * 70.0 + 26.0, 0.0, 255.0).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _normalize_uint8(delta_map: np.ndarray) -> np.ndarray:
    arr = delta_map.astype(np.float32)
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val <= 0:
        return np.zeros_like(delta_map, dtype=np.uint8)
    out = np.clip(arr / max_val * 255.0, 0.0, 255.0)
    return out.astype(np.uint8)


def _fallback_heatmap(normalized: np.ndarray) -> np.ndarray:
    norm = normalized.astype(np.float32) / 255.0
    r = np.clip((norm - 0.28) * 2.3, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(norm - 0.55) * 2.0, 0.0, 1.0)
    b = np.clip(1.0 - norm * 1.15, 0.12, 1.0)
    rgb = np.stack([r, g, b], axis=2)
    return (rgb * 255.0).astype(np.uint8)
