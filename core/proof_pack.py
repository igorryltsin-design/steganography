from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
import json
import zipfile

from PIL import Image

from .analysis import compute_change_heatmap
from .report import render_report_text


FIXED_ZIP_DATETIME = (2020, 1, 1, 0, 0, 0)


def export_proof_pack(
    zip_path: str | Path,
    report: Dict[str, Any],
    source_image: Image.Image,
    encoded_image: Image.Image,
    attacks: Sequence[Mapping[str, Any]] | None = None,
    extra_png_artifacts: Mapping[str, bytes] | None = None,
) -> str:
    """Создает детерминированный ZIP-архив с ключевыми артефактами."""
    zip_path = str(zip_path)

    artifacts = report.setdefault("artifacts", {})
    artifacts["proof_pack_path"] = zip_path
    visual_artifacts = report.setdefault("visual_artifacts", {})
    visual_artifacts.update(
        {
            "before": "proof_pack/before.png",
            "after": "proof_pack/after.png",
            "heatmap": "proof_pack/heatmap.png",
            "attacks_csv": "proof_pack/attacks.csv",
        }
    )
    if extra_png_artifacts:
        for arc_name in sorted(extra_png_artifacts):
            key = Path(arc_name).stem
            visual_artifacts[key] = arc_name

    heatmap = compute_change_heatmap(source_image, encoded_image)
    report_json = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    report_txt = render_report_text(report).encode("utf-8")

    before_png = _pil_to_png_bytes(source_image)
    after_png = _pil_to_png_bytes(encoded_image)
    heatmap_png = _pil_to_png_bytes(heatmap)
    attacks_csv = _attacks_to_csv_bytes(attacks or [])

    files = [
        ("proof_pack/report.json", report_json),
        ("proof_pack/report.txt", report_txt),
        ("proof_pack/before.png", before_png),
        ("proof_pack/after.png", after_png),
        ("proof_pack/heatmap.png", heatmap_png),
        ("proof_pack/attacks.csv", attacks_csv),
    ]
    if extra_png_artifacts:
        for arc_name, payload in sorted(extra_png_artifacts.items()):
            files.append((arc_name, payload))

    with zipfile.ZipFile(zip_path, mode="w") as zf:
        for arc_name, payload in files:
            _writestr_deterministic(zf, arc_name, payload)
    return zip_path


def _writestr_deterministic(zf: zipfile.ZipFile, arc_name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(filename=arc_name, date_time=FIXED_ZIP_DATETIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    zf.writestr(info, payload)


def _pil_to_png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _attacks_to_csv_bytes(attacks: Sequence[Mapping[str, Any]]) -> bytes:
    # csv ожидает строковый буфер; собираем через list и кодируем UTF-8.
    lines: list[str] = []
    lines.append("attack_id,attack_name,success,error,preview_text")
    for row in attacks:
        attack_id = _csv_escape(str(row.get("id", "")))
        name = _csv_escape(str(row.get("name", "")))
        success = "1" if row.get("success") else "0"
        error = _csv_escape(str(row.get("error") or ""))
        preview = _csv_escape(str(row.get("preview_text") or ""))
        lines.append(f"{attack_id},{name},{success},{error},{preview}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _csv_escape(value: str) -> str:
    if any(ch in value for ch in [",", '"', "\n", "\r"]):
        return '"' + value.replace('"', '""') + '"'
    return value
