from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence
import uuid


APP_NAME = "Стего Студия"
APP_VERSION = "0.3.0"
REPORT_SCHEMA_NAME = "stegano_report"
REPORT_SCHEMA_VERSION = "1.1.0"


def build_stegano_report(
    *,
    source_image_path: Optional[str],
    image_size: Sequence[int],
    method: str,
    bits_per_channel: int,
    password_used: bool,
    message_chars: int,
    message_bytes_utf8: int,
    capacity_bytes: int,
    psnr_db: Optional[float],
    mse: Optional[float],
    ssim: Optional[float],
    metrics_error: Optional[str],
    chi_original: Mapping[str, Any],
    chi_stego: Mapping[str, Any],
    risk_level: Optional[str] = None,
    risk_reason: Optional[str] = None,
    demo_summary: Optional[Mapping[str, Any]] = None,
    robustness_score: Optional[float] = None,
    visual_artifacts: Optional[Mapping[str, Any]] = None,
    recommendation: Optional[str] = None,
) -> Dict[str, Any]:
    width = int(image_size[0])
    height = int(image_size[1])
    usage_ratio = 0.0
    if capacity_bytes > 0:
        usage_ratio = min(1.0, float(message_bytes_utf8) / float(capacity_bytes))

    return {
        "schema": {
            "name": REPORT_SCHEMA_NAME,
            "version": REPORT_SCHEMA_VERSION,
        },
        "meta": {
            "report_id": str(uuid.uuid4()),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
        },
        "input": {
            "source_image_path": source_image_path,
            "image": {
                "width": width,
                "height": height,
                "mode": "RGB",
            },
        },
        "embedding": {
            "method": method,
            "bits_per_channel": int(bits_per_channel),
            "password_used": bool(password_used),
            "message": {
                "chars": int(message_chars),
                "bytes_utf8": int(message_bytes_utf8),
            },
            "capacity": {
                "bytes": int(capacity_bytes),
                "kb": round(float(capacity_bytes) / 1024.0, 3),
                "usage_ratio": round(usage_ratio, 6),
            },
        },
        "quality_metrics": {
            "psnr_db": _round_or_none(psnr_db, 6),
            "mse": _round_or_none(mse, 8),
            "ssim": _round_or_none(ssim, 8),
            "error": metrics_error,
        },
        "steganalysis": {
            "chi_square": {
                "original": _normalize_chi_stats(chi_original),
                "stego": _normalize_chi_stats(chi_stego),
            }
        },
        "risk": {
            "level": risk_level,
            "reason": risk_reason,
        },
        "demo_summary": dict(demo_summary or {}),
        "robustness_score": _round_or_none(robustness_score, 4),
        "visual_artifacts": dict(visual_artifacts or {}),
        "recommendation": recommendation,
        "artifacts": {
            "encoded_image_path": None,
            "report_path": None,
            "report_format": None,
            "proof_pack_path": None,
        },
    }


def render_report_text(report: Mapping[str, Any]) -> str:
    schema = report.get("schema", {})
    meta = report.get("meta", {})
    input_block = report.get("input", {})
    image = input_block.get("image", {})
    embedding = report.get("embedding", {})
    message = embedding.get("message", {})
    capacity = embedding.get("capacity", {})
    quality = report.get("quality_metrics", {})
    chi = report.get("steganalysis", {}).get("chi_square", {})
    chi_orig = chi.get("original", {})
    chi_stego = chi.get("stego", {})
    risk = report.get("risk", {})
    artifacts = report.get("artifacts", {})
    demo_summary = report.get("demo_summary", {})
    visual_artifacts = report.get("visual_artifacts", {})

    method_raw = embedding.get("method", "-")
    method_name = _method_label(method_raw)
    password_used = "да" if embedding.get("password_used") else "нет"
    suspicious_orig = "да" if chi_orig.get("suspicious") else "нет"
    suspicious_stego = "да" if chi_stego.get("suspicious") else "нет"

    lines = [
        "Стего Студия — отчёт",
        f"Схема отчёта: {schema.get('name', '-')}/{schema.get('version', '-')}",
        f"ID отчёта: {meta.get('report_id', '-')}",
        f"Сформирован (UTC): {meta.get('generated_at_utc', '-')}",
        "",
        "Входные данные:",
        f"- Исходное изображение: {input_block.get('source_image_path', '-')}",
        f"- Размер: {image.get('width', '-')}x{image.get('height', '-')} ({image.get('mode', '-')})",
        "",
        "Параметры встраивания:",
        f"- Метод: {method_name}",
        f"- Бит на канал: {embedding.get('bits_per_channel', '-')}",
        f"- Пароль использован: {password_used}",
        f"- Длина сообщения (символы): {message.get('chars', '-')}",
        f"- Длина сообщения (UTF-8 байт): {message.get('bytes_utf8', '-')}",
        f"- Вместимость (байт): {capacity.get('bytes', '-')}",
        f"- Вместимость (КБ): {capacity.get('kb', '-')}",
        f"- Заполнение контейнера: {capacity.get('usage_ratio', '-')}",
        "",
        "Качество:",
        f"- PSNR (дБ): {quality.get('psnr_db', '-')}",
        f"- MSE: {quality.get('mse', '-')}",
        f"- SSIM: {quality.get('ssim', '-')}",
    ]
    if quality.get("error"):
        lines.append(f"- Ошибка расчёта метрик: {quality.get('error')}")

    lines.extend(
        [
            "",
            "Стегоанализ (хи-квадрат):",
            (
                "- Оригинал: χ²={chi2}, нулевых LSB={zero_bits}, единичных LSB={one_bits}, подозрительно={suspicious}".format(
                    chi2=chi_orig.get("chi2", "-"),
                    zero_bits=chi_orig.get("zero_bits", "-"),
                    one_bits=chi_orig.get("one_bits", "-"),
                    suspicious=suspicious_orig,
                )
            ),
            (
                "- Стего: χ²={chi2}, нулевых LSB={zero_bits}, единичных LSB={one_bits}, подозрительно={suspicious}".format(
                    chi2=chi_stego.get("chi2", "-"),
                    zero_bits=chi_stego.get("zero_bits", "-"),
                    one_bits=chi_stego.get("one_bits", "-"),
                    suspicious=suspicious_stego,
                )
            ),
            "",
            "Оценка риска:",
            f"- Уровень: {risk.get('level', '-')}",
            f"- Причина: {risk.get('reason', '-')}",
            "",
            "Дополнительные поля 1.1:",
            f"- Устойчивость (robustness_score): {report.get('robustness_score', '-')}",
            f"- Рекомендация: {report.get('recommendation', '-')}",
            f"- Demo summary: {demo_summary if demo_summary else '-'}",
            f"- Visual artifacts: {visual_artifacts if visual_artifacts else '-'}",
            f"- Скриншот hotspot: {visual_artifacts.get('hotspot', '-')}",
            f"- Скриншот инспектора: {visual_artifacts.get('inspector', '-')}",
            "",
            "Артефакты:",
            f"- Файл стего-изображения: {artifacts.get('encoded_image_path', '-')}",
            f"- Файл отчёта: {artifacts.get('report_path', '-')}",
            f"- Формат отчёта: {artifacts.get('report_format', '-')}",
            f"- Архив данных: {artifacts.get('proof_pack_path', '-')}",
            "",
        ]
    )
    return "\n".join(lines)


def render_presentation_summary(report: Mapping[str, Any]) -> str:
    risk = report.get("risk", {})
    quality = report.get("quality_metrics", {})
    robustness = report.get("robustness_score")
    recommendation = report.get("recommendation", "-")
    demo_summary = report.get("demo_summary", {})
    method = _method_label(report.get("embedding", {}).get("method", "-"))
    bits = report.get("embedding", {}).get("bits_per_channel", "-")

    return "\n".join(
        [
            "Краткая сводка",
            f"Режим: {method} / {bits} бит на канал",
            f"PSNR: {quality.get('psnr_db', '-')} дБ, SSIM: {quality.get('ssim', '-')}",
            f"Риск: {risk.get('level', '-')} ({risk.get('reason', '-')})",
            f"Устойчивость: {robustness if robustness is not None else '-'}",
            f"Рекомендация: {recommendation}",
            f"Hotspot: {report.get('visual_artifacts', {}).get('hotspot', '-')}",
            f"Инспектор: {report.get('visual_artifacts', {}).get('inspector', '-')}",
            f"Demo summary: {demo_summary if demo_summary else '-'}",
        ]
    )


def _normalize_chi_stats(stats: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "chi2": _round_or_none(stats.get("chi2"), 8),
        "zero_bits": int(stats.get("zero_bits", 0)),
        "one_bits": int(stats.get("one_bits", 0)),
        "suspicious": bool(stats.get("suspicious", False)),
    }


def _round_or_none(value: Any, digits: int) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _method_label(method_raw: Any) -> str:
    if method_raw == "sequential":
        return "Последовательный (R→G→B)"
    if method_raw == "interleaved":
        return "Чередование каналов"
    return str(method_raw)
