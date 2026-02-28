from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class PixelInspectorWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setMinimumHeight(320)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title_row = QHBoxLayout()
        self.title = QLabel("Пиксельный инспектор")
        self.title.setObjectName("SectionTitle")
        title_row.addWidget(self.title)
        title_row.addStretch(1)
        title_row.addWidget(QLabel("Лупа"))
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["12x", "24x"])
        title_row.addWidget(self.zoom_combo)
        root.addLayout(title_row)

        self.hint = QLabel("Кликните по изображению или включите слежение курсора.")
        self.hint.setObjectName("Hint")
        self.hint.setWordWrap(True)
        root.addWidget(self.hint)

        top = QHBoxLayout()
        top.setSpacing(12)
        root.addLayout(top)

        self.magnifier = QLabel()
        self.magnifier.setFixedSize(146, 146)
        self.magnifier.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.magnifier.setStyleSheet("border: 1px solid rgba(126,173,214,0.55); border-radius: 8px;")
        top.addWidget(self.magnifier)

        form_box = QWidget()
        form = QFormLayout(form_box)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        self.coord_label = QLabel("—")
        self.intensity_label = QLabel("—")
        self.before_label = QLabel("—")
        self.after_label = QLabel("—")
        self.delta_label = QLabel("—")
        form.addRow("Координаты:", self.coord_label)
        form.addRow("Сила изменения:", self.intensity_label)
        form.addRow("До:", self.before_label)
        form.addRow("После:", self.after_label)
        form.addRow("Дельта:", self.delta_label)
        top.addWidget(form_box, 1)

        self.channel_labels: dict[str, QLabel] = {}
        self.lsb_labels: dict[str, QLabel] = {}
        for channel in ("R", "G", "B"):
            block = QFrame()
            block.setObjectName("CardSoft")
            block_layout = QVBoxLayout(block)
            block_layout.setContentsMargins(10, 8, 10, 8)
            block_layout.setSpacing(4)
            head = QLabel(f"Канал {channel}")
            head.setStyleSheet("font-weight: 700;")
            block_layout.addWidget(head)
            bits = QLabel("—")
            bits.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            bits.setWordWrap(True)
            bits.setFont(QFont("Menlo", 11))
            block_layout.addWidget(bits)
            lsb = QLabel("LSB: —")
            lsb.setObjectName("Hint")
            lsb.setWordWrap(True)
            block_layout.addWidget(lsb)
            root.addWidget(block)
            self.channel_labels[channel] = bits
            self.lsb_labels[channel] = lsb

        self.clear()

    def clear(self) -> None:
        self.coord_label.setText("—")
        self.intensity_label.setText("—")
        self.before_label.setText("—")
        self.after_label.setText("—")
        self.delta_label.setText("—")
        self.magnifier.setPixmap(QPixmap())
        self.magnifier.setText("Нет точки")
        for channel in ("R", "G", "B"):
            self.channel_labels[channel].setText("—")
            self.lsb_labels[channel].setText("LSB: —")

    def set_probe_data(
        self,
        probe: Mapping[str, Any],
        original: Image.Image | None,
        modified: Image.Image | None,
    ) -> None:
        self.coord_label.setText(f"x={probe['x']}, y={probe['y']}")
        self.intensity_label.setText(f"{probe['intensity']}/100")
        self.before_label.setText(str(tuple(probe["before"])))
        self.after_label.setText(str(tuple(probe["after"])))
        self.delta_label.setText(str(tuple(probe["delta"])))
        for channel in probe["channels"]:
            name = str(channel["name"])
            self.channel_labels[name].setText(
                f"{channel['before_bits']} -> {channel['after_bits']}\n"
                f"{channel['before']} -> {channel['after']} (Δ {channel['delta']:+d})"
            )
            self.lsb_labels[name].setText(f"LSB: {channel['before_lsb']} -> {channel['after_lsb']}")
        if original is not None and modified is not None:
            self.magnifier.setPixmap(self._build_magnifier(original, modified, int(probe["x"]), int(probe["y"])))

    def _build_magnifier(self, original: Image.Image, modified: Image.Image, x: int, y: int) -> QPixmap:
        zoom = 12 if self.zoom_combo.currentText() == "12x" else 24
        crop_size = 11
        orig = np.array(original.convert("RGB"), dtype=np.uint8)
        mod = np.array(modified.convert("RGB"), dtype=np.uint8)
        h, w = orig.shape[:2]
        x0 = max(0, x - crop_size // 2)
        x1 = min(w, x0 + crop_size)
        y0 = max(0, y - crop_size // 2)
        y1 = min(h, y0 + crop_size)
        x0 = max(0, x1 - crop_size)
        y0 = max(0, y1 - crop_size)

        orig_crop = orig[y0:y1, x0:x1]
        mod_crop = mod[y0:y1, x0:x1]
        diff = np.abs(mod_crop.astype(np.int16) - orig_crop.astype(np.int16)).max(axis=2).astype(np.uint8)

        left = Image.fromarray(orig_crop, mode="RGB").resize((orig_crop.shape[1] * zoom, orig_crop.shape[0] * zoom), Image.Resampling.NEAREST)
        right = Image.fromarray(mod_crop, mode="RGB").resize((mod_crop.shape[1] * zoom, mod_crop.shape[0] * zoom), Image.Resampling.NEAREST)

        canvas_w = left.width + right.width + 18
        canvas_h = max(left.height, right.height) + 28
        pix = QPixmap(canvas_w, canvas_h)
        pix.fill(QColor("#0b1f33"))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.drawPixmap(0, 18, _pil_to_pixmap(left))
        painter.drawPixmap(left.width + 18, 18, _pil_to_pixmap(right))

        center_x = (x - x0) * zoom + zoom // 2
        center_y = (y - y0) * zoom + zoom // 2 + 18
        painter.setPen(QColor("#ffd166"))
        painter.drawRect(center_x - zoom // 2, center_y - zoom // 2, zoom, zoom)
        painter.drawRect(left.width + 18 + center_x - zoom // 2, center_y - zoom // 2, zoom, zoom)

        painter.setPen(QColor("#dbefff"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(0, 12, "До")
        painter.drawText(left.width + 18, 12, "После")

        if diff.size:
            painter.setPen(QColor("#45e1bc"))
            painter.drawText(0, canvas_h - 4, f"Локальный max Δ: {int(diff.max())}")
        painter.end()
        return pix


def _pil_to_pixmap(image: Image.Image) -> QPixmap:
    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())
