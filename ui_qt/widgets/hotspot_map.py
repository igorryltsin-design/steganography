from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QWidget


class HotspotMapWidget(QFrame):
    tileActivated = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setMinimumHeight(170)
        self._grid = np.zeros((12, 12), dtype=np.float32)
        self._image_size = (1, 1)
        self._selected: Optional[tuple[int, int]] = None
        self.setToolTip("Миникарта интенсивности изменений. Клик по ячейке переносит фокус в область изображения.")

    def set_grid(self, grid: np.ndarray, image_size: tuple[int, int]) -> None:
        self._grid = np.array(grid, dtype=np.float32) if grid.size else np.zeros((12, 12), dtype=np.float32)
        self._image_size = image_size
        self.update()

    def clear(self) -> None:
        self._grid = np.zeros((12, 12), dtype=np.float32)
        self._selected = None
        self.update()

    def set_selected(self, row: int, col: int) -> None:
        self._selected = (int(row), int(col))
        self.update()

    def mousePressEvent(self, event):
        if self._grid.size == 0:
            return super().mousePressEvent(event)
        tile = self._tile_at(event.position().toPoint())
        if tile is not None:
            self._selected = tile
            self.update()
            self.tileActivated.emit(tile[0], tile[1])
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        rect = self.contentsRect().adjusted(10, 10, -10, -10)
        rows, cols = self._grid.shape
        tile_w = rect.width() / max(1, cols)
        tile_h = rect.height() / max(1, rows)
        painter.setPen(Qt.PenStyle.NoPen)

        for row in range(rows):
            for col in range(cols):
                value = float(self._grid[row, col])
                painter.setBrush(_hotspot_color(value))
                painter.drawRect(
                    int(rect.left() + col * tile_w),
                    int(rect.top() + row * tile_h),
                    int(tile_w + 1),
                    int(tile_h + 1),
                )

        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        for row in range(rows + 1):
            y = int(rect.top() + row * tile_h)
            painter.drawLine(rect.left(), y, rect.right(), y)
        for col in range(cols + 1):
            x = int(rect.left() + col * tile_w)
            painter.drawLine(x, rect.top(), x, rect.bottom())

        if self._selected is not None:
            row, col = self._selected
            painter.setPen(QPen(QColor("#ffd166"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(
                int(rect.left() + col * tile_w),
                int(rect.top() + row * tile_h),
                int(tile_w),
                int(tile_h),
            )

        painter.setPen(QColor("#d7ecff"))
        painter.drawText(rect.left(), rect.top() - 2, "Hotspot-карта")
        painter.end()

    def _tile_at(self, point: QPoint) -> Optional[tuple[int, int]]:
        rect = self.contentsRect().adjusted(10, 10, -10, -10)
        if not rect.contains(point):
            return None
        rows, cols = self._grid.shape
        tile_w = rect.width() / max(1, cols)
        tile_h = rect.height() / max(1, rows)
        col = min(cols - 1, max(0, int((point.x() - rect.left()) / max(1.0, tile_w))))
        row = min(rows - 1, max(0, int((point.y() - rect.top()) / max(1.0, tile_h))))
        return row, col


def _hotspot_color(value: float) -> QColor:
    v = max(0.0, min(1.0, value))
    r = int(30 + 225 * v)
    g = int(68 + 190 * (1.0 - abs(v - 0.5) * 1.8))
    b = int(95 + 140 * (1.0 - v))
    return QColor(r, max(40, min(255, g)), max(30, min(255, b)))
