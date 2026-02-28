from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class ImageGraphicsView(QGraphicsView):
    pixelClicked = Signal(int, int)
    pixelHovered = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._cross_v = QGraphicsLineItem()
        self._cross_h = QGraphicsLineItem()
        pen = QPen(QColor("#ffd166"))
        pen.setWidth(1)
        self._cross_v.setPen(pen)
        self._cross_h.setPen(pen)
        self._cross_v.setZValue(3)
        self._cross_h.setZValue(3)
        self._scene.addItem(self._cross_v)
        self._scene.addItem(self._cross_h)
        self._cross_v.hide()
        self._cross_h.hide()
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._has_image = False
        self._base_rect = QRectF()
        self.setMouseTracking(True)

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        if pixmap is None or pixmap.isNull():
            self._pixmap_item.setPixmap(QPixmap())
            self._scene.setSceneRect(QRectF())
            self._has_image = False
            self.clear_probe_point()
            self.viewport().update()
            return
        self._pixmap_item.setPixmap(pixmap)
        rect = QRectF(pixmap.rect())
        self._scene.setSceneRect(rect)
        self._base_rect = rect
        self._has_image = True
        self.resetTransform()
        self.fitInView(self._base_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if not self._has_image:
            return super().wheelEvent(event)
        factor = 1.15 if event.angleDelta().y() > 0 else 0.87
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._has_image and self.transform().isIdentity():
            self.fitInView(self._base_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event):
        if self._has_image:
            point = self._scene_point(event.position())
            if point is not None:
                self.pixelClicked.emit(point[0], point[1])
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._has_image:
            point = self._scene_point(event.position())
            if point is not None:
                self.pixelHovered.emit(point[0], point[1])
        super().mouseMoveEvent(event)

    def set_probe_point(self, x: int, y: int) -> None:
        if not self._has_image:
            self.clear_probe_point()
            return
        px = max(0, min(int(x), int(self._base_rect.width()) - 1))
        py = max(0, min(int(y), int(self._base_rect.height()) - 1))
        self._cross_v.setLine(px, 0, px, self._base_rect.height())
        self._cross_h.setLine(0, py, self._base_rect.width(), py)
        self._cross_v.show()
        self._cross_h.show()

    def clear_probe_point(self) -> None:
        self._cross_v.hide()
        self._cross_h.hide()

    def center_on_pixel(self, x: int, y: int) -> None:
        if not self._has_image:
            return
        self.centerOn(float(x), float(y))
        self.set_probe_point(x, y)

    def _scene_point(self, position) -> tuple[int, int] | None:
        mapped = self.mapToScene(position.toPoint())
        if not self._base_rect.contains(mapped):
            return None
        return int(mapped.x()), int(mapped.y())
