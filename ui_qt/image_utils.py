from __future__ import annotations

from io import BytesIO

from PIL import Image
from PySide6.QtGui import QImage, QPixmap


def pil_to_qimage(image: Image.Image) -> QImage:
    rgb = image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimg = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)
    return qimg.copy()


def pil_to_pixmap(image: Image.Image) -> QPixmap:
    return QPixmap.fromImage(pil_to_qimage(image))


def pixmap_to_pil(pixmap: QPixmap) -> Image.Image:
    buffer = BytesIO()
    qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    ptr = qimg.bits()
    arr = bytes(ptr[: qimg.width() * qimg.height() * 3])
    return Image.frombytes("RGB", (qimg.width(), qimg.height()), arr)
