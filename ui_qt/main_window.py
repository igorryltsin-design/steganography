from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QRadioButton,
    QScrollArea,
    QSlider,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QPushButton,
)
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from core.analysis import run_attack_suite, run_mode_benchmark
from core.proof_pack import export_proof_pack
from core.report import build_stegano_report
from core.risk import chi_square_lsb_test, evaluate_risk
from core.stego import decode_text_from_image, encode_text_into_image, max_message_bytes
from core.visual_analysis import build_analysis_preview, probe_pixel
from ui_qt.dialogs import (
    AttackLabDialog,
    BenchmarkDialog,
    CompareDialog,
    DemoResultDialog,
    DemoTimelineDialog,
    HelpDialog,
    ReportPreviewDialog,
)
from ui_qt.graphics_view import ImageGraphicsView
from ui_qt.image_utils import pil_to_pixmap
from ui_qt.strings import STRINGS
from ui_qt.theme import build_stylesheet, get_tokens
from ui_qt.widgets import HotspotMapWidget, PixelInspectorWidget


class _AnalysisWorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)


class _AnalysisWorker(QRunnable):
    def __init__(
        self,
        request_id: int,
        original: Image.Image,
        modified: Image.Image,
        mode: str,
        threshold: int,
        split_ratio: float,
        exact: bool,
        preview_limit: int,
        amplify: int = 20,
    ):
        super().__init__()
        self.signals = _AnalysisWorkerSignals()
        self.request_id = request_id
        self.original = original.copy().convert("RGB")
        self.modified = modified.copy().convert("RGB")
        self.mode = mode
        self.threshold = threshold
        self.split_ratio = split_ratio
        self.exact = exact
        self.preview_limit = preview_limit
        self.amplify = amplify

    def run(self):
        try:
            original = self.original
            modified = self.modified
            if not self.exact:
                original = _downscale_for_preview(original, self.preview_limit)
                modified = _downscale_for_preview(modified, self.preview_limit)
            preview, delta_map, stats, hotspot = build_analysis_preview(
                original,
                modified,
                mode=self.mode,
                threshold=self.threshold,
                split_ratio=self.split_ratio,
                amplify=self.amplify,
            )
            self.signals.result.emit(
                {
                    "request_id": self.request_id,
                    "preview": preview,
                    "delta_map": delta_map,
                    "stats": stats,
                    "hotspot": hotspot,
                    "preview_size": preview.size,
                    "exact": self.exact,
                }
            )
        except Exception as exc:
            self.signals.error.emit(str(exc))


def _downscale_for_preview(image: Image.Image, max_side: int) -> Image.Image:
    max_side = max(256, int(max_side))
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image.copy()
    scale = max_side / float(longest)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return image.resize(size, Image.Resampling.BILINEAR)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.theme_mode = "dark"
        self.tokens = get_tokens(self.theme_mode)
        self.setWindowTitle(STRINGS["app_title"])
        self.resize(1320, 920)
        self.setMinimumSize(1120, 820)

        self.image: Image.Image | None = None
        self.encoded_image: Image.Image | None = None
        self.source_image_path: str | None = None
        self.last_report: Dict[str, Any] | None = None
        self.last_attack_rows: List[Dict[str, Any]] = []
        self.image_signature = "none"
        self.encoded_signature = "none"
        self.ui_mode = "basic"
        self.analysis_mode = "split"
        self.analysis_request_id = 0
        self.analysis_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.analysis_delta_map = None
        self.analysis_hotspot = None
        self.analysis_exact = False
        self.analysis_preview_size: tuple[int, int] | None = None
        self.current_probe_point: tuple[int, int] | None = None
        self._analysis_workers: list[_AnalysisWorker] = []
        self.thread_pool = QThreadPool.globalInstance()
        self.analysis_refresh_timer = QTimer(self)
        self.analysis_refresh_timer.setInterval(55)
        self.analysis_refresh_timer.setSingleShot(True)
        self.analysis_refresh_timer.timeout.connect(self._refresh_visual_analysis)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 16)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel(STRINGS["header"])
        title.setObjectName("Title")
        header.addWidget(title)
        header.addStretch(1)

        self.btn_help = QToolButton()
        self.btn_help.setToolTip("Справка")
        self.btn_help.setText("❔")
        self.btn_help.clicked.connect(self.show_help)
        header.addWidget(self.btn_help)

        self.btn_theme = QToolButton()
        self.btn_theme.setToolTip("Переключить тему")
        self.btn_theme.clicked.connect(self.toggle_theme)
        header.addWidget(self.btn_theme)
        self._refresh_theme_button_icon()

        root.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(12)
        root.addLayout(content, 1)

        left_card = QFrame()
        left_card.setObjectName("Card")
        content.addWidget(left_card, 2)
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(8)

        compare_head = QHBoxLayout()
        left_layout.addLayout(compare_head)
        compare_title = QLabel(STRINGS["compare_title"])
        compare_title.setObjectName("SectionTitle")
        compare_head.addWidget(compare_title)
        compare_head.addStretch(1)
        self.capacity_hint = QLabel("")
        self.capacity_hint.setObjectName("Hint")
        compare_head.addWidget(self.capacity_hint)

        analytics_hud = QFrame()
        analytics_hud.setObjectName("CardSoft")
        left_layout.addWidget(analytics_hud)
        hud = QVBoxLayout(analytics_hud)
        hud.setContentsMargins(12, 10, 12, 10)
        hud.setSpacing(8)

        hud_row = QHBoxLayout()
        self.hud_changed = QLabel("Изменено пикселей: —")
        self.hud_changed.setToolTip("Доля пикселей, в которых после встраивания появился ненулевой сдвиг.")
        hud_row.addWidget(self.hud_changed)
        self.hud_mean = QLabel("Средняя дельта: —")
        self.hud_mean.setToolTip("Среднее изменение по карте различий. Чем ниже, тем ближе изображения.")
        hud_row.addWidget(self.hud_mean)
        self.hud_max = QLabel("Макс. дельта: —")
        self.hud_max.setToolTip("Максимальный сдвиг интенсивности по одному пикселю.")
        hud_row.addWidget(self.hud_max)
        self.hud_hotspot = QLabel("Hotspot: —")
        self.hud_hotspot.setToolTip("Плотность интенсивных изменений по 95-му перцентилю карты различий.")
        hud_row.addWidget(self.hud_hotspot)
        hud_row.addStretch(1)
        self.btn_exact_preview = QPushButton("Точно")
        self.btn_exact_preview.setCheckable(True)
        self.btn_exact_preview.setToolTip("Показывать полноразмерный расчёт без downscaled preview.")
        self.btn_exact_preview.toggled.connect(self._schedule_visual_refresh)
        hud_row.addWidget(self.btn_exact_preview)
        hud.addLayout(hud_row)

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Режим анализа"))
        self.analysis_mode_group = QButtonGroup(self)
        self.analysis_mode_group.setExclusive(True)
        self.analysis_mode_buttons: dict[str, QPushButton] = {}
        for label, mode in [
            ("Разделение", "split"),
            ("Смешивание", "blend"),
            ("Теплокарта", "heatmap"),
            ("Усиление ×20", "amplify20"),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            if mode == "split":
                btn.setChecked(True)
            btn.clicked.connect(lambda _checked=False, selected=mode: self._set_analysis_mode(selected))
            controls_row.addWidget(btn)
            self.analysis_mode_group.addButton(btn)
            self.analysis_mode_buttons[mode] = btn
        controls_row.addSpacing(8)
        controls_row.addWidget(QLabel("Положение"))
        self.analysis_split_slider = QSlider(Qt.Orientation.Horizontal)
        self.analysis_split_slider.setRange(0, 100)
        self.analysis_split_slider.setValue(50)
        self.analysis_split_slider.setToolTip("Граница разделения или коэффициент смешивания.")
        self.analysis_split_slider.valueChanged.connect(self._schedule_visual_refresh)
        controls_row.addWidget(self.analysis_split_slider, 1)
        controls_row.addWidget(QLabel("Порог"))
        self.analysis_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.analysis_threshold_slider.setRange(0, 32)
        self.analysis_threshold_slider.setValue(0)
        self.analysis_threshold_slider.setToolTip("Отсекает очень малые различия при построении карты изменений.")
        self.analysis_threshold_slider.valueChanged.connect(self._schedule_visual_refresh)
        controls_row.addWidget(self.analysis_threshold_slider, 1)
        self.analysis_threshold_label = QLabel("0")
        controls_row.addWidget(self.analysis_threshold_label)
        hud.addLayout(controls_row)

        img_row = QHBoxLayout()
        img_row.setSpacing(10)
        left_layout.addLayout(img_row, 3)

        left_col = QVBoxLayout()
        left_col.addWidget(QLabel(STRINGS["original"]))
        self.original_view = ImageGraphicsView()
        self.original_view.pixelClicked.connect(self._on_probe_clicked)
        self.original_view.pixelHovered.connect(self._on_probe_hovered)
        left_col.addWidget(self.original_view, 1)
        img_row.addLayout(left_col, 1)

        right_col = QVBoxLayout()
        right_col.addWidget(QLabel(STRINGS["modified"]))
        self.modified_view = ImageGraphicsView()
        self.modified_view.pixelClicked.connect(self._on_probe_clicked)
        self.modified_view.pixelHovered.connect(self._on_probe_hovered)
        right_col.addWidget(self.modified_view, 1)
        img_row.addLayout(right_col, 1)

        analysis_box = QFrame()
        analysis_box.setObjectName("CardSoft")
        left_layout.addWidget(analysis_box, 2)
        analysis_layout = QVBoxLayout(analysis_box)
        analysis_layout.setContentsMargins(12, 10, 12, 12)
        analysis_layout.setSpacing(8)
        analysis_head = QHBoxLayout()
        analysis_title = QLabel("Живой анализ изменений")
        analysis_title.setObjectName("SectionTitle")
        analysis_head.addWidget(analysis_title)
        analysis_head.addStretch(1)
        self.analysis_hint = QLabel("Сначала выполните сокрытие сообщения.")
        self.analysis_hint.setObjectName("Hint")
        analysis_head.addWidget(self.analysis_hint)
        analysis_layout.addLayout(analysis_head)
        self.analysis_view = ImageGraphicsView()
        self.analysis_view.pixelClicked.connect(self._on_analysis_probe_clicked)
        self.analysis_view.pixelHovered.connect(self._on_analysis_probe_hovered)
        analysis_layout.addWidget(self.analysis_view, 1)

        right_holder = QFrame()
        right_holder.setObjectName("CardSoft")
        content.addWidget(right_holder, 1)
        right_holder_layout = QVBoxLayout(right_holder)
        right_holder_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_holder_layout.addWidget(scroll)

        controls = QWidget()
        controls.setStyleSheet("background: transparent;")
        scroll.setWidget(controls)
        c = QVBoxLayout(controls)
        c.setContentsMargins(14, 14, 14, 14)
        c.setSpacing(8)

        controls_title = QLabel(STRINGS["controls"])
        controls_title.setObjectName("SectionTitle")
        c.addWidget(controls_title)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Режим"))
        self.btn_basic_mode = QPushButton("Базовый")
        self.btn_basic_mode.setCheckable(True)
        self.btn_basic_mode.setChecked(True)
        self.btn_basic_mode.clicked.connect(lambda: self._set_ui_mode("basic"))
        mode_row.addWidget(self.btn_basic_mode)
        self.btn_expert_mode = QPushButton("Эксперт")
        self.btn_expert_mode.setCheckable(True)
        self.btn_expert_mode.clicked.connect(lambda: self._set_ui_mode("expert"))
        mode_row.addWidget(self.btn_expert_mode)
        mode_row.addStretch(1)
        c.addLayout(mode_row)

        self.expert_hint = QLabel("Базовый режим скрывает продвинутую аналитику. Переключите в «Эксперт» для hotspot и инспектора.")
        self.expert_hint.setObjectName("Hint")
        self.expert_hint.setWordWrap(True)
        c.addWidget(self.expert_hint)

        self.risk_badge = QLabel(STRINGS["risk_unknown"])
        self.risk_badge.setStyleSheet("padding: 6px 10px; border-radius: 8px; background: #375f83; font-weight: 700;")
        c.addWidget(self.risk_badge, 0, Qt.AlignmentFlag.AlignLeft)

        self.risk_reason = QLabel("Оценка риска появится после встраивания.")
        self.risk_reason.setObjectName("Hint")
        self.risk_reason.setWordWrap(True)
        c.addWidget(self.risk_reason)

        self.capacity_bar = QProgressBar()
        self.capacity_bar.setRange(0, 1000)
        self.capacity_bar.setValue(0)
        self.capacity_bar.setFormat("Загрузка контейнера: 0.0%")
        c.addWidget(self.capacity_bar)

        self.robustness_bar = QProgressBar()
        self.robustness_bar.setRange(0, 1000)
        self.robustness_bar.setValue(0)
        self.robustness_bar.setFormat("Устойчивость к атакам: —")
        c.addWidget(self.robustness_bar)

        self.btn_open = QPushButton(STRINGS["open"])
        self.btn_open.clicked.connect(self.open_image)
        c.addWidget(self.btn_open)

        self.btn_save = QPushButton(STRINGS["save_as"])
        self.btn_save.clicked.connect(self.save_encoded_image)
        c.addWidget(self.btn_save)

        self.btn_export_report = QPushButton(STRINGS["export_report"])
        self.btn_export_report.clicked.connect(self.export_report_preview)
        c.addWidget(self.btn_export_report)

        self.btn_export_pack = QPushButton(STRINGS["export_pack"])
        self.btn_export_pack.clicked.connect(self.export_proof_pack_zip)
        c.addWidget(self.btn_export_pack)

        self.btn_hist = QPushButton(STRINGS["show_hist"])
        self.btn_hist.clicked.connect(self.show_lsb_histogram)
        c.addWidget(self.btn_hist)

        self.btn_compare = QPushButton(STRINGS["show_compare"])
        self.btn_compare.clicked.connect(self.open_compare)
        c.addWidget(self.btn_compare)

        self.btn_attack = QPushButton(STRINGS["attack_lab"])
        self.btn_attack.clicked.connect(self.open_attack_lab)
        c.addWidget(self.btn_attack)

        self.btn_benchmark = QPushButton(STRINGS["benchmark"])
        self.btn_benchmark.clicked.connect(self.open_benchmark)
        c.addWidget(self.btn_benchmark)

        self.btn_demo = QPushButton(STRINGS["demo"])
        self.btn_demo.clicked.connect(self.run_demo_mode)
        c.addWidget(self.btn_demo)

        c.addSpacing(6)
        c.addWidget(QLabel(STRINGS["message"]))
        self.message_text = QTextEdit()
        self.message_text.setFixedHeight(120)
        self.message_text.textChanged.connect(self.update_capacity)
        c.addWidget(self.message_text)

        c.addWidget(QLabel(STRINGS["password"]))
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        c.addWidget(self.password)

        c.addWidget(QLabel(STRINGS["bits"]))
        bits_row = QHBoxLayout()
        self.bits_group = QButtonGroup(self)
        for text, val in [("1 бит", 1), ("2 бит", 2), ("3 бит", 3)]:
            rb = QRadioButton(text)
            if val == 1:
                rb.setChecked(True)
            self.bits_group.addButton(rb, val)
            rb.toggled.connect(self.update_capacity)
            bits_row.addWidget(rb)
        bits_row.addStretch(1)
        c.addLayout(bits_row)

        c.addWidget(QLabel(STRINGS["method"]))
        self.method_group = QButtonGroup(self)
        self.rb_seq = QRadioButton("Последовательный (R→G→B)")
        self.rb_int = QRadioButton("Чередование каналов")
        self.rb_seq.setChecked(True)
        self.method_group.addButton(self.rb_seq)
        self.method_group.addButton(self.rb_int)
        c.addWidget(self.rb_seq)
        c.addWidget(self.rb_int)

        action_row = QHBoxLayout()
        self.btn_hide = QPushButton(STRINGS["hide"])
        self.btn_hide.clicked.connect(self.encode_message)
        action_row.addWidget(self.btn_hide)
        self.btn_extract = QPushButton(STRINGS["extract"])
        self.btn_extract.clicked.connect(self.decode_message)
        action_row.addWidget(self.btn_extract)
        c.addLayout(action_row)

        self.status = QLabel(STRINGS["status_wait"])
        self.status.setWordWrap(True)
        self.status.setObjectName("Hint")
        c.addWidget(self.status)

        probe_mode_row = QHBoxLayout()
        self.chk_follow_probe = QCheckBox("Следовать за курсором")
        self.chk_follow_probe.setToolTip("Если включено и точка не зафиксирована, инспектор обновляется при движении мыши.")
        probe_mode_row.addWidget(self.chk_follow_probe)
        self.chk_pin_probe = QCheckBox("Фиксировать точку")
        self.chk_pin_probe.setToolTip("Фиксирует выбранный пиксель. Клик по изображению включает фиксацию.")
        probe_mode_row.addWidget(self.chk_pin_probe)
        probe_mode_row.addStretch(1)
        self.btn_clear_probe = QPushButton("Сбросить точку")
        self.btn_clear_probe.clicked.connect(self._clear_probe)
        probe_mode_row.addWidget(self.btn_clear_probe)
        c.addLayout(probe_mode_row)

        self.hotspot_map = HotspotMapWidget()
        self.hotspot_map.tileActivated.connect(self._focus_hotspot_tile)
        c.addWidget(self.hotspot_map)

        self.pixel_inspector = PixelInspectorWidget()
        c.addWidget(self.pixel_inspector)
        c.addStretch(1)

        self._set_buttons_enabled(False)
        self._show_placeholder(self.original_view, "Откройте изображение")
        self._show_placeholder(self.modified_view, "После встраивания")
        self._show_placeholder(self.analysis_view, "Аналитический слой")
        self.analytics_hud = analytics_hud
        self.analysis_box = analysis_box
        self.expert_widgets = [
            self.analytics_hud,
            self.analysis_box,
            self.btn_hist,
            self.btn_compare,
            self.btn_attack,
            self.btn_benchmark,
            self.hotspot_map,
            self.pixel_inspector,
            self.chk_follow_probe,
            self.chk_pin_probe,
            self.btn_clear_probe,
        ]
        self._apply_ui_mode()

    def _set_buttons_enabled(self, enabled: bool):
        self.btn_save.setEnabled(enabled)
        self.btn_export_report.setEnabled(enabled)
        self.btn_export_pack.setEnabled(enabled)
        self.btn_compare.setEnabled(enabled)
        self.btn_attack.setEnabled(enabled)
        self.btn_benchmark.setEnabled(enabled)

    def _show_placeholder(self, view: ImageGraphicsView, text: str):
        pix = QPixmap(720, 520)
        pix.fill(QColor(self.tokens["canvas"]))
        painter = QPainter(pix)
        painter.setPen(QColor(self.tokens["muted"]))
        f = QFont("Segoe UI", 14)
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, text)
        painter.end()
        view.set_pixmap(pix)

    def _set_ui_mode(self, mode: str):
        self.ui_mode = mode
        self.btn_basic_mode.setChecked(mode == "basic")
        self.btn_expert_mode.setChecked(mode == "expert")
        self._apply_ui_mode()

    def _apply_ui_mode(self):
        expert = self.ui_mode == "expert"
        self.expert_hint.setVisible(not expert)
        for widget in self.expert_widgets:
            widget.setVisible(expert)
        if not expert:
            self._clear_probe()

    def _set_analysis_mode(self, mode: str):
        self.analysis_mode = mode
        self._schedule_visual_refresh()

    def _schedule_visual_refresh(self):
        self.analysis_threshold_label.setText(str(self.analysis_threshold_slider.value()))
        self.analysis_refresh_timer.start()

    def _refresh_visual_analysis(self):
        if self.image is None or self.encoded_image is None:
            self._reset_visual_analytics()
            return
        self.analysis_request_id += 1
        request_id = self.analysis_request_id
        key = (
            self.image_signature,
            self.encoded_signature,
            self.analysis_mode,
            self.analysis_threshold_slider.value(),
            self.analysis_split_slider.value(),
            self.btn_exact_preview.isChecked(),
        )
        if key in self.analysis_cache:
            self._apply_visual_result(self.analysis_cache[key])
            return

        self.analysis_hint.setText("Пересчёт визуальной аналитики...")
        worker = _AnalysisWorker(
            request_id=request_id,
            original=self.image,
            modified=self.encoded_image,
            mode=self.analysis_mode,
            threshold=self.analysis_threshold_slider.value(),
            split_ratio=self.analysis_split_slider.value() / 100.0,
            exact=self.btn_exact_preview.isChecked(),
            preview_limit=1600,
        )
        self._analysis_workers.append(worker)
        worker.signals.result.connect(lambda payload, cache_key=key: self._handle_analysis_result(cache_key, payload))
        worker.signals.error.connect(self._handle_analysis_error)
        self.thread_pool.start(worker)

    def _handle_analysis_result(self, cache_key, payload):
        self._analysis_workers = [w for w in self._analysis_workers if w.request_id != payload.get("request_id")]
        if payload.get("request_id") != self.analysis_request_id:
            return
        self.analysis_cache[cache_key] = payload
        self._apply_visual_result(payload)

    def _handle_analysis_error(self, message: str):
        self._analysis_workers.clear()
        self.analysis_hint.setText(f"Аналитика недоступна: {message}")

    def _apply_visual_result(self, payload: Dict[str, Any]):
        self.analysis_view.set_pixmap(pil_to_pixmap(payload["preview"]))
        self.analysis_delta_map = payload["delta_map"]
        self.analysis_hotspot = payload["hotspot"]
        self.analysis_preview_size = tuple(payload.get("preview_size", (0, 0)))
        stats = payload["stats"]
        self.hud_changed.setText(f"Изменено пикселей: {stats.changed_pct:.2f}%")
        self.hud_mean.setText(f"Средняя дельта: {stats.mean_delta:.3f}")
        self.hud_max.setText(f"Макс. дельта: {stats.max_delta}")
        self.hud_hotspot.setText(f"Hotspot: {stats.hotspot_score:.2f}")
        self.analysis_exact = bool(payload.get("exact"))
        exact_label = "полный расчёт" if self.analysis_exact else "быстрый preview"
        self.analysis_hint.setText(f"{self._analysis_mode_label()} · порог {stats.threshold} · {exact_label}")
        if self.analysis_hotspot is not None and self.image is not None:
            self.hotspot_map.set_grid(self.analysis_hotspot, self.image.size)
        if self.current_probe_point is not None:
            self._update_probe(self.current_probe_point[0], self.current_probe_point[1], set_fixed=False)

    def _analysis_mode_label(self) -> str:
        return {
            "split": "Разделение",
            "blend": "Смешивание",
            "heatmap": "Теплокарта",
            "amplify20": "Усиление ×20",
        }.get(self.analysis_mode, "Анализ")

    def _reset_visual_analytics(self):
        self.hud_changed.setText("Изменено пикселей: —")
        self.hud_mean.setText("Средняя дельта: —")
        self.hud_max.setText("Макс. дельта: —")
        self.hud_hotspot.setText("Hotspot: —")
        self.analysis_hint.setText("Сначала выполните сокрытие сообщения.")
        self.hotspot_map.clear()
        self.pixel_inspector.clear()
        self.current_probe_point = None
        self.analysis_delta_map = None
        self.analysis_hotspot = None
        self.analysis_preview_size = None
        self._show_placeholder(self.analysis_view, "Аналитический слой")

    def _on_probe_clicked(self, x: int, y: int):
        self.chk_pin_probe.setChecked(True)
        self._update_probe(x, y, set_fixed=True)

    def _on_probe_hovered(self, x: int, y: int):
        if self.chk_follow_probe.isChecked() and not self.chk_pin_probe.isChecked():
            self._update_probe(x, y, set_fixed=False)

    def _on_analysis_probe_clicked(self, x: int, y: int):
        ox, oy = self._original_point_from_analysis(x, y)
        self._on_probe_clicked(ox, oy)

    def _on_analysis_probe_hovered(self, x: int, y: int):
        ox, oy = self._original_point_from_analysis(x, y)
        self._on_probe_hovered(ox, oy)

    def _update_probe(self, x: int, y: int, set_fixed: bool):
        if self.image is None or self.encoded_image is None:
            return
        point = self._clamp_point(x, y)
        self.current_probe_point = point
        probe = probe_pixel(self.image, self.encoded_image, point[0], point[1])
        self.pixel_inspector.set_probe_data(probe, self.image, self.encoded_image)
        self.original_view.set_probe_point(point[0], point[1])
        self.modified_view.set_probe_point(point[0], point[1])
        ax, ay = self._analysis_point_from_original(point[0], point[1])
        self.analysis_view.set_probe_point(ax, ay)
        self._select_hotspot_for_point(point[0], point[1])
        if set_fixed:
            self.status.setText(f"Инспектор: точка x={point[0]}, y={point[1]}")

    def _clear_probe(self):
        self.chk_pin_probe.setChecked(False)
        self.current_probe_point = None
        self.original_view.clear_probe_point()
        self.modified_view.clear_probe_point()
        self.analysis_view.clear_probe_point()
        self.hotspot_map.clear()
        if self.analysis_hotspot is not None and self.image is not None:
            self.hotspot_map.set_grid(self.analysis_hotspot, self.image.size)
        self.pixel_inspector.clear()

    def _focus_hotspot_tile(self, row: int, col: int):
        if self.image is None:
            return
        grid = self.analysis_hotspot
        if grid is None:
            return
        rows, cols = grid.shape
        x = int((col + 0.5) * self.image.width / max(1, cols))
        y = int((row + 0.5) * self.image.height / max(1, rows))
        point = self._clamp_point(x, y)
        self.original_view.center_on_pixel(*point)
        self.modified_view.center_on_pixel(*point)
        ax, ay = self._analysis_point_from_original(point[0], point[1])
        self.analysis_view.center_on_pixel(ax, ay)
        self.chk_pin_probe.setChecked(True)
        self._update_probe(point[0], point[1], set_fixed=True)

    def _select_hotspot_for_point(self, x: int, y: int):
        if self.analysis_hotspot is None or self.image is None:
            return
        rows, cols = self.analysis_hotspot.shape
        col = min(cols - 1, max(0, int(x * cols / max(1, self.image.width))))
        row = min(rows - 1, max(0, int(y * rows / max(1, self.image.height))))
        self.hotspot_map.set_selected(row, col)

    def _clamp_point(self, x: int, y: int) -> tuple[int, int]:
        if self.image is None:
            return 0, 0
        px = max(0, min(int(x), self.image.width - 1))
        py = max(0, min(int(y), self.image.height - 1))
        return px, py

    def _analysis_point_from_original(self, x: int, y: int) -> tuple[int, int]:
        if self.image is None or not self.analysis_preview_size:
            return x, y
        pw, ph = self.analysis_preview_size
        if pw <= 0 or ph <= 0:
            return x, y
        ax = int(round(x * pw / max(1, self.image.width)))
        ay = int(round(y * ph / max(1, self.image.height)))
        return max(0, min(ax, pw - 1)), max(0, min(ay, ph - 1))

    def _original_point_from_analysis(self, x: int, y: int) -> tuple[int, int]:
        if self.image is None or not self.analysis_preview_size:
            return self._clamp_point(x, y)
        pw, ph = self.analysis_preview_size
        if pw <= 0 or ph <= 0:
            return self._clamp_point(x, y)
        ox = int(round(x * self.image.width / max(1, pw)))
        oy = int(round(y * self.image.height / max(1, ph)))
        return self._clamp_point(ox, oy)

    def _image_signature(self, image: Image.Image | None) -> str:
        if image is None:
            return "none"
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
        digest = hashlib.sha1(arr.tobytes()).hexdigest()[:16]
        return f"{image.width}x{image.height}:{digest}"

    def _ensure_probe_for_export(self):
        if self.image is None or self.encoded_image is None:
            return
        if self.current_probe_point is None:
            self._update_probe(self.image.width // 2, self.image.height // 2, set_fixed=False)

    def _ensure_visual_analysis_ready_sync(self):
        if self.image is None or self.encoded_image is None:
            return
        preview, delta_map, stats, hotspot = build_analysis_preview(
            self.image,
            self.encoded_image,
            mode=self.analysis_mode,
            threshold=self.analysis_threshold_slider.value(),
            split_ratio=self.analysis_split_slider.value() / 100.0,
            amplify=20,
        )
        payload = {
            "preview": preview,
            "delta_map": delta_map,
            "stats": stats,
            "hotspot": hotspot,
            "preview_size": preview.size,
            "exact": True,
            "request_id": self.analysis_request_id,
        }
        self._apply_visual_result(payload)
        self._ensure_probe_for_export()

    def _widget_png_bytes(self, widget: QWidget) -> bytes:
        pixmap = widget.grab()
        arr = QByteArray()
        buffer = QBuffer(arr)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "PNG")
        buffer.close()
        return bytes(arr.data())

    def _write_visual_artifacts_near_report(self, report_path: str) -> Dict[str, str]:
        if self.image is None or self.encoded_image is None or self.last_report is None:
            return {}
        captures = self._capture_visual_artifact_bytes()
        base = Path(report_path)
        hotspot_path = str(base.with_name(f"{base.stem}_hotspot.png"))
        inspector_path = str(base.with_name(f"{base.stem}_inspector.png"))
        with open(hotspot_path, "wb") as f:
            f.write(captures["hotspot"])
        with open(inspector_path, "wb") as f:
            f.write(captures["inspector"])
        self.last_report.setdefault("visual_artifacts", {}).update(
            {
                "hotspot": hotspot_path,
                "inspector": inspector_path,
            }
        )
        return {"hotspot": hotspot_path, "inspector": inspector_path}

    def _proof_pack_extra_artifacts(self) -> Dict[str, bytes]:
        if self.image is None or self.encoded_image is None:
            return {}
        captures = self._capture_visual_artifact_bytes()
        return {
            "proof_pack/hotspot.png": captures["hotspot"],
            "proof_pack/inspector.png": captures["inspector"],
        }

    def _capture_visual_artifact_bytes(self) -> Dict[str, bytes]:
        self._ensure_visual_analysis_ready_sync()
        restore_mode = self.ui_mode
        saved_point = self.current_probe_point
        saved_pin = self.chk_pin_probe.isChecked()
        if restore_mode != "expert":
            self._set_ui_mode("expert")
            QApplication.processEvents()
        captures = {
            "hotspot": self._widget_png_bytes(self.hotspot_map),
            "inspector": self._widget_png_bytes(self.pixel_inspector),
        }
        if restore_mode != "expert":
            self._set_ui_mode(restore_mode)
            QApplication.processEvents()
            self.current_probe_point = saved_point
            self.chk_pin_probe.setChecked(saved_pin)
        return captures

    def _refresh_theme_button_icon(self):
        if self.theme_mode == "dark":
            self.btn_theme.setText("☀")
            self.btn_theme.setToolTip("Переключить на светлую тему")
        else:
            self.btn_theme.setText("🌙")
            self.btn_theme.setToolTip("Переключить на тёмную тему")

    def toggle_theme(self):
        self.theme_mode = "light" if self.theme_mode == "dark" else "dark"
        self.tokens = get_tokens(self.theme_mode)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_stylesheet(self.theme_mode))
        self._refresh_theme_button_icon()

        if self.image is None:
            self._show_placeholder(self.original_view, "Откройте изображение")
        if self.encoded_image is None:
            self._show_placeholder(self.modified_view, "После встраивания")
            self._show_placeholder(self.analysis_view, "Аналитический слой")

    def show_help(self):
        dlg = HelpDialog(parent=self)
        dlg.exec()

    def _method_value(self) -> str:
        return "sequential" if self.rb_seq.isChecked() else "interleaved"

    def _bits_value(self) -> int:
        return self.bits_group.checkedId() if self.bits_group.checkedId() in {1, 2, 3} else 1

    def update_capacity(self):
        if not self.image:
            self.capacity_hint.setText("")
            self.capacity_bar.setValue(0)
            self.capacity_bar.setFormat("Загрузка контейнера: 0.0%")
            self.robustness_bar.setValue(0)
            self.robustness_bar.setFormat("Устойчивость к атакам: —")
            self.risk_badge.setText("Риск: —")
            self.risk_badge.setStyleSheet("padding: 6px 10px; border-radius: 8px; background: #375f83; font-weight: 700;")
            self.risk_reason.setText("Оценка риска появится после встраивания.")
            self._reset_visual_analytics()
            return
        bits = self._bits_value()
        max_bytes = max_message_bytes(self.image.size, bits)
        payload_len = len(self.message_text.toPlainText().strip().encode("utf-8"))
        usage = (float(payload_len) / float(max_bytes)) if max_bytes else 0.0
        self.capacity_hint.setText(f"≈ {max_bytes:,} байт ({max_bytes / 1024:.1f} КБ) при {bits} бит/канал · сообщение {payload_len} байт")
        self.capacity_bar.setValue(int(max(0.0, min(1.0, usage)) * 1000))
        self.capacity_bar.setFormat(f"Загрузка контейнера: {usage * 100:.1f}%")
        self.risk_reason.setText("Риск будет пересчитан после сокрытия сообщения.")

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "", "Изображения (*.png *.bmp *.jpg *.jpeg *.gif)")
        if not path:
            return
        try:
            self.image = Image.open(path).convert("RGB")
            self.image_signature = self._image_signature(self.image)
            self.source_image_path = path
            self.encoded_image = None
            self.encoded_signature = "none"
            self.last_report = None
            self.last_attack_rows = []
            self.original_view.set_pixmap(pil_to_pixmap(self.image))
            self._show_placeholder(self.modified_view, "После встраивания")
            self._set_buttons_enabled(False)
            self.robustness_bar.setValue(0)
            self.robustness_bar.setFormat("Устойчивость к атакам: —")
            self.status.setText(f"Открыто: {path}")
            self.analysis_cache.clear()
            self._reset_visual_analytics()
            self.decode_message(silent=True)
            self.update_capacity()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть:\n{exc}")

    def encode_message(self):
        if not self.image:
            QMessageBox.warning(self, "Нет изображения", "Сначала откройте изображение.")
            return

        msg = self.message_text.toPlainText().strip()
        if not msg:
            QMessageBox.warning(self, "Пустое сообщение", "Введите текст для сокрытия.")
            return

        bits = self._bits_value()
        method = self._method_value()
        payload_len = len(msg.encode("utf-8"))
        capacity = max_message_bytes(self.image.size, bits)
        if payload_len > capacity:
            QMessageBox.critical(self, "Ошибка", f"Сообщение слишком большое ({payload_len} байт > {capacity})")
            return

        self.status.setText("Кодирование...")
        try:
            encoded = encode_text_into_image(self.image, msg, self.password.text(), bits, method)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка кодирования", str(exc))
            return

        self.encoded_image = encoded
        self.encoded_signature = self._image_signature(encoded)
        self.modified_view.set_pixmap(pil_to_pixmap(encoded))
        self._set_buttons_enabled(True)

        psnr_v = mse_v = ssim_v = None
        changed_pixels_pct = 0.0
        mean_abs_delta = 0.0
        max_abs_delta = 0
        metrics_error = None
        try:
            o = np.array(self.image)
            e = np.array(encoded)
            psnr_v = float(psnr(o, e, data_range=255))
            mse_v = float(mse(o, e))
            ssim_v = float(ssim(o, e, multichannel=True, data_range=255, channel_axis=-1))
            abs_delta = np.abs(e.astype(np.int16) - o.astype(np.int16))
            changed_mask = np.any(abs_delta > 0, axis=2)
            changed_pixels_pct = float(np.mean(changed_mask) * 100.0)
            mean_abs_delta = float(np.mean(abs_delta))
            max_abs_delta = int(np.max(abs_delta))
        except Exception as exc:
            metrics_error = str(exc)

        stats_orig = chi_square_lsb_test(self.image)
        stats_stego = chi_square_lsb_test(encoded)
        usage = (float(payload_len) / float(capacity)) if capacity else 0.0
        risk_level, risk_reason = evaluate_risk(usage, stats_orig, stats_stego)

        recommendation = self._build_recommendation(risk_level, usage, method, bits)
        self.last_report = build_stegano_report(
            source_image_path=self.source_image_path,
            image_size=self.image.size,
            method=method,
            bits_per_channel=bits,
            password_used=bool(self.password.text()),
            message_chars=len(msg),
            message_bytes_utf8=payload_len,
            capacity_bytes=capacity,
            psnr_db=psnr_v,
            mse=mse_v,
            ssim=ssim_v,
            metrics_error=metrics_error,
            chi_original=stats_orig,
            chi_stego=stats_stego,
            risk_level=risk_level,
            risk_reason=risk_reason,
            demo_summary={"steps": ["open", "encode"], "completed": False},
            robustness_score=None,
            visual_artifacts={},
            recommendation=recommendation,
        )

        self._set_risk_badge(risk_level, risk_reason)
        self.status.setText(
            "\n".join(
                [
                    "Сообщение спрятано ✓",
                    f"PSNR={psnr_v:.2f} дБ, MSE={mse_v:.4f}, SSIM={ssim_v:.4f}" if psnr_v is not None else f"Метрики: {metrics_error}",
                    (
                        f"Изменено пикселей: {changed_pixels_pct:.1f}% | "
                        f"средний сдвиг: {mean_abs_delta:.3f} | макс. сдвиг: {max_abs_delta}"
                    ),
                    f"Оригинал χ²={stats_orig['chi2']:.2f}, Стего χ²={stats_stego['chi2']:.2f}",
                    f"Риск: {risk_level} — {risk_reason}",
                    "Внешне отличия могут быть незаметны — это норма для LSB-стеганографии.",
                ]
            )
        )
        self.update_capacity()
        self._schedule_visual_refresh()

    def _build_recommendation(self, risk: str, usage: float, method: str, bits: int) -> str:
        if risk == "HIGH":
            return "Снизьте биты на канал до 1 и сократите длину сообщения для уменьшения детектируемости."
        if risk == "MEDIUM":
            return "Режим рабочий, но лучше немного уменьшить загрузку контейнера."
        method_ru = "последовательный" if method == "sequential" else "чередование каналов"
        return f"Оптимальный режим: {method_ru}, {bits} бит/канал."

    def _set_risk_badge(self, level: str, reason: str):
        color = {"LOW": "#2c9d73", "MEDIUM": "#b3872f", "HIGH": "#b33b4d"}.get(level, "#375f83")
        self.risk_badge.setText(f"Риск: {level}")
        self.risk_badge.setStyleSheet(f"padding: 6px 10px; border-radius: 8px; background: {color}; font-weight: 700;")
        self.risk_reason.setText(reason)

    def decode_message(self, silent: bool = False):
        img = self.encoded_image if self.encoded_image is not None else self.image
        if img is None:
            if not silent:
                QMessageBox.warning(self, "Нет изображения", "Сначала откройте изображение.")
            return
        try:
            text = decode_text_from_image(img, self.password.text(), self._bits_value(), self._method_value())
            self.message_text.setPlainText(text)
            if not silent:
                self.status.setText("Сообщение извлечено ✓")
        except Exception as exc:
            if not silent:
                self.status.setText(f"Не удалось извлечь: {exc}")

    def save_encoded_image(self):
        if self.encoded_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение", "", "PNG (*.png);;BMP (*.bmp);;JPEG (*.jpg *.jpeg)")
        if not path:
            return
        ext = Path(path).suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            ok = QMessageBox.question(
                self,
                "JPEG и потери",
                "JPEG использует сжатие с потерями и может повредить сообщение. Сохранить в JPEG?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ok != QMessageBox.StandardButton.Yes:
                return
        try:
            self.encoded_image.save(path)
            if self.last_report:
                self.last_report.setdefault("artifacts", {})["encoded_image_path"] = path
            self.status.setText(f"Сохранено: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))

    def export_report_preview(self):
        if not self.last_report:
            QMessageBox.warning(self, "Нет отчёта", "Сначала выполните сокрытие сообщения.")
            return
        dlg = ReportPreviewDialog(
            self.last_report,
            on_save_json=self.save_report_json,
            on_save_txt=self.save_report_txt,
            parent=self,
        )
        dlg.exec()

    def save_report_json(self):
        if not self.last_report:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчёт JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.last_report.setdefault("artifacts", {})["report_path"] = path
            self.last_report.setdefault("artifacts", {})["report_format"] = "json"
            self._write_visual_artifacts_near_report(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.last_report, f, ensure_ascii=False, indent=2, sort_keys=True)
            self.status.setText(f"Отчёт сохранён: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))

    def save_report_txt(self):
        if not self.last_report:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчёт TXT", "", "Text (*.txt)")
        if not path:
            return
        try:
            from core.report import render_report_text

            self.last_report.setdefault("artifacts", {})["report_path"] = path
            self.last_report.setdefault("artifacts", {})["report_format"] = "txt"
            self._write_visual_artifacts_near_report(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(render_report_text(self.last_report))
            self.status.setText(f"Отчёт сохранён: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))

    def open_compare(self):
        if self.image is None or self.encoded_image is None:
            QMessageBox.warning(self, "Нет данных", "Сначала выполните сокрытие сообщения.")
            return
        dlg = CompareDialog(self.image, self.encoded_image, parent=self)
        dlg.exec()

    def open_attack_lab(self):
        if self.encoded_image is None:
            QMessageBox.warning(self, "Нет данных", "Сначала выполните сокрытие сообщения.")
            return
        expected = self.message_text.toPlainText().strip()
        if not expected:
            QMessageBox.warning(self, "Нет сообщения", "Введите сообщение для проверки атак.")
            return
        try:
            self._run_attack_analysis(show_dialog=True)
        except Exception as exc:
            QMessageBox.warning(self, "Симулятор атак", str(exc))

    def open_benchmark(self):
        if self.image is None:
            QMessageBox.warning(self, "Нет изображения", "Сначала откройте изображение.")
            return
        msg = self.message_text.toPlainText().strip()
        if not msg:
            QMessageBox.warning(self, "Нет сообщения", "Введите сообщение для автосравнения.")
            return
        try:
            self._run_benchmark_analysis(show_dialog=True)
        except Exception as exc:
            QMessageBox.warning(self, "Сравнение режимов", str(exc))

    def auto_select_mode(self, auto_encode: bool = False):
        if self.image is None:
            QMessageBox.warning(self, "Нет изображения", "Сначала откройте изображение.")
            return
        msg = self.message_text.toPlainText().strip()
        if not msg:
            QMessageBox.warning(self, "Нет сообщения", "Введите сообщение для автоподбора режима.")
            return
        best = self._select_best_mode(msg)

        if best is None:
            QMessageBox.warning(self, "Автоподбор", "Не удалось подобрать режим: сообщение не помещается или возникла ошибка.")
            return

        self._set_mode_controls(best["method"], int(best["bits"]))
        method_ru = "Последовательный" if best["method"] == "sequential" else "Чередование"
        recommendation = (
            f"Автовыбор: {method_ru} / {best['bits']} бит, "
            f"SSIM={best['ssim']:.4f}, PSNR={best['psnr']:.2f}, "
            f"риск {best['risk_level']}."
        )
        self.status.setText(recommendation)
        if self.last_report is not None:
            self.last_report["recommendation"] = recommendation
            self.last_report["demo_summary"] = {"steps": ["open", "auto_mode"], "completed": False}

        if auto_encode:
            self.encode_message()

    def _select_best_mode(self, message_text: str):
        if self.image is None:
            return None
        payload_len = len(message_text.encode("utf-8"))
        stats_orig = chi_square_lsb_test(self.image)
        best = None
        orig_np = np.array(self.image)

        for method in ("sequential", "interleaved"):
            for bits in (1, 2, 3):
                capacity = max_message_bytes(self.image.size, bits)
                if payload_len > capacity:
                    continue
                try:
                    encoded = encode_text_into_image(self.image, message_text, self.password.text(), bits, method)
                    enc_np = np.array(encoded)
                    psnr_v = float(psnr(orig_np, enc_np, data_range=255))
                    ssim_v = float(ssim(orig_np, enc_np, multichannel=True, data_range=255, channel_axis=-1))
                    usage = (float(payload_len) / float(capacity)) if capacity else 1.0
                    risk_level, risk_reason = evaluate_risk(usage, stats_orig, chi_square_lsb_test(encoded))
                    risk_penalty = {"LOW": 0.0, "MEDIUM": 8.0, "HIGH": 18.0}.get(risk_level, 12.0)
                    score = (ssim_v * 100.0) + (psnr_v * 0.15) - (usage * 25.0) - risk_penalty
                    cand = {
                        "method": method,
                        "bits": bits,
                        "psnr": psnr_v,
                        "ssim": ssim_v,
                        "usage": usage,
                        "risk_level": risk_level,
                        "risk_reason": risk_reason,
                        "score": score,
                    }
                    if best is None or cand["score"] > best["score"]:
                        best = cand
                except Exception:
                    continue
        return best

    def _set_mode_controls(self, method: str, bits: int):
        if method == "sequential":
            self.rb_seq.setChecked(True)
        else:
            self.rb_int.setChecked(True)
        for btn in self.bits_group.buttons():
            if self.bits_group.id(btn) == bits:
                btn.setChecked(True)
                break
        self.update_capacity()

    def run_demo_mode(self):
        if self.image is None:
            QMessageBox.warning(self, "Демо-режим", "Сначала откройте изображение.")
            return
        if not self.message_text.toPlainText().strip():
            self.message_text.setPlainText("Демо-режим: проверка LSB-стеганографии с отчётом, теплокартой и атаками.")
        steps = [
            ("Автоподбор режима и кодирование", self._demo_step_auto_encode),
            ("Визуальная проверка (теплокарта)", self._demo_step_compare_prep),
            ("Проверка устойчивости к атакам", self._demo_step_attacks),
            ("Сравнение режимов и рекомендация", self._demo_step_benchmark),
            ("Финализация отчёта демо", self._demo_step_finalize),
        ]
        timeline = DemoTimelineDialog(steps, parent=self)
        timeline.exec()
        if not timeline.completed:
            return

        risk_level = self.last_report.get("risk", {}).get("level", "-") if self.last_report else "-"
        risk_reason = self.last_report.get("risk", {}).get("reason", "-") if self.last_report else "-"
        robustness = self.last_report.get("robustness_score") if self.last_report else None
        recommendation = self.last_report.get("recommendation", "-") if self.last_report else "-"
        result = DemoResultDialog(
            risk_level=risk_level,
            risk_reason=risk_reason,
            robustness_score=robustness,
            recommendation=recommendation,
            on_open_compare=self.open_compare,
            on_open_attacks=self.open_attack_lab,
            on_open_benchmark=self.open_benchmark,
            on_export_pack=self.export_proof_pack_zip,
            parent=self,
        )
        result.exec()

    def _run_attack_analysis(self, show_dialog: bool = True):
        if self.encoded_image is None:
            raise ValueError("Нет данных: сначала выполните сокрытие сообщения.")
        expected = self.message_text.toPlainText().strip()
        if not expected:
            raise ValueError("Нет сообщения для проверки атак.")
        rows = run_attack_suite(self.encoded_image, expected, self.password.text(), self._bits_value(), self._method_value())
        self.last_attack_rows = [dict(r) for r in rows]
        passed = sum(1 for r in rows if r.get("success"))
        score = round(100.0 * passed / max(1, len(rows)), 2)
        if self.last_report is not None:
            self.last_report["robustness_score"] = score
            self.last_report["demo_summary"] = {"steps": ["open", "encode", "attacks"], "completed": False}
        self.robustness_bar.setValue(int(max(0.0, min(100.0, score)) * 10))
        self.robustness_bar.setFormat(f"Устойчивость к атакам: {score:.1f}%")
        if show_dialog:
            dlg = AttackLabDialog(rows, parent=self)
            dlg.exec()
        return rows, score

    def _run_benchmark_analysis(self, show_dialog: bool = True):
        if self.image is None:
            raise ValueError("Нет изображения.")
        msg = self.message_text.toPlainText().strip()
        if not msg:
            raise ValueError("Нет сообщения для автосравнения.")
        rows = run_mode_benchmark(self.image, msg, self.password.text())
        if show_dialog:
            dlg = BenchmarkDialog(rows, parent=self)
            dlg.exec()

        best = None
        for row in rows:
            if row.get("decode_ok") and row.get("ssim") is not None and (best is None or row["ssim"] > best["ssim"]):
                best = row
        if self.last_report is not None and best is not None:
            method = "Последовательный" if best["method"] == "sequential" else "Чередование"
            self.last_report["recommendation"] = (
                f"Для качества изображения лучше {method} / {best['bits']} бит: SSIM={best['ssim']:.4f}, PSNR={best['psnr_db']:.2f}."
            )
        return rows, best

    def _demo_step_auto_encode(self) -> str:
        msg = self.message_text.toPlainText().strip()
        if not msg:
            msg = "Демо-режим: проверка LSB-стеганографии с отчётом, теплокартой и атаками."
            self.message_text.setPlainText(msg)
        best = self._select_best_mode(msg)
        if best is None:
            raise ValueError("Автоподбор не нашел подходящий режим.")
        self._set_mode_controls(best["method"], int(best["bits"]))
        self.encode_message()
        if self.encoded_image is None:
            raise ValueError("Не удалось выполнить кодирование.")
        method_ru = "Последовательный" if best["method"] == "sequential" else "Чередование"
        return f"{method_ru}, {best['bits']} бит (SSIM={best['ssim']:.4f}, риск {best['risk_level']})."

    def _demo_step_compare_prep(self) -> str:
        if self.image is None or self.encoded_image is None:
            raise ValueError("Нет пары изображений для сравнения.")
        return "Пара изображений готова для окна До/После + Теплокарта."

    def _demo_step_attacks(self) -> str:
        _rows, score = self._run_attack_analysis(show_dialog=False)
        return f"Устойчивость {score:.1f}%."

    def _demo_step_benchmark(self) -> str:
        _rows, best = self._run_benchmark_analysis(show_dialog=False)
        if best is None:
            return "Нет успешного режима в benchmark."
        method = "Последовательный" if best["method"] == "sequential" else "Чередование"
        return f"Лучший режим: {method}/{best['bits']} бит."

    def _demo_step_finalize(self) -> str:
        if self.last_report is not None:
            self.last_report["demo_summary"] = {
                "steps": ["open", "auto_mode", "encode", "compare_prep", "attacks", "benchmark"],
                "completed": True,
            }
        return "Отчёт обновлён, демо завершено."

    def export_proof_pack_zip(self):
        if self.last_report is None or self.image is None or self.encoded_image is None:
            QMessageBox.warning(self, "Нет данных", "Сначала выполните сокрытие сообщения.")
            return

        attacks = self.last_attack_rows
        if not attacks:
            expected = self.message_text.toPlainText().strip()
            if expected:
                attacks = run_attack_suite(
                    self.encoded_image,
                    expected,
                    self.password.text(),
                    self._bits_value(),
                    self._method_value(),
                )
                self.last_attack_rows = [dict(r) for r in attacks]

        path, _ = QFileDialog.getSaveFileName(self, "Сохранить архив данных", "archive_data.zip", "ZIP (*.zip)")
        if not path:
            return
        try:
            export_proof_pack(
                path,
                self.last_report,
                self.image,
                self.encoded_image,
                self.last_attack_rows,
                extra_png_artifacts=self._proof_pack_extra_artifacts(),
            )
            self.status.setText(f"Архив данных сохранён: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def show_lsb_histogram(self):
        if self.image is None:
            QMessageBox.warning(self, "Нет изображения", "Сначала откройте изображение.")
            return

        try:
            import pyqtgraph as pg
            from PySide6.QtWidgets import QDialog
            dlg = QDialog(self)
            dlg.setWindowTitle("Гистограмма LSB")
            dlg.resize(860, 420)
            layout = QHBoxLayout(dlg)
            plots = pg.GraphicsLayoutWidget()
            layout.addWidget(plots)

            def lsb_counts(img: Image.Image):
                arr = np.array(img.convert("RGB"), dtype=np.uint8)
                out = []
                for c in range(3):
                    lsb = arr[:, :, c] & 1
                    out.append((int(np.sum(lsb == 0)), int(np.sum(lsb == 1))))
                return out

            def draw(plot, title, counts):
                plot.addLegend()
                plot.setTitle(title)
                x = np.array([0, 1], dtype=float)
                for idx, (_name, color) in enumerate([("R", (255, 100, 100)), ("G", (100, 255, 140)), ("B", (100, 180, 255))]):
                    y = counts[idx]
                    bars = pg.BarGraphItem(x=x + (idx - 1) * 0.18, height=y, width=0.16, brush=color)
                    plot.addItem(bars)
                plot.getAxis("bottom").setTicks([[(0, "0"), (1, "1")]])
                plot.showGrid(x=True, y=True, alpha=0.2)

            p1 = plots.addPlot(row=0, col=0)
            draw(p1, "Оригинал", lsb_counts(self.image))
            p2 = plots.addPlot(row=0, col=1)
            if self.encoded_image is not None:
                draw(p2, "После встраивания", lsb_counts(self.encoded_image))
            else:
                p2.setTitle("После встраивания")

            dlg.exec()
            return
        except Exception:
            pass

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            from PySide6.QtWidgets import QDialog

            dlg = QDialog(self)
            dlg.setWindowTitle("Гистограмма LSB (резервный режим)")
            dlg.resize(920, 460)
            layout = QHBoxLayout(dlg)

            fig = Figure(figsize=(10, 4), dpi=100)
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            def draw_hist(ax, img: Image.Image, title: str):
                arr = np.array(img.convert("RGB"), dtype=np.uint8)
                lsb_r = (arr[:, :, 0] & 1).flatten()
                lsb_g = (arr[:, :, 1] & 1).flatten()
                lsb_b = (arr[:, :, 2] & 1).flatten()
                ax.hist([lsb_r, lsb_g, lsb_b], bins=[-0.5, 0.5, 1.5], label=["R LSB", "G LSB", "B LSB"])
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["0", "1"])
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend()

            draw_hist(ax1, self.image, "Оригинал")
            if self.encoded_image is not None:
                draw_hist(ax2, self.encoded_image, "После встраивания")
            else:
                ax2.set_title("После встраивания")
                ax2.text(0.5, 0.5, "Нет модифицированного\nизображения", ha="center", va="center")

            fig.tight_layout()
            canvas.draw()
            dlg.exec()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Гистограмма",
                "Не удалось открыть график ни через pyqtgraph, ни через matplotlib.\n"
                f"Подробности: {exc}",
            )
