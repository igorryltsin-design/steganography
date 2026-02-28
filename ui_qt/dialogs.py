from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.analysis import compute_change_heatmap
from core.report import render_presentation_summary, render_report_text
from ui_qt.graphics_view import ImageGraphicsView
from ui_qt.image_utils import pil_to_pixmap


class CompareDialog(QDialog):
    def __init__(self, original: Image.Image, modified: Image.Image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("До/После + Теплокарта")
        self.resize(980, 700)
        self.original = original.convert("RGB")
        self.modified = modified.convert("RGB")
        self.mode = "split"
        self.blink = False
        self._blink_state = False

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Визуальный анализ изменений"))

        self.mode_tabs = QTabWidget()
        for title, mode in [
            ("Разделение", "split"),
            ("Смешивание", "blend"),
            ("Теплокарта", "heatmap"),
            ("Усиление ×20", "amplify20"),
        ]:
            tab = QWidget()
            self.mode_tabs.addTab(tab, title)
            tab.setProperty("mode", mode)
        self.mode_tabs.currentChanged.connect(self._on_mode_changed)
        top.addWidget(self.mode_tabs)
        root.addLayout(top)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Положение:"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(lambda _: self.render())
        controls.addWidget(self.slider, 1)
        self.blink_check = QCheckBox("Мигание слоёв")
        self.blink_check.toggled.connect(self._set_blink)
        controls.addWidget(self.blink_check)
        root.addLayout(controls)

        self.view = ImageGraphicsView()
        root.addWidget(self.view, 1)
        self.hud = QLabel("")
        root.addWidget(self.hud)

        self.timer = QTimer(self)
        self.timer.setInterval(450)
        self.timer.timeout.connect(self._tick_blink)

        self._animate_open()
        self.render()

    def _animate_open(self):
        self.setWindowOpacity(0.0)
        anim = QPropertyAnimation(self, b"windowOpacity")
        anim.setDuration(180)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim = anim
        anim.start()

    def _on_mode_changed(self, idx: int):
        w = self.mode_tabs.widget(idx)
        self.mode = w.property("mode")
        self.render()

    def _set_blink(self, enabled: bool):
        self.blink = enabled
        self._blink_state = False
        if enabled:
            self.timer.start()
        else:
            self.timer.stop()
            self.render()

    def _tick_blink(self):
        self._blink_state = not self._blink_state
        self.render()

    def render(self):
        ratio = max(0.0, min(1.0, self.slider.value() / 100.0))
        orig = self.original
        mod = self.modified
        delta = None
        changed_pct = None

        if self.blink and self.mode in {"split", "blend"}:
            composed = orig if self._blink_state else mod
            mode_label = "МИГАНИЕ: ОРИГИНАЛ" if self._blink_state else "МИГАНИЕ: ИЗМЕНЁННОЕ"
        elif self.mode == "blend":
            composed = Image.blend(orig, mod, ratio)
            mode_label = f"СМЕШИВАНИЕ {ratio * 100:.0f}%"
        elif self.mode == "heatmap":
            delta = np.abs(np.array(mod, dtype=np.int16) - np.array(orig, dtype=np.int16))
            changed_pct = float(np.mean(np.any(delta > 0, axis=2)) * 100.0)
            composed = compute_change_heatmap(orig, mod)
            mode_label = f"ТЕПЛОКАРТА · изменено {changed_pct:.2f}%"
        elif self.mode == "amplify20":
            delta = np.abs(np.array(mod, dtype=np.int16) - np.array(orig, dtype=np.int16))
            changed_pct = float(np.mean(np.any(delta > 0, axis=2)) * 100.0)

            # Усиление отличий + автонормализация, чтобы режим оставался видимым
            # даже при очень малых LSB-сдвигах (обычно 1..3).
            amplified = np.clip(delta * 20, 0, 255).astype(np.uint8)
            max_amp = int(amplified.max())
            if max_amp > 0:
                scaled = (amplified.astype(np.float32) / float(max_amp) * 255.0).astype(np.uint8)
            else:
                scaled = amplified

            mag = scaled.max(axis=2).astype(np.float32) / 255.0
            # Небольшой базовый синий фон убирает "полностью чёрный" вид
            # и делает режим более читабельным для глаз.
            r = np.clip((mag ** 0.7) * 255.0, 0, 255).astype(np.uint8)
            g = np.clip((mag ** 1.0) * 210.0 + 6.0, 0, 255).astype(np.uint8)
            b = np.clip((mag ** 1.6) * 70.0 + 26.0, 0, 255).astype(np.uint8)
            composed = Image.fromarray(np.stack([r, g, b], axis=2), mode="RGB")
            mode_label = f"УСИЛЕНИЕ ×20 · изменено {changed_pct:.2f}%"
        else:
            split_x = int(orig.width * ratio)
            composed = mod.copy()
            if split_x > 0:
                composed.paste(orig.crop((0, 0, split_x, orig.height)), (0, 0))
            # Явный разделитель для режима split, чтобы граница была заметна на любом фоне.
            if 0 < split_x < composed.width:
                draw = ImageDraw.Draw(composed)
                line_w = max(2, composed.width // 420)
                x = max(1, min(composed.width - 2, split_x))
                draw.line((x - 1, 0, x - 1, composed.height), fill=(8, 22, 35), width=line_w + 2)
                draw.line((x, 0, x, composed.height), fill=(255, 209, 102), width=line_w)
            mode_label = f"РАЗДЕЛЕНИЕ {ratio * 100:.0f}%"

        self.view.set_pixmap(pil_to_pixmap(composed))
        self.hud.setText(mode_label)


class AttackLabDialog(QDialog):
    def __init__(self, rows: Sequence[Mapping[str, Any]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Симулятор атак")
        self.resize(860, 460)

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Результаты устойчивости к атакам"))

        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Атака", "Извлечение", "Детали"])
        table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(table, 1)

        passed = 0
        for row in rows:
            ok = bool(row.get("success"))
            if ok:
                passed += 1
            detail = "без ошибок" if ok else (row.get("error") or row.get("preview_text") or "сообщение не совпало")
            ridx = table.rowCount()
            table.insertRow(ridx)
            table.setItem(ridx, 0, QTableWidgetItem(str(row.get("name", "-"))))
            table.setItem(ridx, 1, QTableWidgetItem("УСПЕХ" if ok else "СБОЙ"))
            table.setItem(ridx, 2, QTableWidgetItem(str(detail)))

        root.addWidget(QLabel(f"Успешно: {passed}/{len(rows)}"))


class BenchmarkDialog(QDialog):
    def __init__(self, rows: Sequence[Mapping[str, Any]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Сравнение режимов")
        self.resize(900, 500)

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Автопрогон: метод × биты"))

        table = QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(["Метод", "Биты", "Влезает", "Извлечение", "PSNR", "SSIM"])
        table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(table, 1)

        best = None
        for row in rows:
            method_label = "Последовательный" if row.get("method") == "sequential" else "Чередование"
            ok = bool(row.get("decode_ok"))
            ridx = table.rowCount()
            table.insertRow(ridx)
            table.setItem(ridx, 0, QTableWidgetItem(method_label))
            table.setItem(ridx, 1, QTableWidgetItem(str(row.get("bits", "-"))))
            table.setItem(ridx, 2, QTableWidgetItem("ДА" if row.get("fit") else "НЕТ"))
            table.setItem(ridx, 3, QTableWidgetItem("УСПЕХ" if ok else "СБОЙ"))
            table.setItem(ridx, 4, QTableWidgetItem("-" if row.get("psnr_db") is None else f"{row['psnr_db']:.2f}"))
            table.setItem(ridx, 5, QTableWidgetItem("-" if row.get("ssim") is None else f"{row['ssim']:.4f}"))
            if ok and row.get("ssim") is not None and (best is None or row["ssim"] > best["ssim"]):
                best = dict(row)

        if best:
            method = "Последовательный" if best.get("method") == "sequential" else "Чередование"
            text = f"Лучший режим: {method} / {best['bits']} бит (SSIM={best['ssim']:.4f}, PSNR={best['psnr_db']:.2f})"
        else:
            text = "Нет режима с успешным извлечением."
        root.addWidget(QLabel(text))


class ReportPreviewDialog(QDialog):
    def __init__(self, report: Dict[str, Any], on_save_json, on_save_txt, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Предпросмотр отчёта")
        self.resize(940, 660)
        self.report = report
        self._on_save_json = on_save_json
        self._on_save_txt = on_save_txt

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Предпросмотр отчёта перед экспортом"))

        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        self.txt_tab = QPlainTextEdit()
        self.txt_tab.setReadOnly(True)
        self.txt_tab.setPlainText(render_report_text(report))
        tabs.addTab(self.txt_tab, "TXT")

        self.json_tab = QPlainTextEdit()
        self.json_tab.setReadOnly(True)
        self.json_tab.setPlainText(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        tabs.addTab(self.json_tab, "JSON")

        self.summary_tab = QPlainTextEdit()
        self.summary_tab.setReadOnly(True)
        self.summary_tab.setPlainText(render_presentation_summary(report))
        tabs.addTab(self.summary_tab, "Сводка")

        bottom = QHBoxLayout()
        btn_json = QPushButton("Сохранить JSON...")
        btn_json.clicked.connect(self._on_save_json)
        bottom.addWidget(btn_json)

        btn_txt = QPushButton("Сохранить TXT...")
        btn_txt.clicked.connect(self._on_save_txt)
        bottom.addWidget(btn_txt)

        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.accept)
        bottom.addWidget(btn_close)
        root.addLayout(bottom)


class DemoTimelineDialog(QDialog):
    def __init__(self, steps: Sequence[tuple[str, Callable[[], str]]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Демо за 2 минуты")
        self.resize(720, 460)
        self.steps = list(steps)
        self.step_index = 0
        self.completed = False

        root = QVBoxLayout(self)
        self.title = QLabel("Автоматический демонстрационный сценарий")
        root.addWidget(self.title)

        self.progress = QProgressBar()
        self.progress.setRange(0, max(1, len(self.steps)))
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.current_step = QLabel("Подготовка...")
        root.addWidget(self.current_step)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, 1)

        bottom = QHBoxLayout()
        self.btn_start = QPushButton("Запустить")
        self.btn_start.clicked.connect(self.start)
        bottom.addWidget(self.btn_start)
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.clicked.connect(self.reject)
        bottom.addWidget(self.btn_close)
        root.addLayout(bottom)

    def start(self):
        self.btn_start.setEnabled(False)
        self.btn_close.setEnabled(False)
        self._append("Старт демо-сценария.")
        QTimer.singleShot(120, self._run_next)

    def _run_next(self):
        if self.step_index >= len(self.steps):
            self.completed = True
            self.current_step.setText("Сценарий завершён.")
            self._append("✓ Все шаги выполнены.")
            self.btn_close.setEnabled(True)
            self.btn_close.setText("Далее")
            return

        title, fn = self.steps[self.step_index]
        self.current_step.setText(f"Шаг {self.step_index + 1}/{len(self.steps)}: {title}")
        try:
            detail = fn()
            self._append(f"✓ {title}: {detail}")
            self.step_index += 1
            self.progress.setValue(self.step_index)
            QTimer.singleShot(140, self._run_next)
        except Exception as exc:
            self._append(f"✗ {title}: {exc}")
            self.current_step.setText("Сценарий остановлен из-за ошибки.")
            self.btn_close.setEnabled(True)
            self.btn_close.setText("Закрыть")

    def _append(self, line: str):
        text = self.log.toPlainText().strip()
        text = (text + "\n" + line).strip() if text else line
        self.log.setPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())


class DemoResultDialog(QDialog):
    def __init__(
        self,
        risk_level: str,
        risk_reason: str,
        robustness_score: float | None,
        recommendation: str,
        on_open_compare: Callable[[], None],
        on_open_attacks: Callable[[], None],
        on_open_benchmark: Callable[[], None],
        on_export_pack: Callable[[], None],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Итог демо")
        self.resize(780, 500)

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Итог демо"))

        form = QFormLayout()
        form.addRow("Риск:", QLabel(f"{risk_level} — {risk_reason}"))
        form.addRow(
            "Устойчивость:",
            QLabel("—" if robustness_score is None else f"{robustness_score:.1f}%"),
        )
        form.addRow("Рекомендация:", QLabel(recommendation))
        root.addLayout(form)

        root.addWidget(QLabel("Быстрые действия"))
        actions = QHBoxLayout()
        btn_compare = QPushButton("Открыть До/После")
        btn_compare.clicked.connect(on_open_compare)
        actions.addWidget(btn_compare)
        btn_attacks = QPushButton("Открыть Attack Lab")
        btn_attacks.clicked.connect(on_open_attacks)
        actions.addWidget(btn_attacks)
        btn_bench = QPushButton("Открыть бенчмарк")
        btn_bench.clicked.connect(on_open_benchmark)
        actions.addWidget(btn_bench)
        root.addLayout(actions)

        row2 = QHBoxLayout()
        btn_pack = QPushButton("Экспорт архива данных")
        btn_pack.clicked.connect(on_export_pack)
        row2.addWidget(btn_pack)
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.accept)
        row2.addWidget(btn_close)
        root.addLayout(row2)


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справка")
        self.resize(980, 700)

        root = QVBoxLayout(self)
        title = QLabel("Подробная справка по программе")
        title.setStyleSheet("font-weight: 700;")
        root.addWidget(title)
        root.addWidget(
            QLabel(
                "Здесь собраны инструкции по работе, пояснения метрик, "
                "описание режимов и разбор частых ошибок."
            )
        )

        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        tab_data: List[tuple[str, str]] = [
            (
                "Быстрый старт",
                "\n".join(
                    [
                        "Сценарий 1: скрыть и извлечь сообщение",
                        "",
                        "1. Нажмите «Открыть изображение» и выберите контейнер.",
                        "2. В поле «Сообщение» введите текст для скрытия.",
                        "3. Опционально заполните «Пароль»: он применяется как XOR-маска.",
                        "4. Выберите «Бит на канал» (1, 2 или 3).",
                        "5. Выберите «Метод сокрытия»:",
                        "   - Последовательный (R->G->B): выше качество изображения.",
                        "   - Чередование каналов: немного равномернее распределение изменений.",
                        "6. Нажмите «Спрятать».",
                        "7. После встраивания нажмите «Извлечь» для проверки.",
                        "8. Сохраните результат: «Сохранить как...», «Экспорт отчёта», «Экспорт архива данных».",
                        "",
                        "Сценарий 2: только проверить чужое изображение",
                        "",
                        "1. Откройте файл.",
                        "2. При необходимости введите пароль.",
                        "3. Нажмите «Извлечь».",
                        "4. Оцените метрики и риск в правой панели.",
                    ]
                ),
            ),
            (
                "Панель управления",
                "\n".join(
                    [
                        "Что означают элементы справа",
                        "",
                        "Риск:",
                        "LOW / MEDIUM / HIGH — оценка заметности скрытия по LSB-статистике и загрузке контейнера.",
                        "",
                        "Загрузка контейнера:",
                        "Показывает, какую долю доступной ёмкости занимает сообщение.",
                        "Чем выше процент, тем выше шанс статистического обнаружения.",
                        "",
                        "Устойчивость к атакам:",
                        "Итог по симуляции атак (без атаки, JPEG, resize, шум, blur).",
                        "100% означает, что сообщение восстановилось во всех сценариях.",
                        "",
                        "Кнопки:",
                        "Открыть изображение — загрузка контейнера.",
                        "Сохранить как... — сохранить модифицированное изображение.",
                        "Экспорт отчёта — текст/JSON отчёт с метриками.",
                        "Экспорт архива данных — zip с артефактами анализа.",
                        "Показать гистограмму LSB — распределение младших битов.",
                        "До/После + Теплокарта — визуальный анализ отличий.",
                        "Симулятор атак — проверка устойчивости к искажениям.",
                        "Сравнить режимы — автопрогон метод × биты.",
                        "Демо-режим — быстрый сценарий всех основных инструментов.",
                        "",
                        "Живая аналитика в главном окне:",
                        "Верхняя HUD-плашка показывает процент изменённых пикселей, среднюю и максимальную дельту.",
                        "Миникарта hotspot показывает, в каких областях изменения наиболее плотные.",
                        "Пиксельный инспектор показывает биты канала, LSB до/после и локальную лупу.",
                        "",
                        "Режимы экрана:",
                        "Базовый — показывает только основные действия без перегрузки интерфейса.",
                        "Эксперт — открывает hotspot, инспектор, расширенный анализ и служебные инструменты.",
                        "",
                        "Кнопки в заголовке:",
                        "❔ — открывает это окно справки.",
                        "☀/🌙 — переключает тему (день/ночь) в реальном времени.",
                    ]
                ),
            ),
            (
                "Режимы сравнения",
                "\n".join(
                    [
                        "Окно «До/После + Теплокарта»",
                        "",
                        "Разделение:",
                        "Показывает оригинал и модифицированную версию с вертикальным разделителем.",
                        "Ползунок «Положение» меняет границу сравнения.",
                        "",
                        "Смешивание:",
                        "Накладывает изображения друг на друга.",
                        "Удобно для оценки глобальных изменений без резких границ.",
                        "",
                        "Теплокарта:",
                        "Цветом показывает, где изменения наиболее выражены.",
                        "Синий/холодный — малые изменения, тёплые оттенки — более сильные.",
                        "",
                        "Усиление ×20:",
                        "Специальный режим для наглядного усиления малых различий LSB.",
                        "Используется для визуальной диагностики, а не как итоговое изображение.",
                        "",
                        "Мигание слоёв:",
                        "Быстро переключает оригинал/модифицированное для визуального поиска отличий.",
                        "",
                        "Порог чувствительности:",
                        "Позволяет скрыть слишком малые изменения и сосредоточиться на более выраженных.",
                        "",
                        "Режим «Точно»:",
                        "Строит карту по полноразмерному изображению.",
                        "Если выключен, используется быстрый preview для плавной работы на больших файлах.",
                    ]
                ),
            ),
            (
                "Метрики и риск",
                "\n".join(
                    [
                        "PSNR (дБ):",
                        "Чем выше, тем меньше визуальная деградация после скрытия.",
                        "Обычно > 40 дБ считается очень хорошим качеством.",
                        "",
                        "MSE:",
                        "Средняя квадратичная ошибка между оригиналом и стего.",
                        "Чем ближе к нулю, тем лучше.",
                        "",
                        "SSIM:",
                        "Сходство структуры изображения (0..1).",
                        "Чем ближе к 1.0, тем изображения похожее по структуре.",
                        "",
                        "χ² (хи-квадрат) по LSB:",
                        "Проверка статистической аномалии младших битов.",
                        "Сильное отклонение может повышать вероятность обнаружения.",
                        "",
                        "Как формируется Risk:",
                        "Учитывается загрузка контейнера, статистика LSB и качество восстановления.",
                        "LOW: низкая заметность, запас ёмкости есть.",
                        "MEDIUM: заметные отклонения, приемлемо для практики.",
                        "HIGH: высокий шанс обнаружения, стоит снизить нагрузку или сменить режим.",
                        "",
                        "Как читать hotspot:",
                        "Чем ярче ячейка миникарты, тем выше средняя интенсивность изменений в этой зоне.",
                        "Клик по ячейке переносит фокус инспектора в соответствующую область изображения.",
                    ]
                ),
            ),
            (
                "Отчёты и ошибки",
                "\n".join(
                    [
                        "Экспорт отчёта",
                        "",
                        "В окне предпросмотра доступны вкладки:",
                        "Текст — удобный читаемый отчёт для человека.",
                        "JSON — структурированные данные для автоматической обработки.",
                        "Сводка — короткий итог с ключевыми выводами.",
                        "",
                        "Экспорт архива данных (zip)",
                        "",
                        "В архив входят:",
                        "- report.json",
                        "- report.txt",
                        "- before.png",
                        "- after.png",
                        "- heatmap.png",
                        "- hotspot.png",
                        "- inspector.png",
                        "- attacks.csv",
                        "",
                        "Частые ошибки и решения",
                        "",
                        "1) «Сообщение слишком большое»",
                        "Уменьшите текст, выберите большее изображение или увеличьте «бит на канал».",
                        "",
                        "2) «Не удалось извлечь сообщение»",
                        "Проверьте пароль и убедитесь, что изображение не проходило JPEG-сжатие/редактирование.",
                        "",
                        "3) Низкая устойчивость к атакам",
                        "Снизьте загрузку контейнера, используйте 1 бит/канал, избегайте JPEG после встраивания.",
                        "",
                        "4) pyqtgraph не установлен",
                        "Графики будут открываться через резервный режим. "
                        "Для интерактива установите зависимость: pip install pyqtgraph",
                    ]
                ),
            ),
        ]

        for tab_title, tab_text in tab_data:
            text_box = QPlainTextEdit()
            text_box.setReadOnly(True)
            text_box.setPlainText(tab_text)
            tabs.addTab(text_box, tab_title)

        tabs.insertTab(1, self._build_lsb_tab(), "LSB подробно")

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.accept)
        close_row.addWidget(btn_close)
        root.addLayout(close_row)

    def _build_lsb_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        intro = QLabel(
            "LSB (Least Significant Bit) — младший бит каждого цветового канала (R, G, B). "
            "Это самый «слабый» по влиянию бит, поэтому его замена обычно не видна глазом."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        diagram = QLabel()
        diagram.setAlignment(Qt.AlignmentFlag.AlignCenter)
        diagram.setPixmap(self._build_lsb_diagram_pixmap())
        layout.addWidget(diagram)

        details = QPlainTextEdit()
        details.setReadOnly(True)
        details.setPlainText(
            "\n".join(
                [
                    "Как работает LSB-встраивание (простыми словами)",
                    "",
                    "1) Каждый пиксель RGB содержит 3 байта: R, G, B (по 8 бит в каждом).",
                    "2) Мы берём бит сообщения (0 или 1) и подставляем его в последний бит канала.",
                    "3) Значение канала меняется максимум на 1 при режиме 1 бит/канал.",
                    "4) Для глаза это почти всегда незаметно, но сообщение можно восстановить по тем же правилам.",
                    "",
                    "Пример для одного канала:",
                    "R было: 10110110 (182)",
                    "нужно записать бит «1» -> R станет: 10110111 (183)",
                    "Изменение: +1 (визуально почти неразличимо).",
                    "",
                    "Что дают режимы 1/2/3 бита на канал:",
                    "- 1 бит: лучшее качество изображения, ниже риск обнаружения.",
                    "- 2 бита: больше ёмкость, выше вероятность статистических следов.",
                    "- 3 бита: максимальная ёмкость, но заметность и риск выше.",
                    "",
                    "Оценка вместимости (приблизительно):",
                    "capacity_bytes ≈ width * height * 3 * bits_per_channel / 8",
                    "",
                    "Почему после встраивания нет видимой разницы:",
                    "- Меняется только младший разряд (минимальный вклад в яркость канала).",
                    "- Изменения распределены по большому числу пикселей.",
                    "",
                    "Важно:",
                    "- JPEG (сжатие с потерями) может разрушить LSB-данные.",
                    "- Для надёжной передачи лучше сохранять стего-изображение в PNG/BMP.",
                    "- Пароль в программе используется для XOR-маски текста перед встраиванием.",
                ]
            )
        )
        layout.addWidget(details, 1)
        return panel

    def _build_lsb_diagram_pixmap(self) -> QPixmap:
        width, height = 860, 240
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor("#07263f"))

        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        frame_pen = QPen(QColor("#4bb9ff"))
        frame_pen.setWidth(2)
        p.setPen(frame_pen)

        left_x, top_y, box_w, box_h = 24, 36, 360, 164
        right_x = width - box_w - 24
        p.drawRoundedRect(left_x, top_y, box_w, box_h, 10, 10)
        p.drawRoundedRect(right_x, top_y, box_w, box_h, 10, 10)

        p.setPen(QColor("#d5ecff"))
        p.drawText(left_x + 14, top_y + 24, "До встраивания")
        p.drawText(right_x + 14, top_y + 24, "После встраивания (бит сообщения = 1,0,1)")

        p.setPen(QColor("#bde2ff"))
        p.drawText(left_x + 14, top_y + 56, "R: 10110110")
        p.drawText(left_x + 14, top_y + 82, "G: 01101001")
        p.drawText(left_x + 14, top_y + 108, "B: 11001010")
        p.drawText(left_x + 14, top_y + 142, "LSB:      0        1        0")

        p.setPen(QColor("#ffd27a"))
        p.drawText(right_x + 14, top_y + 56, "R: 10110111   (+1)")
        p.drawText(right_x + 14, top_y + 82, "G: 01101000   (-1)")
        p.drawText(right_x + 14, top_y + 108, "B: 11001011   (+1)")
        p.drawText(right_x + 14, top_y + 142, "LSB:      1        0        1")

        p.setPen(QPen(QColor("#38f0d0"), 3))
        arrow_y = top_y + box_h // 2
        p.drawLine(left_x + box_w + 16, arrow_y, right_x - 16, arrow_y)
        p.drawLine(right_x - 22, arrow_y - 6, right_x - 16, arrow_y)
        p.drawLine(right_x - 22, arrow_y + 6, right_x - 16, arrow_y)

        p.setPen(QColor("#8fd8ff"))
        p.drawText(290, height - 12, "При 1 бит/канал изменение каждого канала обычно только ±1")

        p.end()
        return pixmap
