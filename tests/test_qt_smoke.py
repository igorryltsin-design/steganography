import os
import unittest

import numpy as np
from PIL import Image


class QtSmokeTests(unittest.TestCase):
    def test_main_window_constructs(self):
        try:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PySide6.QtWidgets import QApplication
            from PySide6.QtTest import QTest
            from ui_qt.main_window import MainWindow
        except Exception:
            self.skipTest("PySide6 недоступен в окружении тестов")
            return

        app = QApplication.instance() or QApplication([])
        win = MainWindow()
        self.assertEqual(win.windowTitle(), "Стего Студия — LSB")
        self.assertTrue(hasattr(win, "btn_theme"))
        self.assertTrue(hasattr(win, "btn_help"))
        old_mode = win.theme_mode
        win.toggle_theme()
        self.assertNotEqual(old_mode, win.theme_mode)
        self.assertTrue(hasattr(win, "pixel_inspector"))
        self.assertTrue(hasattr(win, "hotspot_map"))
        self.assertEqual(win.ui_mode, "basic")
        win._set_ui_mode("expert")
        self.assertEqual(win.ui_mode, "expert")
        win.close()

    def test_auto_mode_selects_valid_config(self):
        try:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PySide6.QtWidgets import QApplication
            from PySide6.QtTest import QTest
            from ui_qt.main_window import MainWindow
        except Exception:
            self.skipTest("PySide6 недоступен в окружении тестов")
            return

        app = QApplication.instance() or QApplication([])
        win = MainWindow()
        win.image = Image.new("RGB", (128, 128), (120, 130, 140))
        win.message_text.setPlainText("Автотест выбора режима")
        win.auto_select_mode()
        self.assertIn(win._bits_value(), {1, 2, 3})
        self.assertIn(win._method_value(), {"sequential", "interleaved"})
        win.close()

    def test_visual_analytics_updates_probe(self):
        try:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PySide6.QtWidgets import QApplication
            from PySide6.QtTest import QTest
            from ui_qt.main_window import MainWindow
        except Exception:
            self.skipTest("PySide6 недоступен в окружении тестов")
            return

        app = QApplication.instance() or QApplication([])
        win = MainWindow()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[..., 0] = 110
        arr[..., 1] = 130
        arr[..., 2] = 150
        win.image = Image.fromarray(arr, mode="RGB")
        win.image_signature = win._image_signature(win.image)
        win.message_text.setPlainText("Проверка визуальной аналитики")
        win.encode_message()
        win._refresh_visual_analysis()
        for _ in range(30):
            QTest.qWait(40)
            app.processEvents()
            if win.analysis_hotspot is not None:
                break

        win._update_probe(5, 5, set_fixed=True)
        self.assertEqual(win.pixel_inspector.coord_label.text(), "x=5, y=5")
        self.assertIsNotNone(win.analysis_hotspot)
        self.assertIn(win.analysis_mode, {"split", "blend", "heatmap", "amplify20"})
        win.close()


if __name__ == "__main__":
    unittest.main()
