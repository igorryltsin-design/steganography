from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui_qt.main_window import MainWindow
from ui_qt.theme import build_stylesheet


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Стего Студия")
    app.setStyleSheet(build_stylesheet("dark"))

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
