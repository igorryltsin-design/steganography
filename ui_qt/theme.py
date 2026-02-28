from __future__ import annotations

DARK_TOKENS = {
    "bg": "#0b1f33",
    "panel": "#173857",
    "panel_soft": "#244e73",
    "canvas": "#0f2e4d",
    "line": "#7eadd6",
    "text": "#eef6ff",
    "muted": "#c9ddf1",
    "accent": "#45e1bc",
    "warning": "#ffd166",
    "danger": "#ff6176",
    "success": "#58d6b5",
    "btn": "#2f6292",
    "btn_hover": "#3a76af",
    "btn_press": "#285985",
    "btn_disabled": "#5d7387",
    "btn_text": "#f6fbff",
    "btn_text_disabled": "#dce6ee",
}


LIGHT_TOKENS = {
    "bg": "#eef4fb",
    "panel": "#f7fbff",
    "panel_soft": "#edf5ff",
    "canvas": "#ffffff",
    "line": "#8aa7c5",
    "text": "#10263b",
    "muted": "#3f607f",
    "accent": "#0cbfa1",
    "warning": "#d18a00",
    "danger": "#cf3b53",
    "success": "#2e9c76",
    "btn": "#d8e8f8",
    "btn_hover": "#c7ddf3",
    "btn_press": "#b6d2ee",
    "btn_disabled": "#d5dde6",
    "btn_text": "#16334d",
    "btn_text_disabled": "#6e849a",
}


# Обратная совместимость: старый код может импортировать TOKENS.
TOKENS = DARK_TOKENS


def get_tokens(theme_mode: str) -> dict[str, str]:
    return LIGHT_TOKENS if theme_mode == "light" else DARK_TOKENS


def build_stylesheet(theme_mode: str = "dark") -> str:
    t = get_tokens(theme_mode)
    return f"""
    QWidget {{
      background: {t['bg']};
      color: {t['text']};
      font-family: 'Segoe UI', 'Noto Sans', sans-serif;
      font-size: 13px;
    }}
    QFrame#Card {{
      background: {t['panel']};
      border: 1px solid {t['line']};
      border-radius: 10px;
    }}
    QFrame#CardSoft {{
      background: {t['panel_soft']};
      border: 1px solid {t['line']};
      border-radius: 10px;
    }}
    QLabel#Title {{
      font-size: 36px;
      font-weight: 700;
      color: {t['text']};
    }}
    QLabel#SectionTitle {{
      font-size: 18px;
      font-weight: 700;
    }}
    QLabel#Hint {{
      color: {t['muted']};
      font-size: 12px;
    }}
    QPushButton {{
      background: {t['btn']};
      color: {t['btn_text']};
      border: 1px solid {t['line']};
      border-radius: 8px;
      padding: 9px 12px;
      font-weight: 600;
    }}
    QPushButton:hover {{ background: {t['btn_hover']}; }}
    QPushButton:pressed {{ background: {t['btn_press']}; }}
    QPushButton:checked {{ background: {t['btn_hover']}; border-color: {t['accent']}; }}
    QPushButton:disabled {{
      background: {t['btn_disabled']};
      color: {t['btn_text_disabled']};
      border-color: {t['btn_disabled']};
    }}
    QToolButton {{
      background: {t['btn']};
      color: {t['btn_text']};
      border: 1px solid {t['line']};
      border-radius: 8px;
      padding: 6px 10px;
      font-weight: 700;
    }}
    QToolButton:hover {{ background: {t['btn_hover']}; }}
    QToolButton:pressed {{ background: {t['btn_press']}; }}
    QLineEdit, QTextEdit, QPlainTextEdit {{
      background: {t['canvas']};
      color: {t['text']};
      border: 1px solid {t['line']};
      border-radius: 8px;
      padding: 6px;
    }}
    QComboBox {{
      background: {t['canvas']};
      color: {t['text']};
      border: 1px solid {t['line']};
      border-radius: 8px;
      padding: 5px 8px;
      min-width: 72px;
    }}
    QComboBox::drop-down {{
      border: 0px;
      width: 20px;
    }}
    QRadioButton {{
      color: {t['text']};
      spacing: 8px;
    }}
    QCheckBox {{
      color: {t['text']};
      spacing: 8px;
    }}
    QSlider::groove:horizontal {{
      height: 8px;
      background: {t['canvas']};
      border: 1px solid {t['line']};
      border-radius: 4px;
    }}
    QSlider::handle:horizontal {{
      background: {t['warning']};
      border: 1px solid {t['line']};
      width: 16px;
      margin: -5px 0;
      border-radius: 8px;
    }}
    QProgressBar {{
      background: {t['canvas']};
      border: 1px solid {t['line']};
      border-radius: 5px;
      text-align: center;
      color: {t['text']};
      font-weight: 600;
    }}
    QProgressBar::chunk {{
      background: {t['accent']};
      border-radius: 4px;
    }}
    QTabWidget::pane {{
      border: 1px solid {t['line']};
      border-radius: 8px;
      background: {t['panel']};
      top: -1px;
    }}
    QTabBar::tab {{
      background: {t['btn']};
      border: 1px solid {t['line']};
      border-top-left-radius: 6px;
      border-top-right-radius: 6px;
      padding: 7px 12px;
      margin-right: 4px;
      color: {t['text']};
      font-weight: 600;
    }}
    QTabBar::tab:selected {{
      background: {t['btn_hover']};
    }}
    QTableWidget {{
      background: {t['canvas']};
      border: 1px solid {t['line']};
      gridline-color: {t['line']};
      selection-background-color: {t['btn_hover']};
      selection-color: {t['text']};
    }}
    QHeaderView::section {{
      background: {t['panel_soft']};
      color: {t['text']};
      padding: 6px;
      border: 0px;
      border-bottom: 1px solid {t['line']};
      font-weight: 700;
    }}
    """
