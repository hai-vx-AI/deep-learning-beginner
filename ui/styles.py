"""
ui/styles.py — QSS stylesheet tập trung.
Áp dụng 1 lần duy nhất khi khởi động app.

Nguyên tắc performance:
  - border-radius: 0px toàn bộ — không vẽ đường cong
  - Không gradient, không shadow, không animation
  - Font Consolas monospace — render nhanh hơn proportional
"""

# ── PALETTE ──────────────────────────────────────────────────────────────────

BG_DARK        = "#1a1a2e"
BG_PANEL       = "#16213e"
BG_CARD        = "#0f3460"
BG_DARKER      = "#0a0a1a"
ACCENT         = "#00d4ff"
SUCCESS        = "#00ff88"
WARNING        = "#ffaa00"
DANGER         = "#ff4757"
TEXT_PRIMARY   = "#e0e0e0"
TEXT_SECONDARY = "#8892b0"

# ── MAIN STYLESHEET ───────────────────────────────────────────────────────────

STYLESHEET = f"""
/* ── GLOBAL ── */
* {{
    border-radius: 0px;
    outline: none;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
    color: {TEXT_PRIMARY};
}}

QMainWindow, QWidget {{
    background-color: {BG_DARK};
}}

/* ── LABELS ── */
QLabel {{
    background: transparent;
    color: {TEXT_PRIMARY};
}}
QLabel#section_title {{
    color: {ACCENT};
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
}}
QLabel#sublabel {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
}}

/* ── FRAMES / SECTIONS ── */
QFrame#card {{
    background-color: {BG_CARD};
    border: 1px solid {BG_PANEL};
}}

/* ── LINE EDIT ── */
QLineEdit {{
    background-color: {BG_DARKER};
    border: 1px solid {BG_CARD};
    color: {TEXT_PRIMARY};
    padding: 4px 6px;
    font-size: 12px;
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}

/* ── PUSH BUTTONS ── */
QPushButton {{
    background-color: {BG_CARD};
    color: {ACCENT};
    border: 1px solid {ACCENT};
    padding: 4px 10px;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: {ACCENT};
    color: #000000;
}}
QPushButton:disabled {{
    opacity: 0.4;
    color: {TEXT_SECONDARY};
    border-color: {TEXT_SECONDARY};
    background-color: transparent;
}}

QPushButton#btn_start {{
    background-color: {ACCENT};
    color: #000000;
    font-size: 14px;
    font-weight: bold;
    border: none;
    padding: 8px;
}}
QPushButton#btn_start:hover {{
    background-color: #00a8cc;
}}
QPushButton#btn_start:disabled {{
    background-color: {TEXT_SECONDARY};
    color: {BG_DARKER};
}}

QPushButton#btn_stop {{
    background-color: transparent;
    color: {DANGER};
    border: 1px solid {DANGER};
    padding: 6px;
}}
QPushButton#btn_stop:hover {{
    background-color: {DANGER};
    color: #ffffff;
}}

QPushButton#btn_save {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    border: 1px solid {TEXT_SECONDARY};
    padding: 6px;
}}
QPushButton#btn_save:hover {{
    background-color: {TEXT_SECONDARY};
    color: #000000;
}}

QPushButton#btn_convert {{
    background-color: transparent;
    color: {WARNING};
    border: 1px dashed {WARNING};
    padding: 5px;
    font-size: 12px;
}}
QPushButton#btn_convert:hover {{
    background-color: {WARNING};
    color: #000000;
}}
QPushButton#btn_convert:disabled {{
    color: {TEXT_SECONDARY};
    border-color: {TEXT_SECONDARY};
}}

/* ── TRANSPORT BUTTONS (video controls) ── */
QPushButton#btn_rewind, QPushButton#btn_forward {{
    background-color: {BG_DARKER};
    color: {TEXT_SECONDARY};
    border: none;
    font-size: 14px;
    padding: 4px 8px;
}}
QPushButton#btn_play {{
    background-color: {ACCENT};
    color: #000000;
    border: none;
    font-size: 14px;
    font-weight: bold;
    padding: 4px 10px;
}}

/* ── RADIO BUTTONS ── */
QRadioButton {{
    color: {TEXT_PRIMARY};
    font-size: 13px;
    spacing: 8px;
}}
QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 2px solid {TEXT_SECONDARY};
    border-radius: 7px;
    background: transparent;
}}
QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

/* ── SLIDERS ── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {BG_DARKER};
    border: none;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    height: 4px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 10px;
    height: 10px;
    margin: -3px 0;
}}

/* ── SPINBOX ── */
QSpinBox, QDoubleSpinBox {{
    background-color: {BG_DARKER};
    border: 1px solid {BG_CARD};
    color: {TEXT_PRIMARY};
    padding: 3px 6px;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {BG_CARD};
    border: none;
    width: 16px;
}}

/* ── PROGRESS BAR ── */
QProgressBar {{
    background-color: {BG_DARKER};
    border: none;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {WARNING};
}}
QProgressBar#progress_possession_0::chunk {{
    background-color: {DANGER};
}}
QProgressBar#progress_possession_1::chunk {{
    background-color: #f1f1f1;
}}

/* ── SCROLL BAR ── */
QScrollBar:vertical {{
    background: {BG_DARKER};
    width: 6px;
}}
QScrollBar::handle:vertical {{
    background: {BG_CARD};
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ── TEXT EDIT (log) ── */
QTextEdit {{
    background-color: {BG_DARKER};
    border: 1px solid {BG_CARD};
    color: {TEXT_SECONDARY};
    font-size: 11px;
}}

/* ── HEADER ── */
QWidget#header {{
    background-color: {BG_DARKER};
    border-bottom: 1px solid {ACCENT};
}}

/* ── STATUS BAR ── */
QStatusBar {{
    background-color: {BG_DARKER};
    border-top: 1px solid {BG_CARD};
    color: {TEXT_SECONDARY};
    font-size: 11px;
}}

/* ── DIALOG ── */
QDialog {{
    background-color: {BG_DARK};
    border: 2px solid {ACCENT};
}}
"""