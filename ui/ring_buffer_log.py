"""
ui/ring_buffer_log.py — Log panel với Ring Buffer deque(maxlen=50).
Ẩn mặc định, toggle bằng phím L.
Không update UI khi panel đang ẩn — tiết kiệm CPU.
"""

from collections import deque
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
from PyQt6.QtCore    import Qt

from ui.styles import BG_DARKER, TEXT_SECONDARY, ACCENT


class RingBufferLog(QWidget):
    """Widget log — chỉ render text khi đang visible."""

    MAX_LINES = 50

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buffer = deque(maxlen=self.MAX_LINES)
        self.setVisible(False)   # ẩn mặc định
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        header = QLabel("LOG  (50 dòng gần nhất)  —  phím L để ẩn/hiện")
        header.setStyleSheet(f"color: {ACCENT}; font-size: 11px;")
        layout.addWidget(header)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet(f"""
            QTextEdit {{
                background: {BG_DARKER};
                color: {TEXT_SECONDARY};
                font-size: 11px;
                border: none;
            }}
        """)
        layout.addWidget(self.text_edit)

    def append(self, msg: str):
        """
        Thêm dòng log vào buffer.
        Chỉ update QTextEdit nếu widget đang visible — tiết kiệm CPU.
        """
        self._buffer.append(msg)
        if self.isVisible():
            self._refresh()

    def toggle(self):
        """Toggle ẩn/hiện. Khi hiện lại — refresh toàn bộ buffer."""
        if self.isVisible():
            self.setVisible(False)
        else:
            self._refresh()
            self.setVisible(True)

    def _refresh(self):
        self.text_edit.setPlainText("\n".join(self._buffer))
        # Scroll xuống cuối
        sb = self.text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())