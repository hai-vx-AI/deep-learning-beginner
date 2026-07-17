"""
ui/video_display.py — Hiển thị frame output từ AI thread.

Ưu tiên tốc độ:
  - Timer preview 20 FPS để giảm tải UI.
  - Queue non-blocking.
  - Scale nhanh và giữ đúng tỉ lệ hình.
"""

from queue import Queue, Empty
import numpy as np

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider
from PyQt6.QtCore    import Qt, QTimer
from PyQt6.QtGui     import QImage, QPixmap

class VideoDisplay(QWidget):
    VIDEO_W = 896
    VIDEO_H = 630
    DISPLAY_INTERVAL_MS = 50  # 20 FPS preview; AI thread vẫn chạy tối đa tốc độ.

    def __init__(self, frame_queue: Queue, parent=None):
        super().__init__(parent)
        self.frame_queue = frame_queue
        self.setFixedSize(self.VIDEO_W, 670)

        self._build_ui()

        self._display_timer = QTimer(self)
        self._display_timer.setInterval(self.DISPLAY_INTERVAL_MS)
        self._display_timer.timeout.connect(self._pull_frame)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.video_label = QLabel()
        self.video_label.setFixedSize(self.VIDEO_W, self.VIDEO_H)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000;")
        self.video_label.setText("Chưa có video")

        controls = self._build_controls()
        layout.addWidget(self.video_label)
        layout.addWidget(controls)

    def _build_controls(self) -> QWidget:
        widget = QWidget()
        widget.setFixedHeight(40)
        widget.setStyleSheet("background-color: #0a0a1a;")

        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        self.btn_rewind  = QPushButton("◀◀")
        self.btn_play    = QPushButton("▶")
        self.btn_forward = QPushButton("▶▶")

        self.btn_rewind.setObjectName("btn_rewind")
        self.btn_play.setObjectName("btn_play")
        self.btn_forward.setObjectName("btn_forward")

        self.btn_rewind.setFixedSize(36, 32)
        self.btn_play.setFixedSize(44, 32)
        self.btn_forward.setFixedSize(36, 32)

        # Các nút này hiện chỉ là preview/status, không điều khiển stream realtime.
        self.btn_rewind.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_forward.setEnabled(False)

        self.seek_bar = QSlider(Qt.Orientation.Horizontal)
        self.seek_bar.setRange(0, 100)
        self.seek_bar.setValue(0)
        self.seek_bar.setEnabled(False)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedWidth(100)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.time_label.setStyleSheet("color: #8892b0; font-size: 11px;")

        layout.addWidget(self.btn_rewind)
        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_forward)
        layout.addWidget(self.seek_bar, 1)
        layout.addWidget(self.time_label)

        return widget

    def start_display(self):
        self._display_timer.start()

    def stop_display(self):
        self._display_timer.stop()

    def show_message(self, text: str):
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText(text)

    def _pull_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
            self._show_frame(frame)
        except Empty:
            pass

    def _show_frame(self, frame: np.ndarray):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(
            frame.data, w, h, bytes_per_line,
            QImage.Format.Format_BGR888
        )
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.video_label.setPixmap(pixmap)
        self.video_label.setText("")

    def update_seek(self, current_frame: int, total_frames: int):
        if total_frames > 0:
            pct = int(current_frame / total_frames * 100)
            self.seek_bar.setValue(pct)

    def update_time_label(self, current_sec: float, total_sec: float):
        def fmt(s):
            m = int(s) // 60
            s = int(s) % 60
            return f"{m:02d}:{s:02d}"
        self.time_label.setText(f"{fmt(current_sec)} / {fmt(total_sec)}")