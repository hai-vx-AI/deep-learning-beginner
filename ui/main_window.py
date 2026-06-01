"""
ui/main_window.py — QMainWindow chính, lắp ráp UI.

Thêm floating START/STOP button để chạy screen capture khi đang xem YouTube.
"""

from pathlib import Path
from queue import Queue

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QStatusBar, QFileDialog, QApplication, QScrollArea, QPushButton
)
from PyQt6.QtCore  import Qt, QTimer, pyqtSlot
from PyQt6.QtGui   import QKeyEvent

from ui.styles            import STYLESHEET, ACCENT, SUCCESS, WARNING, DANGER, TEXT_SECONDARY
from ui.video_display     import VideoDisplay
from ui.control_panel     import ControlPanel
from ui.convert_dialog    import ConvertDialog
from ui.processing_thread import ProcessingThread
from ui.ring_buffer_log   import RingBufferLog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Football Analysis System")
        self.resize(1280, 760)
        self.setMinimumSize(1024, 720)

        self._ui_queue: Queue = Queue(maxsize=2)
        self._proc_thread: ProcessingThread = None
        self._output_path: str = "output.mp4"
        self._latest_stats = None

        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(1000)
        self._stats_timer.timeout.connect(self._on_stats_tick)

        self._build_ui()
        self._connect_signals()
        self._build_floating_button()

        QApplication.instance().setStyleSheet(STYLESHEET)

    # ── BUILD UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.video_display = VideoDisplay(self._ui_queue)
        self.control_panel = ControlPanel()

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.control_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(400)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #16213e; }")

        body.addWidget(self.video_display, 1)
        body.addWidget(scroll_area)
        root_layout.addLayout(body, 1)

        self.log_panel = RingBufferLog()
        self.log_panel.setFixedHeight(120)
        root_layout.addWidget(self.log_panel)

        self._build_status_bar()

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("header")
        header.setFixedHeight(50)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 0, 16, 0)

        title = QLabel("⚽  Football Analysis System")
        title.setStyleSheet(f"color: {ACCENT}; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        layout.addStretch()

        hint = QLabel("Phím L: toggle log | Floating button: START/STOP screen")
        hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(hint)

        return header

    def _build_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self._status_dot   = QLabel("●")
        self._status_text  = QLabel("Ready")
        self._status_gpu   = QLabel(self._get_gpu_info())
        self._status_fps   = QLabel("AI: -- FPS")

        for lbl in [self._status_dot, self._status_text,
                    self._status_gpu, self._status_fps]:
            lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")

        self._status_dot.setStyleSheet(f"color: {SUCCESS}; font-size: 14px;")

        self.status_bar.addWidget(self._status_dot)
        self.status_bar.addWidget(self._status_text)
        self.status_bar.addPermanentWidget(self._status_gpu)
        self.status_bar.addPermanentWidget(self._status_fps)

    def _build_floating_button(self):
        self.floating_btn = QPushButton("START AI")
        self.floating_btn.setWindowFlags(
            Qt.WindowType.Tool |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.floating_btn.setFixedSize(110, 42)
        self.floating_btn.move(30, 90)
        self.floating_btn.setStyleSheet(
            "QPushButton { background: #00d4ff; color: #000; border: 1px solid #00d4ff; "
            "font-weight: bold; font-size: 12px; }"
            "QPushButton:hover { background: #ffaa00; color: #000; }"
        )
        self.floating_btn.clicked.connect(self._on_floating_toggle)
        self.floating_btn.show()

    # ── CONNECT SIGNALS ───────────────────────────────────────────────────────

    def _connect_signals(self):
        cp = self.control_panel
        cp.sig_start.connect(self._on_start)
        cp.sig_stop.connect(self._on_stop)
        cp.sig_save.connect(self._on_save)
        cp.sig_convert.connect(self._on_convert)

    # ── SLOTS ─────────────────────────────────────────────────────────────────

    def _on_floating_toggle(self):
        if self._proc_thread and self._proc_thread.isRunning():
            self._on_stop()
            return

        # Floating button ưu tiên screen mode.
        self.control_panel.radio_screen_source.setChecked(True)
        self.control_panel._on_source_changed(1)
        self.control_panel._on_start()

    @pyqtSlot(dict)
    def _on_start(self, config: dict):
        # Dọn queue preview cũ để không hiện frame cũ.
        while not self._ui_queue.empty():
            try:
                self._ui_queue.get_nowait()
            except Exception:
                break

        self._output_path = "screen_output.mp4" if config.get("input_mode") == "screen" else "output.mp4"
        config["output_path"] = self._output_path

        self._proc_thread = ProcessingThread(config, self._ui_queue, self)
        self._proc_thread.stats_updated.connect(self._on_stats_updated)
        self._proc_thread.log_message.connect(self._on_log)
        self._proc_thread.finished_ok.connect(self._on_finished)
        self._proc_thread.finished_error.connect(self._on_processing_error)
        self._proc_thread.start()

        self.video_display.show_message("Đang tải model và xử lý frame đầu tiên...")
        self.video_display.start_display()
        self._stats_timer.start()

        self._set_status("Processing...", WARNING)
        self.floating_btn.setText("STOP AI")
        self.log_panel.append("▶ Bắt đầu xử lý...")

    @pyqtSlot()
    def _on_stop(self):
        if self._proc_thread and self._proc_thread.isRunning():
            self._proc_thread.stop()
            self.log_panel.append("⏹ Dừng thủ công.")

    @pyqtSlot()
    def _on_finished(self):
        self.video_display.stop_display()
        self._stats_timer.stop()
        self.control_panel.on_stopped()
        self._set_status("Ready", SUCCESS)
        self.floating_btn.setText("START AI")
        self.log_panel.append("✓ Hoàn thành.")

    @pyqtSlot(str)
    def _on_processing_error(self, msg: str):
        self.video_display.stop_display()
        self._stats_timer.stop()
        self.control_panel.on_stopped()
        self._set_status("Error", DANGER)
        self.floating_btn.setText("START AI")
        self.log_panel.append(f"✗ Lỗi xử lý: {msg}")

    @pyqtSlot()
    def _on_save(self):
        if not Path(self._output_path).exists():
            self.log_panel.append(f"Không tìm thấy file output: {self._output_path}")
            return

        dest, _ = QFileDialog.getSaveFileName(
            self, "Lưu video output", self._output_path, "Video (*.mp4)"
        )
        if dest:
            import shutil
            shutil.copy2(self._output_path, dest)
            self.log_panel.append(f"💾 Đã lưu: {dest}")

    @pyqtSlot(str)
    def _on_convert(self, pt_path: str):
        dialog = ConvertDialog(pt_path, self)
        dialog.convert_done.connect(self._on_convert_done)
        dialog.exec()

    @pyqtSlot(str)
    def _on_convert_done(self, engine_path: str):
        self.control_panel.edit_deepball.setText(engine_path)
        self.control_panel.radio_tensorrt.setChecked(True)
        self.control_panel.hide_convert_progress()
        self.log_panel.append(f"⚡ Engine sẵn sàng: {engine_path}")

    @pyqtSlot(dict)
    def _on_stats_updated(self, stats: dict):
        self._latest_stats = stats

    @pyqtSlot(str)
    def _on_log(self, msg: str):
        self.log_panel.append(msg)

    def _on_stats_tick(self):
        if not self._latest_stats:
            return
        stats = self._latest_stats
        self.control_panel.update_stats(stats)

        fps = stats.get("fps", 0.0)
        color = SUCCESS if fps >= 20 else (WARNING if fps >= 10 else DANGER)
        self._status_fps.setText(f"AI: {fps} FPS")
        self._status_fps.setStyleSheet(f"color: {color}; font-size: 11px;")

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str):
        self._status_dot.setStyleSheet(f"color: {color}; font-size: 14px;")
        self._status_text.setText(text)
        self._status_text.setStyleSheet(f"color: {color}; font-size: 11px;")

    @staticmethod
    def _get_gpu_info() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                free = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return f"GPU: {name}  ({free:.1f}GB)"
        except Exception:
            pass
        return "GPU: CPU only"

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_L:
            self.log_panel.toggle()
        elif event.key() == Qt.Key.Key_F8:
            self._on_floating_toggle()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        try:
            self.floating_btn.close()
        except Exception:
            pass
        super().closeEvent(event)
