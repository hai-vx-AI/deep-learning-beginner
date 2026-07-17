"""
ui/main_window.py — QMainWindow chính, lắp ráp UI.

Screen Capture mode dùng transparent overlay window:
  - AI thread xử lý frame capture.
  - Pipeline gửi bbox/ball/team/fps dạng dict nhẹ.
  - ScreenOverlay vẽ trực tiếp lên vùng màn hình laptop.
"""

from pathlib import Path
from queue import Queue

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QStatusBar, QFileDialog, QApplication, QScrollArea, QPushButton, QMessageBox
)
from PyQt6.QtCore  import Qt, QTimer, pyqtSlot
from PyQt6.QtGui   import QKeyEvent

from ui.styles            import STYLESHEET, ACCENT, SUCCESS, WARNING, DANGER, TEXT_SECONDARY
from ui.video_display     import VideoDisplay
from ui.control_panel     import ControlPanel
from ui.convert_dialog    import ConvertDialog
from ui.processing_thread import ProcessingThread
from ui.ring_buffer_log   import RingBufferLog
from ui.screen_overlay    import ScreenOverlay, exclude_window_from_capture
from ui.region_selector   import ScreenRegionSelector

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
        self._overlay: ScreenOverlay = None
        self._screen_mode_running: bool = False
        self._main_was_minimized_by_screen: bool = False
        self._region_selector: ScreenRegionSelector = None

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

        hint = QLabel("Phím L: toggle log | F8/floating button: START/STOP screen overlay")
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
        self.floating_btn.hide()  # Chỉ hiện khi chọn Screen Capture.
        exclude_window_from_capture(self.floating_btn)

    # ── CONNECT SIGNALS ───────────────────────────────────────────────────────

    def _connect_signals(self):
        cp = self.control_panel
        cp.sig_start.connect(self._on_start)
        cp.sig_stop.connect(self._on_stop)
        cp.sig_save.connect(self._on_save)
        cp.sig_convert.connect(self._on_convert)
        if hasattr(cp, "sig_source_changed"):
            cp.sig_source_changed.connect(self._on_source_changed)
        if hasattr(cp, "sig_select_region"):
            cp.sig_select_region.connect(self._on_select_region)
        self._on_source_changed(cp.get_params().get("input_mode", "video"))

    # ── SLOTS ─────────────────────────────────────────────────────────────────

    def _on_floating_toggle(self):
        if self._proc_thread and self._proc_thread.isRunning():
            self._on_stop()
            return

        # F8/floating button chỉ phục vụ Screen Capture. Không tự đổi từ Video File
        # sang Screen để tránh lỗi dxcam khi người dùng đang định chạy file.
        if not self.control_panel.radio_screen_source.isChecked():
            self._set_status("Select Screen Capture first", WARNING)
            QMessageBox.information(
                self,
                "Screen Capture",
                "Floating button/F8 chỉ dùng cho Screen Capture.\n"
                "Nếu muốn chạy video đã upload, hãy dùng nút START FILE trong panel bên phải."
            )
            return

        self.control_panel._on_start()

    @pyqtSlot(str)
    def _on_source_changed(self, mode: str):
        self._sync_floating_button(mode)

    def _sync_floating_button(self, mode: str = None):
        if not hasattr(self, "floating_btn"):
            return
        if mode is None:
            mode = "screen" if self.control_panel.radio_screen_source.isChecked() else "video"
        running = bool(self._proc_thread and self._proc_thread.isRunning())
        self.floating_btn.setText("STOP AI" if running else "START AI")
        self.floating_btn.setVisible(mode == "screen")
        if mode == "screen":
            self.floating_btn.raise_()

    @pyqtSlot(dict)
    def _on_start(self, config: dict):
        input_mode = config.get("input_mode", "video")
        is_screen = input_mode == "screen"
        self._screen_mode_running = is_screen

        # Dọn queue preview cũ để không hiện frame cũ.
        while not self._ui_queue.empty():
            try:
                self._ui_queue.get_nowait()
            except Exception:
                break

        self._output_path = "screen_output.mp4" if is_screen else "output.mp4"
        config["output_path"] = self._output_path

        if is_screen:
            self._open_screen_overlay(config.get("overlay_region"))
            # Screen capture không preview frame trong QLabel để tránh capture lại UI chính.
            self.video_display.stop_display()
            self.video_display.show_message(
                "Screen Overlay đang chạy.\n"
                "BBox sẽ được vẽ trực tiếp lên vùng màn hình đã chọn.\n"
                "Nhấn F8 hoặc floating STOP AI để dừng."
            )
            ui_queue = None
        else:
            self._close_screen_overlay(restore_main=False)
            self.video_display.show_message("Đang tải model và xử lý frame đầu tiên...")
            self.video_display.start_display()
            ui_queue = self._ui_queue

        self._proc_thread = ProcessingThread(config, ui_queue, self)
        self._proc_thread.stats_updated.connect(self._on_stats_updated)
        self._proc_thread.overlay_updated.connect(self._on_overlay_updated)
        self._proc_thread.log_message.connect(self._on_log)
        self._proc_thread.finished_ok.connect(self._on_finished)
        self._proc_thread.finished_error.connect(self._on_processing_error)
        self._proc_thread.start()

        self._stats_timer.start()
        self._set_status("Processing...", WARNING)
        self._sync_floating_button(input_mode)
        self.log_panel.append("▶ Bắt đầu xử lý...")

        if is_screen:
            # Cho overlay/floating button kịp show trước, rồi thu nhỏ UI chính
            # để tránh capture ngược chính cửa sổ app.
            self._main_was_minimized_by_screen = True
            QTimer.singleShot(500, self.showMinimized)

    @pyqtSlot()
    def _on_stop(self):
        if self._proc_thread and self._proc_thread.isRunning():
            self._proc_thread.stop()
            self.log_panel.append("⏹ Dừng thủ công.")

    @pyqtSlot()
    def _on_finished(self):
        self.video_display.stop_display()
        self._stats_timer.stop()
        self._proc_thread = None
        self.control_panel.force_idle()
        self._set_status("Ready", SUCCESS)
        self._close_screen_overlay(restore_main=True)
        self._sync_floating_button()
        self.log_panel.append("✓ Hoàn thành.")

    @pyqtSlot(str)
    def _on_processing_error(self, msg: str):
        self.video_display.stop_display()
        self._stats_timer.stop()
        self._proc_thread = None
        self.control_panel.force_idle(can_save=False)
        self._set_status("Error", DANGER)
        self._close_screen_overlay(restore_main=True)
        self.video_display.show_message(f"Lỗi xử lý:\n{msg}")
        self._sync_floating_button()
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

    @pyqtSlot(dict)
    def _on_overlay_updated(self, data: dict):
        if self._overlay is not None:
            self._overlay.update_data(data)

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

    # ── REGION SELECTION ──────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_select_region(self):
        """Open a CVAT-like selector: click corner 1, click corner 2."""
        if self._proc_thread and self._proc_thread.isRunning():
            QMessageBox.information(
                self,
                "Select Region",
                "Hãy dừng AI trước khi chọn lại vùng capture."
            )
            return

        self._set_status("Selecting screen region...", WARNING)
        self.log_panel.append("▣ Chọn vùng capture: click góc trên-trái rồi click góc dưới-phải.")

        # Minimize main UI so the user can select the browser/video underneath.
        self.showMinimized()
        QTimer.singleShot(250, self._show_region_selector)

    def _show_region_selector(self):
        self._region_selector = ScreenRegionSelector()
        self._region_selector.region_selected.connect(self._on_region_selected)
        self._region_selector.selection_cancelled.connect(self._on_region_selection_cancelled)
        self._region_selector.show()
        self._region_selector.raise_()
        self._region_selector.activateWindow()

    @pyqtSlot(object, object)
    def _on_region_selected(self, logical_region, physical_region):
        self.control_panel.set_screen_region(logical_region, physical_region)
        self._region_selector = None
        self._set_status("Region selected", SUCCESS)
        self.log_panel.append(
            f"✓ Region logical={logical_region}, capture={physical_region}"
        )
        self.showNormal()
        self.raise_()
        self.activateWindow()

    @pyqtSlot()
    def _on_region_selection_cancelled(self):
        self._region_selector = None
        self._set_status("Ready", SUCCESS)
        self.log_panel.append("↩ Hủy chọn vùng capture. Giữ cấu hình vùng hiện tại.")
        self.showNormal()
        self.raise_()
        self.activateWindow()

    # ── SCREEN OVERLAY ────────────────────────────────────────────────────────

    def _open_screen_overlay(self, region):
        """
        Open overlay in Qt logical pixels.

        region is overlay_region from ControlPanel. If None, use full primary screen.
        Capture itself may use a different physical-pixel region in ScreenFrameSource.
        """
        self._close_screen_overlay(restore_main=False)

        if region is None:
            screen = QApplication.primaryScreen()
            geom = screen.geometry()
            region = (geom.left(), geom.top(), geom.left() + geom.width(), geom.top() + geom.height())

        self._overlay = ScreenOverlay(region)
        self._overlay.show()
        self._overlay.raise_()
        excluded = exclude_window_from_capture(self._overlay)
        if excluded:
            self.log_panel.append("✓ Overlay đã được yêu cầu loại khỏi screen capture.")
        else:
            self.log_panel.append(
                "ℹ Overlay chạy bình thường. Nếu bị capture ngược, hãy thu nhỏ UI chính hoặc chỉnh vùng capture."
            )

    def _close_screen_overlay(self, restore_main: bool = True):
        if self._overlay is not None:
            try:
                self._overlay.close()
            except Exception:
                pass
            self._overlay = None

        self._screen_mode_running = False
        if restore_main and self._main_was_minimized_by_screen:
            self._main_was_minimized_by_screen = False
            self.showNormal()
            self.raise_()
            self.activateWindow()

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
            self._close_screen_overlay(restore_main=False)
        except Exception:
            pass
        try:
            if self._region_selector is not None:
                self._region_selector.close()
        except Exception:
            pass
        try:
            self.floating_btn.close()
        except Exception:
            pass
        super().closeEvent(event)
