"""
ui/control_panel.py — Panel điều khiển bên phải.

Thêm Screen Capture Mode:
  - Video File: đọc từ file video như cũ.
  - Screen Capture: đọc trực tiếp vùng màn hình laptop.
"""

from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QRadioButton, QButtonGroup, QSlider, QSpinBox,
    QProgressBar, QFrame, QFileDialog, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from User_interface.styles import DANGER


class ControlPanel(QWidget):
    sig_start   = pyqtSignal(dict)
    sig_stop    = pyqtSignal()
    sig_save    = pyqtSignal()
    sig_convert = pyqtSignal(str)
    sig_source_changed = pyqtSignal(str)   # "video" | "screen"
    sig_select_region = pyqtSignal()      # ask MainWindow to show region selector

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(384)
        self.setStyleSheet("background-color: #16213e;")
        self._last_can_save = False
        self._screen_region_logical = None    # Qt logical pixels for overlay
        self._screen_region_physical = None   # Physical pixels for dxcam/mss capture

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(self._build_input_files())
        layout.addWidget(self._build_deepball_mode())
        layout.addWidget(self._build_parameters())
        layout.addWidget(self._build_stats_control())
        layout.addStretch()

        self._on_source_changed(0)

    # ── SECTION 1: INPUT / SOURCE ────────────────────────────────────────────

    def _build_input_files(self) -> QFrame:
        frame = self._card()
        layout = QVBoxLayout(frame)
        layout.setSpacing(6)

        layout.addWidget(self._section_title("INPUT SOURCE"))

        self.radio_video_source = QRadioButton("Video File")
        self.radio_screen_source = QRadioButton("Screen Capture")
        self.radio_video_source.setChecked(True)

        self._source_group = QButtonGroup()
        self._source_group.addButton(self.radio_video_source, 0)
        self._source_group.addButton(self.radio_screen_source, 1)
        self._source_group.idClicked.connect(self._on_source_changed)

        src_row = QHBoxLayout()
        src_row.addWidget(self.radio_video_source)
        src_row.addWidget(self.radio_screen_source)
        layout.addLayout(src_row)

        self.edit_video    = self._file_row(layout, "Video", "*.mp4 *.avi *.mov")
        self.edit_yolo     = self._file_row(layout, "YOLO Weight", "*.pt")
        self.edit_deepball = self._file_row(layout, "DeepBall Weight", "*.pt *.engine")

        layout.addWidget(self._sublabel("Screen region: default = full screen"))
        self.lbl_screen_region = self._sublabel("Region: Full screen (default)")
        layout.addWidget(self.lbl_screen_region)

        region_row = QHBoxLayout()
        self.btn_select_region = QPushButton("Select Region")
        self.btn_reset_region = QPushButton("Full Screen")
        self.btn_select_region.setFixedHeight(30)
        self.btn_reset_region.setFixedHeight(30)
        self.btn_select_region.clicked.connect(self.sig_select_region.emit)
        self.btn_reset_region.clicked.connect(self.clear_screen_region)
        region_row.addWidget(self.btn_select_region, 1)
        region_row.addWidget(self.btn_reset_region, 1)
        layout.addLayout(region_row)

        self.spin_screen_fps    = self._spin_row(layout, "Capture FPS", 5, 60, 30)

        return frame

    def _file_row(self, layout: QVBoxLayout, label: str, filt: str) -> QLineEdit:
        layout.addWidget(self._sublabel(label))
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setFixedHeight(28)
        edit.setPlaceholderText("Chọn đường dẫn...")
        btn = QPushButton("Browse")
        btn.setFixedSize(68, 28)
        btn.clicked.connect(lambda _, e=edit, f=filt: self._browse(e, f))
        row.addWidget(edit)
        row.addWidget(btn)
        layout.addLayout(row)
        # Lưu button để bật/tắt theo mode video/screen.
        edit._browse_button = btn
        return edit

    def _browse(self, edit: QLineEdit, filt: str):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn file", "", filt)
        if path:
            edit.setText(path)
            self._on_deepball_path_changed()

    def _on_source_changed(self, source_id: int):
        is_screen = source_id == 1
        mode = "screen" if is_screen else "video"

        # Video path chỉ cần ở chế độ upload file.
        self.edit_video.setEnabled(not is_screen)
        if hasattr(self.edit_video, "_browse_button"):
            self.edit_video._browse_button.setEnabled(not is_screen)

        # Screen region controls chỉ cần ở chế độ capture màn hình.
        for w in [
            self.lbl_screen_region,
            self.btn_select_region,
            self.btn_reset_region,
            self.spin_screen_fps,
        ]:
            w.setEnabled(is_screen)

        # Đổi text nút chính để người dùng biết đang chạy luồng nào.
        if hasattr(self, "btn_start"):
            self.btn_start.setText("▶  START OVERLAY" if is_screen else "▶  START FILE")

        self.sig_source_changed.emit(mode)

    def set_screen_region(self, logical_region, physical_region):
        """Receive a selected region from MainWindow's ScreenRegionSelector."""
        self._screen_region_logical = tuple(int(v) for v in logical_region)
        self._screen_region_physical = tuple(int(v) for v in physical_region)
        l, t, r, b = self._screen_region_logical
        pl, pt, pr, pb = self._screen_region_physical
        self.lbl_screen_region.setText(
            f"Region: {l},{t} → {r},{b}  | capture: {pl},{pt} → {pr},{pb}"
        )

    def clear_screen_region(self):
        """Use full screen capture/overlay when the user does not select a region."""
        self._screen_region_logical = None
        self._screen_region_physical = None
        self.lbl_screen_region.setText("Region: Full screen (default)")

    # ── SECTION 2: DEEPBALL MODE ──────────────────────────────────────────────

    def _build_deepball_mode(self) -> QFrame:
        frame = self._card()
        layout = QVBoxLayout(frame)
        layout.setSpacing(6)

        layout.addWidget(self._section_title("DEEPBALL MODE"))

        self.radio_pytorch  = QRadioButton("PyTorch  (.pt)")
        self.radio_tensorrt = QRadioButton("TensorRT  (.engine)")
        self.radio_pytorch.setChecked(True)

        sub_pt  = self._sublabel("Chạy ngay, không cần convert")
        sub_trt = self._sublabel("Nhanh hơn, nhưng cần .engine đúng")

        self._mode_group = QButtonGroup()
        self._mode_group.addButton(self.radio_pytorch,  0)
        self._mode_group.addButton(self.radio_tensorrt, 1)
        self._mode_group.idClicked.connect(self._on_mode_changed)

        layout.addWidget(self.radio_pytorch)
        layout.addWidget(sub_pt)
        layout.addWidget(self.radio_tensorrt)
        layout.addWidget(sub_trt)

        self.btn_convert = QPushButton("Convert & Save Engine...")
        self.btn_convert.setObjectName("btn_convert")
        self.btn_convert.setFixedHeight(32)
        self.btn_convert.setEnabled(False)
        self.btn_convert.clicked.connect(self._on_convert_clicked)
        layout.addWidget(self.btn_convert)

        self.convert_progress = QProgressBar()
        self.convert_progress.setFixedHeight(6)
        self.convert_progress.setRange(0, 100)
        self.convert_progress.setValue(0)
        self.convert_progress.setVisible(False)

        self.convert_step_label = self._sublabel("")
        self.convert_step_label.setVisible(False)

        layout.addWidget(self.convert_progress)
        layout.addWidget(self.convert_step_label)

        return frame

    def _on_mode_changed(self, mode_id: int):
        is_trt = mode_id == 1
        engine_exists = self._engine_exists()
        self.btn_convert.setEnabled(is_trt and not engine_exists)

    def _on_deepball_path_changed(self):
        self._on_mode_changed(self._mode_group.checkedId())

    def _engine_exists(self) -> bool:
        raw_path = self.edit_deepball.text().strip()
        if not raw_path:
            return False
        path = Path(raw_path)
        if path.suffix.lower() == ".engine":
            return path.exists()
        return path.with_suffix(".engine").exists()

    def _on_convert_clicked(self):
        pt_path = self.edit_deepball.text().strip()
        if not pt_path:
            return
        if Path(pt_path).suffix.lower() != ".pt":
            QMessageBox.warning(self, "Sai định dạng", "Convert chỉ nhận file .pt")
            return
        self.sig_convert.emit(pt_path)

    def show_convert_progress(self, value: int, step_text: str):
        self.convert_progress.setVisible(True)
        self.convert_step_label.setVisible(True)
        self.convert_progress.setValue(value)
        self.convert_step_label.setText(step_text)

    def hide_convert_progress(self):
        self.convert_progress.setVisible(False)
        self.convert_step_label.setVisible(False)
        self.btn_convert.setEnabled(False)

    def get_deepball_path(self) -> str:
        raw_path = self.edit_deepball.text().strip()
        if self.radio_tensorrt.isChecked():
            path = Path(raw_path)
            if path.suffix.lower() == ".engine":
                return str(path)
            return str(path.with_suffix(".engine"))
        return raw_path

    # ── SECTION 3: PARAMETERS ─────────────────────────────────────────────────

    def _build_parameters(self) -> QFrame:
        frame = self._card()
        layout = QVBoxLayout(frame)
        layout.setSpacing(6)

        layout.addWidget(self._section_title("PARAMETERS"))

        self.slider_conf, self.lbl_conf = self._slider_row(layout, "Confidence", 0.05, 1.0, 0.25)
        self.slider_dist, self.lbl_dist = self._slider_row(layout, "Ball distance (px)", 20, 300, 100, is_int=True)
        self.spin_warmup = self._spin_row(layout, "Warmup samples", 50, 500, 100)
        self.spin_vote   = self._spin_row(layout, "Vote window", 5, 50, 15)

        return frame

    def _slider_row(self, layout, label, min_val, max_val, default, is_int=False):
        row_label = QHBoxLayout()
        lbl_name  = self._sublabel(label)
        lbl_val   = self._sublabel(str(default))
        lbl_val.setFixedWidth(35)
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row_label.addWidget(lbl_name)
        row_label.addStretch()
        row_label.addWidget(lbl_val)
        layout.addLayout(row_label)

        slider = QSlider(Qt.Orientation.Horizontal)
        if is_int:
            slider.setRange(int(min_val), int(max_val))
            slider.setValue(int(default))
        else:
            slider.setRange(0, 100)
            slider.setValue(int((default - min_val) / (max_val - min_val) * 100))
            slider._min = min_val
            slider._max = max_val

        slider.sliderReleased.connect(
            lambda s=slider, l=lbl_val, ii=is_int: self._on_slider_release(s, l, ii)
        )
        layout.addWidget(slider)
        return slider, lbl_val

    def _on_slider_release(self, slider, lbl, is_int):
        if is_int:
            val = slider.value()
        else:
            val = slider._min + slider.value() / 100 * (slider._max - slider._min)
            val = round(val, 2)
        lbl.setText(str(val))

    def _spin_row(self, layout, label, min_val, max_val, default) -> QSpinBox:
        row = QHBoxLayout()
        row.addWidget(self._sublabel(label))
        row.addStretch()
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setFixedWidth(80)
        row.addWidget(spin)
        layout.addLayout(row)
        return spin

    def get_params(self) -> dict:
        conf = self.slider_conf._min + self.slider_conf.value() / 100 * (self.slider_conf._max - self.slider_conf._min)
        is_screen = self.radio_screen_source.isChecked()

        return {
            "input_mode":      "screen" if is_screen else "video",
            "video_path":      self.edit_video.text().strip(),
            # None means full screen. When selected, keep physical/logical regions separate
            # to avoid bbox offset on Windows display scaling.
            "screen_region":   self._screen_region_physical if is_screen else None,
            "overlay_region":  self._screen_region_logical if is_screen else None,
            "screen_fps":      self.spin_screen_fps.value(),
            "save_screen_output": False,
            "save_video_output": self.chk_save_video.isChecked() if hasattr(self, "chk_save_video") else False,
            "yolo_weight":     self.edit_yolo.text().strip(),
            "deepball_weight": self.get_deepball_path(),
            "deepball_thresh": round(conf, 2),
            "dis_ball_player": self.slider_dist.value(),
            "n_warmup_colors": self.spin_warmup.value(),
            "vote_window":     self.spin_vote.value(),
        }

    # ── SECTION 4: STATS & CONTROL ────────────────────────────────────────────

    def _build_stats_control(self) -> QFrame:
        frame = self._card()
        layout = QVBoxLayout(frame)
        layout.setSpacing(6)

        layout.addWidget(self._section_title("POSSESSION"))

        row0 = QHBoxLayout()
        lbl0 = QLabel("Team 0")
        lbl0.setFixedWidth(55)
        lbl0.setStyleSheet(f"color: {DANGER};")
        self.bar_team0 = QProgressBar()
        self.bar_team0.setObjectName("progress_possession_0")
        self.bar_team0.setRange(0, 100)
        self.bar_team0.setValue(50)
        self.bar_team0.setFixedHeight(8)
        self.lbl_pct0 = self._sublabel("50%")
        self.lbl_pct0.setFixedWidth(35)
        self.lbl_pct0.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row0.addWidget(lbl0)
        row0.addWidget(self.bar_team0, 1)
        row0.addWidget(self.lbl_pct0)
        layout.addLayout(row0)

        row1 = QHBoxLayout()
        lbl1 = QLabel("Team 1")
        lbl1.setFixedWidth(55)
        lbl1.setStyleSheet("color: #aaaaaa;")
        self.bar_team1 = QProgressBar()
        self.bar_team1.setObjectName("progress_possession_1")
        self.bar_team1.setRange(0, 100)
        self.bar_team1.setValue(50)
        self.bar_team1.setFixedHeight(8)
        self.lbl_pct1 = self._sublabel("50%")
        self.lbl_pct1.setFixedWidth(35)
        self.lbl_pct1.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row1.addWidget(lbl1)
        row1.addWidget(self.bar_team1, 1)
        row1.addWidget(self.lbl_pct1)
        layout.addLayout(row1)

        self.chk_save_video = QCheckBox("Save processed video file")
        self.chk_save_video.setToolTip(
            "Tắt mặc định để giảm khựng khi preview. Bật nếu bạn muốn xuất output.mp4."
        )
        self.chk_save_video.setChecked(False)
        layout.addWidget(self.chk_save_video)

        self.btn_start = QPushButton("▶  START FILE")
        self.btn_stop  = QPushButton("STOP")
        self.btn_save  = QPushButton("Save Output")

        self.btn_start.setObjectName("btn_start")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_save.setObjectName("btn_save")

        self.btn_start.setFixedHeight(38)
        self.btn_stop.setFixedHeight(34)
        self.btn_save.setFixedHeight(34)

        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self.sig_stop.emit)
        self.btn_save.clicked.connect(self.sig_save.emit)

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_save)

        return frame

    def _on_start(self):
        params = self.get_params()
        missing_fields = []

        if params["input_mode"] == "video" and not params["video_path"]:
            missing_fields.append("Video")
        if not params["yolo_weight"]:
            missing_fields.append("YOLO Weight")
        if not params["deepball_weight"]:
            missing_fields.append("DeepBall Weight")

        if missing_fields:
            QMessageBox.critical(
                self,
                "Lỗi Tham Số Đầu Vào",
                "Bạn chưa nạp các tài nguyên sau:\n- " + "\n- ".join(missing_fields)
            )
            return

        invalid_files = []
        if params["input_mode"] == "video" and not Path(params["video_path"]).is_file():
            invalid_files.append("Video")
        if not Path(params["yolo_weight"]).is_file():
            invalid_files.append("YOLO Weight")
        if not Path(params["deepball_weight"]).is_file():
            invalid_files.append("DeepBall Weight / TensorRT Engine")

        if invalid_files:
            QMessageBox.critical(
                self,
                "Không tìm thấy file",
                "Các file sau không tồn tại hoặc sai đường dẫn:\n- " + "\n- ".join(invalid_files)
            )
            return

        self._last_can_save = (params["input_mode"] == "video" and params.get("save_video_output", False)) or params.get("save_screen_output", False)
        self.set_running()
        self.sig_start.emit(params)

    def on_stopped(self):
        self.force_idle()

    def force_idle(self, can_save=None):
        """Đưa cụm nút về trạng thái nghỉ, kể cả khi worker lỗi rất sớm."""
        if can_save is not None:
            self._last_can_save = bool(can_save)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(self._last_can_save)

    def set_running(self):
        """Đưa cụm nút về trạng thái đang xử lý."""
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)

    def update_stats(self, stats: dict):
        pct = stats.get("possession_pct", {0: 50, 1: 50})
        p0  = pct.get(0, 50)
        p1  = pct.get(1, 50)
        self.bar_team0.setValue(p0)
        self.bar_team1.setValue(p1)
        self.lbl_pct0.setText(f"{p0}%")
        self.lbl_pct1.setText(f"{p1}%")

    # ── HELPERS ───────────────────────────────────────────────────────────────

    @staticmethod
    def _card() -> QFrame:
        frame = QFrame()
        frame.setObjectName("card")
        return frame

    @staticmethod
    def _section_title(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("section_title")
        return lbl

    @staticmethod
    def _sublabel(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sublabel")
        return lbl
