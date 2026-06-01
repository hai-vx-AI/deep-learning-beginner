"""
ui/convert_dialog.py — Modal dialog convert .pt → TensorRT .engine.
Chạy convert trong QThread riêng để không block UI.
"""

from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ui.styles import ACCENT, WARNING, TEXT_PRIMARY, TEXT_SECONDARY, BG_DARK


class ConvertThread(QThread):
    """QThread thực hiện convert .pt → ONNX → TRT engine."""

    progress_updated = pyqtSignal(int, str)    # (percent, step_text)
    finished_ok      = pyqtSignal(str)         # engine_path
    finished_error   = pyqtSignal(str)         # error message

    def __init__(self, pt_path: str, parent=None):
        super().__init__(parent)
        self.pt_path = pt_path

    def run(self):
        try:
            import torch
            from pathlib import Path

            pt_path     = Path(self.pt_path)
            onnx_path   = pt_path.with_suffix(".onnx")
            engine_path = pt_path.with_suffix(".engine")

            # ── BƯỚC 1: Load model ──────────────────────────────────────────
            self.progress_updated.emit(5, "Loading PyTorch model...")
            from football_tracking_station.core.deepball_architecture import DeepBall

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model  = DeepBall()
            ckpt   = torch.load(self.pt_path, map_location=device)
            state  = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)
            model.eval().to(device)

            # ── BƯỚC 2: Export ONNX ─────────────────────────────────────────
            self.progress_updated.emit(25, "Exporting to ONNX...")
            dummy = torch.randn(1, 9, 256, 256).to(device)
            torch.onnx.export(
                model, dummy, str(onnx_path),
                input_names   = ["input"],
                output_names  = ["logits"],
                opset_version = 11,
                do_constant_folding = True,
            )

            # ── BƯỚC 3: Convert TRT ─────────────────────────────────────────
            self.progress_updated.emit(50, "Converting ONNX → TensorRT (FP16)...")
            import subprocess
            result = subprocess.run([
                "trtexec",
                f"--onnx={onnx_path}",
                f"--saveEngine={engine_path}",
                "--fp16",
                "--workspace=512",
            ], capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                raise RuntimeError(f"trtexec failed:\n{result.stderr[-500:]}")

            # ── BƯỚC 4: Cleanup ONNX ────────────────────────────────────────
            self.progress_updated.emit(95, "Cleaning up...")
            onnx_path.unlink(missing_ok=True)

            self.progress_updated.emit(100, "Done!")
            self.finished_ok.emit(str(engine_path))

        except Exception as e:
            self.finished_error.emit(str(e))


class ConvertDialog(QDialog):
    """Modal dialog hiển thị tiến trình convert."""

    convert_done = pyqtSignal(str)   # engine_path khi xong

    def __init__(self, pt_path: str, parent=None):
        super().__init__(parent)
        self.pt_path = pt_path
        self.setWindowTitle("Convert TensorRT Engine")
        self.setFixedSize(460, 260)
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.FramelessWindowHint
        )
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {BG_DARK};
                border: 2px solid {ACCENT};
            }}
        """)

        self._build_ui()
        self._start_convert()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Convert to TensorRT Engine")
        title.setStyleSheet(f"color: {ACCENT}; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        info = QLabel(
            "Quá trình này chỉ cần thực hiện 1 lần.\n"
            "Engine sẽ được lưu cùng thư mục với file .pt\n"
            "Thời gian ước tính: 2-5 phút (tùy GPU)"
        )
        info.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; line-height: 1.6;")
        layout.addWidget(info)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{ background: #0a0a1a; border: none; }}
            QProgressBar::chunk {{ background: {WARNING}; }}
        """)
        layout.addWidget(self.progress_bar)

        self.step_label = QLabel("Đang khởi tạo...")
        self.step_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(self.step_label)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setFixedSize(100, 34)
        self.btn_cancel.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: #ff4757;
                border: 1px solid #ff4757;
            }}
            QPushButton:hover {{
                background: #ff4757;
                color: white;
            }}
        """)
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

    def _start_convert(self):
        self._thread = ConvertThread(self.pt_path, self)
        self._thread.progress_updated.connect(self._on_progress)
        self._thread.finished_ok.connect(self._on_done)
        self._thread.finished_error.connect(self._on_error)
        self._thread.start()

    def _on_progress(self, value: int, text: str):
        self.progress_bar.setValue(value)
        self.step_label.setText(text)

    def _on_done(self, engine_path: str):
        self.step_label.setText(f"✓ Đã lưu: {engine_path}")
        self.btn_cancel.setText("Close")
        self.convert_done.emit(engine_path)

    def _on_error(self, msg: str):
        self.step_label.setText(f"Lỗi: {msg[:120]}")
        self.step_label.setStyleSheet("color: #ff4757; font-size: 11px;")
        self.btn_cancel.setText("Close")

    def _on_cancel(self):
        if self._thread.isRunning():
            self._thread.terminate()
        self.reject()