"""
ui/processing_thread.py — QThread chạy VideoProcessor độc lập với UI thread.

Hỗ trợ 2 input mode:
  - video : đọc từ file video
  - screen: capture trực tiếp vùng màn hình và gửi overlay data về UI thread
"""

from queue import Queue
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from football_tracking_station.post_processing.pipeline import VideoProcessor


class ProcessingThread(QThread):
    stats_updated   = pyqtSignal(dict)
    overlay_updated = pyqtSignal(dict)
    log_message     = pyqtSignal(str)
    finished_ok     = pyqtSignal()
    finished_error  = pyqtSignal(str)

    def __init__(self, config: dict, ui_queue: Optional[Queue], parent=None):
        super().__init__(parent)
        self.config    = config
        self.ui_queue  = ui_queue
        self.processor = None

    def run(self):
        try:
            self.processor = VideoProcessor(
                yolo_weight      = self.config["yolo_weight"],
                trt_weight       = self.config["deepball_weight"],
                dis_ball_player  = self.config.get("dis_ball_player", 100),
                n_warmup_colors  = self.config.get("n_warmup_colors", 100),
                deepball_thresh  = self.config.get("deepball_thresh", 0.5),
                report_interval  = self.config.get("report_interval", 100),
                vote_window      = self.config.get("vote_window", 15),
                ui_queue         = self.ui_queue,
                stats_callback   = self.stats_updated.emit,
                overlay_callback = self.overlay_updated.emit,
            )

            mode = self.config.get("input_mode", "video")
            output = self.config.get("output_path", "output.mp4")

            if mode == "screen":
                region = self.config.get("screen_region")
                fps = self.config.get("screen_fps", 30)
                save_output = self.config.get("save_screen_output", False)
                self.log_message.emit(
                    f"▶ Screen overlay mode: region={region}, fps={fps}, save_output={save_output}"
                )
                self.processor.process_screen(
                    region=region,
                    fps=fps,
                    output_path=output,
                    save_output=save_output,
                )
            else:
                video_path = self.config["video_path"]
                save_output = self.config.get("save_video_output", False)
                self.log_message.emit(f"▶ Video file mode: {video_path} | save_output={save_output}")
                self.processor.process(video_path, output, save_output=save_output)

            self.stats_updated.emit(self.processor.get_stats())
            self.finished_ok.emit()

        except Exception as e:
            msg = str(e)
            self.log_message.emit(f"[ERROR] {msg}")
            self.finished_error.emit(msg)

    def stop(self):
        if self.processor is not None:
            self.processor.stop()
