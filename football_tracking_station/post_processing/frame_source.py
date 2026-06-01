"""
Frame sources for the inference pipeline.

- FileFrameSource: reads frames from a normal video file.
- ScreenFrameSource: captures a screen region in realtime.

Screen capture priority on Windows:
  1) dxcam: fastest path for realtime screen capture.
  2) mss: fallback if dxcam is not installed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


Region = Tuple[int, int, int, int]  # left, top, right, bottom


class FileFrameSource:
    """Read BGR frames from a video file with cv2.VideoCapture."""

    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        if not Path(self.video_path).is_file():
            raise RuntimeError(f"Không tìm thấy video: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được video: {self.video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()


class ScreenFrameSource:
    """
    Capture a screen region and return BGR frames.

    region format: (left, top, right, bottom)
    Example: (200, 120, 1480, 840) for a 1280x720 YouTube area.
    """

    def __init__(self, region: Optional[Region] = None, fps: int = 30):
        self.region = region
        self.fps = int(fps or 30)
        self.total = 0
        self._backend = None
        self._camera = None
        self._sct = None
        self._monitor = None
        self._last_tick = 0.0
        self._frame_interval = 1.0 / max(self.fps, 1)

        self._init_backend()

        ok, frame = self.read()
        if not ok or frame is None:
            raise RuntimeError(
                "Không lấy được frame màn hình. Hãy kiểm tra vùng capture hoặc thử cài: pip install dxcam"
            )

        self.height, self.width = frame.shape[:2]

    def _init_backend(self) -> None:
        # Fast backend for Windows.
        try:
            import dxcam  # type: ignore

            self._backend = "dxcam"
            self._camera = dxcam.create(output_color="BGR")
            self._camera.start(
                region=self.region,
                target_fps=self.fps,
                video_mode=True,
            )
            return
        except Exception:
            pass

        # Portable fallback.
        try:
            import mss  # type: ignore

            self._backend = "mss"
            self._sct = mss.mss()
            if self.region is None:
                mon = self._sct.monitors[1]
                self._monitor = {
                    "left": mon["left"],
                    "top": mon["top"],
                    "width": mon["width"],
                    "height": mon["height"],
                }
            else:
                left, top, right, bottom = self.region
                self._monitor = {
                    "left": int(left),
                    "top": int(top),
                    "width": int(right - left),
                    "height": int(bottom - top),
                }
            return
        except Exception as exc:
            raise RuntimeError(
                "Chưa cài thư viện capture màn hình. Cài nhanh bằng: pip install dxcam"
            ) from exc

    def read(self):
        if self._backend == "dxcam":
            frame = self._camera.get_latest_frame()
            if frame is None:
                time.sleep(0.001)
                return False, None
            return True, frame

        # mss fallback: throttle a little to avoid burning CPU in capture only.
        now = time.perf_counter()
        delay = self._frame_interval - (now - self._last_tick)
        if delay > 0:
            time.sleep(delay)
        self._last_tick = time.perf_counter()

        shot = self._sct.grab(self._monitor)
        frame = np.asarray(shot)
        # mss returns BGRA. Drop alpha -> BGR.
        frame = frame[:, :, :3]
        return True, frame

    def release(self) -> None:
        if self._backend == "dxcam" and self._camera is not None:
            self._camera.stop()
        if self._backend == "mss" and self._sct is not None:
            self._sct.close()
