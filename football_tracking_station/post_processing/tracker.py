from collections import deque
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import torch

from football_tracking_station.post_processing.detector import Detection, BallPosition
from football_tracking_station.core.deepball_architecture import DeepBall


class DeepBallTracker:
    """
    Hỗ trợ 2 mode — tự động chọn dựa vào đuôi file:
      - .pt     → PyTorch inference (chậm hơn, không cần convert)
      - .engine → TensorRT inference (nhanh hơn, cần convert trước 1 lần)

    State:
      frame_buffer  : deque 3 frame [t-2, t-1, t]
      last_position : BallPosition hợp lệ cuối cùng
      lost_frames   : số frame liên tiếp mất bóng
    """

    CROP_SIZE = 256
    MAX_LOST  = 30

    def __init__(self, weight_path: str, threshold: float = 0.5):
        self.frame_buffer:  deque                  = deque(maxlen=3)
        self.last_position: Optional[BallPosition] = None
        self.lost_frames:   int                    = 0
        self.threshold:     float                  = threshold

        suffix = Path(weight_path).suffix.lower()

        if suffix == ".pt":
            self._mode = "pytorch"
            self._load_pytorch(weight_path)
        elif suffix == ".engine":
            self._mode = "tensorrt"
            self._load_trt(weight_path)
        else:
            raise ValueError(
                f"Đuôi file '{suffix}' không được hỗ trợ. "
                "Dùng .pt (PyTorch) hoặc .engine (TensorRT)."
            )

    # ── LOAD ──────────────────────────────────────────────────────────────────

    def _load_pytorch(self, weight_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = DeepBall().to(self.device)

        checkpoint = torch.load(weight_path, map_location=self.device)
        # Hỗ trợ cả 2 format: checkpoint dict hoặc state_dict thẳng
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # fp16 nếu có GPU để tăng tốc
        self.use_fp16 = self.device.type == "cuda"
        if self.use_fp16:
            self.model.half()

        print(f"[DeepBallTracker] Mode: PyTorch | device: {self.device} | "
              f"fp16: {self.use_fp16} | {weight_path}")

    def _load_trt(self, weight_path: str) -> None:
        try:
            import tensorrt as trt
        except ImportError:
            raise RuntimeError(
                "TensorRT chưa được cài. Dùng file .pt hoặc cài TensorRT trước."
            )
        TRT_LOGGER  = trt.Logger(trt.Logger.WARNING)
        with open(weight_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.trt_engine  = runtime.deserialize_cuda_engine(f.read())
        self.trt_context = self.trt_engine.create_execution_context()

        # Pre-allocate GPU buffers cố định — không malloc mỗi frame
        self.d_input  = torch.empty(
            (1, 9, self.CROP_SIZE, self.CROP_SIZE), dtype=torch.float32, device="cuda"
        )
        self.d_output = torch.empty(
            (1, 1, self.CROP_SIZE // 4, self.CROP_SIZE // 4),
            dtype=torch.float32, device="cuda"
        )
        print(f"[DeepBallTracker] Mode: TensorRT | {weight_path}")

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    @property
    def buffer_ready(self) -> bool:
        return len(self.frame_buffer) == 3

    def update(self, frame: np.ndarray,
               yolo_ball: Optional[Detection],
               frame_w: int, frame_h: int) -> Optional[BallPosition]:
        """Gọi mỗi frame. Trả về BallPosition hoặc None."""
        self.frame_buffer.append(frame.copy())

        if yolo_ball is not None:
            roi_x, roi_y     = yolo_ball.cx, yolo_ball.cy
            self.lost_frames = 0
        elif self.last_position is not None and self.lost_frames < self.MAX_LOST:
            roi_x, roi_y = self.last_position.x, self.last_position.y
            self.lost_frames += 1
        else:
            self.lost_frames  += 1
            self.last_position = None
            return None

        result = self._predict(roi_x, roi_y, frame_w, frame_h)
        if result is not None:
            self.last_position = result
        return result

    # ── PREDICT ───────────────────────────────────────────────────────────────

    def _predict(self, roi_cx: int, roi_cy: int,
                 frame_w: int, frame_h: int) -> Optional[BallPosition]:
        if not self.buffer_ready:
            return None

        tensor_np, x1, y1 = self._prepare_input(roi_cx, roi_cy, frame_w, frame_h)
        if tensor_np is None:
            return None

        if self._mode == "pytorch":
            heatmap = self._infer_pytorch(tensor_np)
        else:
            heatmap = self._infer_trt(tensor_np)

        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
        if max_val < self.threshold:
            return None

        heatmap_h, heatmap_w = heatmap.shape
        ball_x = x1 + int(max_loc[0] * self.CROP_SIZE / heatmap_w)
        ball_y = y1 + int(max_loc[1] * self.CROP_SIZE / heatmap_h)

        return BallPosition(x=ball_x, y=ball_y,
                            source=self._mode, confidence=float(max_val))

    def _prepare_input(self, roi_cx: int, roi_cy: int,
                       frame_w: int, frame_h: int):
        """
        Crop 256×256 ROI đồng bộ trên cả 3 frame.
        Trả về (tensor_np shape (9,256,256), x1, y1) hoặc (None, _, _).
        """
        half = self.CROP_SIZE // 2
        x1   = max(0, roi_cx - half)
        y1   = max(0, roi_cy - half)
        x2   = min(frame_w, x1 + self.CROP_SIZE)
        y2   = min(frame_h, y1 + self.CROP_SIZE)
        x1   = x2 - self.CROP_SIZE   # điều chỉnh nếu crop sát biên phải/dưới
        y1   = y2 - self.CROP_SIZE

        crops = []
        for f in self.frame_buffer:   # [t-2, t-1, t]
            crop = f[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            if crop_rgb.shape[:2] != (self.CROP_SIZE, self.CROP_SIZE):
                crop_rgb = cv2.resize(crop_rgb, (self.CROP_SIZE, self.CROP_SIZE))
            crops.append(crop_rgb)

        # (H, W, 9) → (9, H, W), float32 [0,1]
        stacked  = np.concatenate(crops, axis=-1)
        tensor_np = stacked.transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor_np, x1, y1

    def _infer_pytorch(self, tensor_np: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(tensor_np).unsqueeze(0)  # (1, 9, 256, 256)
        tensor = tensor.to(self.device)
        if self.use_fp16:
            tensor = tensor.half()

        with torch.no_grad():
            logits  = self.model(tensor)                   # raw logits
            heatmap = torch.sigmoid(logits)[0, 0]          # (64, 64)

        return heatmap.float().cpu().numpy()

    def _infer_trt(self, tensor_np: np.ndarray) -> np.ndarray:
        self.d_input.copy_(torch.from_numpy(tensor_np))
        bindings = [int(self.d_input.data_ptr()), int(self.d_output.data_ptr())]
        self.trt_context.execute_v2(bindings=bindings)
        return self.d_output.cpu().numpy()[0, 0]