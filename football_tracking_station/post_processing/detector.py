from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import torch
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# DATA CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Một object được detect + track bởi YOLO trong 1 frame."""
    track_id: int
    class_id: int     # 0=ball  1=player  2=referee  3=goalkeeper
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def foot_x(self) -> int:
        """Tọa độ X điểm chân cầu thủ (tâm cạnh dưới bbox)."""
        return self.cx

    @property
    def foot_y(self) -> int:
        """Tọa độ Y điểm chân cầu thủ (cạnh dưới bbox)."""
        return self.y2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class BallPosition:
    """Vị trí bóng sau khi được xác nhận (từ YOLO hoặc DeepBall)."""
    x: int
    y: int
    source: str        # "yolo" | "deepball" | "predicted"
    confidence: float


# ─────────────────────────────────────────────────────────────────────────────
# YOLO DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class YoloDetector:
    """
    Bọc YOLO model.track() và chuẩn hóa output thành List[Detection].

    Trách nhiệm:
    - Load model
    - Chạy tracking mỗi frame
    - Parse raw tensor output → list Detection có type annotation rõ ràng
    - Trả về bóng tốt nhất (confidence cao nhất) riêng biệt
    """

    def __init__(self, weight_path: str):
        self.model = YOLO(weight_path)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.use_half = torch.cuda.is_available()
        print(
            f"YoloDetector: Đã load model từ {weight_path} | "
            f"device={self.device} | half={self.use_half}"
        )

    def run(self, frame: np.ndarray) -> List[Detection]:
        """
        Chạy track trên 1 frame.
        Trả về list Detection. List rỗng nếu không detect được gì.
        """
        results = self.model.track(
            frame,
            tracker  = "bytetrack.yaml",
            conf     = 0.25,
            iou      = 0.5,
            persist  = True,
            verbose  = False,
            device   = self.device,
            half     = self.use_half,
        )

        detections: List[Detection] = []
        boxes = results[0].boxes

        if boxes is None or boxes.id is None:
            return detections

        xyxy_arr  = boxes.xyxy.cpu().numpy()
        id_arr    = boxes.id.cpu().numpy()
        class_arr = boxes.cls.cpu().numpy()
        conf_arr  = boxes.conf.cpu().numpy()

        for i in range(len(id_arr)):
            x1, y1, x2, y2 = xyxy_arr[i]
            detections.append(Detection(
                track_id = int(id_arr[i]),
                class_id = int(class_arr[i]),
                x1       = int(x1),
                y1       = int(y1),
                x2       = int(x2),
                y2       = int(y2),
                conf     = float(conf_arr[i]),
            ))

        return detections

    @staticmethod
    def best_ball(detections: List[Detection]) -> Optional[Detection]:
        """
        Lọc ra bóng (class_id=0) có confidence cao nhất.
        Trả về None nếu không có bóng nào.
        """
        balls = [d for d in detections if d.class_id == 0]
        if not balls:
            return None
        return max(balls, key=lambda d: d.conf)

    @staticmethod
    def get_players(detections: List[Detection]) -> List[Detection]:
        """Lọc ra các cầu thủ (class_id=1)."""
        return [d for d in detections if d.class_id == 1]