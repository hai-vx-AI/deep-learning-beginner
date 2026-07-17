from pathlib import Path
import torch
from ultralytics import YOLO


def train_model(yaml_path: str, pretrain: str = ""):
    """
    Finetune YOLO26s trên dataset football.

    Args:
        yaml_path: Đường dẫn tới file YAML dataset (tạo bởi create_data/create_yaml.py).
        pretrain:  Đường dẫn tới file weight pretrain (.pt). Nếu để trống, dùng yolov8s.pt.
    """
    if not Path(yaml_path).exists():
        print(f"Không tìm thấy file yaml tại: {yaml_path}")
        return

    if pretrain and not Path(pretrain).exists():
        print(f"Không tìm thấy file pretrain tại: {pretrain}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Bạn đang sử dụng thiết bị: {device}")

    model = YOLO(pretrain) if pretrain else YOLO("yolo26s.pt")

    model.train(
        data    = yaml_path,
        epochs  = 50,
        imgsz   = 640,
        device  = device,
        batch   = 4,
        workers = 2,
        project = "football_objecttracking",
        name    = "finetune_phase_1",
        resume  = False,
        verbose = False,

        # Augmentation — giữ nguyên chiến lược gốc
        hsv_h       = 0.0,   # Tắt đổi tông màu
        hsv_s       = 0.2,   # Giảm độ bão hòa
        hsv_v       = 0.2,   # Giảm chớp sáng/tối
        mixup       = 0.0,   # Tắt — tránh ghosting
        mosaic      = 0.3,   # Giảm — tránh cắt bóng ở rìa ghép
        scale       = 0.1,   # Zoom tối đa 10% — giữ bóng không biến mất
        degrees     = 0.0,   # Tắt xoay
        shear       = 0.0,   # Tắt bẻ cong
        perspective = 0.0,   # Tắt phối cảnh 3D
        flipud      = 0.0,   # Không lật dọc
        fliplr      = 0.5,   # Giữ lật ngang
        box = 7.5,
    )
    print("Huấn luyện YOLO hoàn tất!")


if __name__ == "__main__":
    train_model(
        yaml_path="D:\.vscode\football_distance_last\football_tracking_station\yolo26s.yaml",
        pretrain="",  # để trống để dùng yolov8s.pt mặc định
    )