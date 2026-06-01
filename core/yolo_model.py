from ultralytics import YOLO
from pathlib import Path
import torch

def train_model(pretrain = ""):
    yaml_path = r"D:\.vscode\football_distance\core\yolov8.yaml"
    if not Path(yaml_path).exists():
        print(f"Không tìm thấy file yaml tại đường dẫn: {yaml_path}")
        return
    if pretrain and not Path(pretrain).exists():
        print(f"Không tìm thấy file pretrain tại đường dẫn: {pretrain}")
        print("Chương trình tạm dừng")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Bạn đang sử dụng thiết bị: {device}")
    model = YOLO(pretrain) if pretrain else YOLO("yolov8s.pt")

    results = model.train(
        data = yaml_path,
        epochs = 1,
        imgsz = 1280,
        device = device,
        batch = 4,
        workers = 2,
        project = "football_objecttracking",
        name = "finetune_phase_1",
        resume = False,

        hsv_h=0.0,        # [Mặc định 0.015] Tắt hoàn toàn việc đổi tông màu (Hue)
        hsv_s=0.2,        # [Mặc định 0.7] Giảm việc đổi độ bão hòa màu
        hsv_v=0.2,        # [Mặc định 0.4] Giảm độ chớp sáng/tối
    
        mixup=0.0,        # [Mặc định 0.0] Đảm bảo TẮT. Tránh hiện tượng 2 ảnh đè lên nhau (ghosting)
        mosaic=0.3,       # [Mặc định 1.0] Giảm mạnh. Mosaic tốt cho bối cảnh nhưng hay cắt ngang quả bóng ở rìa ghép
        
        scale=0.1,        # [Mặc định 0.5] RẤT QUAN TRỌNG: Chỉ cho phép zoom in/out tối đa 10% để bóng không bị biến mất
        degrees=0.0,      # Tắt xoay nghiêng ảnh (Bóng đá không quay lộn ngược)
        shear=0.0,        # Tắt bẻ cong hình học
        perspective=0.0,  # Tắt phối cảnh 3D
        flipud=0.0,       # Không lật ngược ảnh từ trên xuống
        fliplr=0.5,       # Chỉ giữ lại lật ngang (Trái-Phải) vì nó phản ánh đúng tính chất sân cỏ
        box=7.5
    )
    print("Huấn luyện hoàn tất!")

if __name__ == "__main__":
    pretrain = r"D:\.vscode\football_distance\weight\yolo_best.pt"
    train_model(pretrain)