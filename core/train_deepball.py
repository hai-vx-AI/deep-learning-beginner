import os
import sys
import torch
import torch.optim as optim
from pathlib import Path
from core.deepball_arkitecture import DeepBall
from core.dataset_deepball import DeepballDataset
# from deepball_arkitecture import DeepBall
# from dataset_deepball import DeepballDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import focal_loss

def train_deepball(root_path, weight_path = "weights", epochs = 50, batch = 8, learning_rate = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device.type.upper()}")
    weight_path = Path(weight_path)
    weight_path.mkdir(parents = True, exist_ok = True)
    last_weight = weight_path / "last.pt"
    best_weight = weight_path / "best.pt"

    if not os.path.isdir(root_path):
        print(f"Đường dẫn gốc bị lỗi {root_path}. Đã tạm dừng chương trình")
        return

    train_dataset = DeepballDataset(data_root = root_path, is_train = True, down_ratio = 4, sigma = 4)
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = batch,
        shuffle = True,
        num_workers = 2,
        drop_last = True
    )
    test_dataset = DeepballDataset(data_root = root_path, is_train = False, down_ratio = 4, sigma = 4)
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = batch,
        shuffle = False,
        num_workers = 2,
        drop_last = False
    )

# INITIALIZE MODEL
    
    model = DeepBall().to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    best_loss = float("inf")
    start_epoch = 0

    if last_weight.exists():
        print(f"Đã tìm thấy last checkpoint tại {last_weight}")
        checkpoint = torch.load(last_weight, map_location = device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint.get("best_loss", float("inf"))

    print("Bắt đầu quá trình huấn luyện")
    for epoch in range(start_epoch, epochs):
        model.train()
        train_running_loss = 0.0
        train_progressbar = tqdm(train_dataloader, desc = f"{epoch + 1}/{epochs} [Train]")
        for images, heatmaps in train_progressbar:
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = focal_loss(predict = predictions, target = heatmaps)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            
            train_progressbar.set_postfix(loss = f"{loss.item():.4f}")
        train_epoch_loss = train_running_loss / len(train_dataloader)
        print(f"Tổng kết train loss trung bình: {train_epoch_loss}")

        model.eval()
        test_running_loss = 0.0
        test_progressbar = tqdm(test_dataloader)
        with torch.no_grad():
            for images, heatmaps in test_progressbar:
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                predictions = model(images)
                loss = focal_loss(predict = predictions, target = heatmaps)
                test_running_loss += loss.item()
                test_progressbar.set_postfix(loss = f"{loss.item():.4f}")
        test_epoch_loss = test_running_loss / len(test_dataloader)
        print(f"Tổng kết val loss trung bình: {test_epoch_loss}")

        if best_loss > test_epoch_loss:
            best_loss = test_epoch_loss
            print(f"Cập nhật kết quả tốt nhất: best loss = {best_loss}")
            best_point = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss
            }
            torch.save(best_point, best_weight)
            print(f"Đã lưu mô hình tốt nhất tại {best_weight}")

        last_point = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "current_val_loss": test_epoch_loss,
            "best_loss": best_loss
        }
        torch.save(last_point, last_weight)

if __name__ == "__main__":
    pass