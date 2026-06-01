import os
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from football_tracking_station.core.deepball_architecture import DeepBall
from football_tracking_station.core.dataset_deepball import DeepballDataset


def focal_loss(predict_logits: torch.Tensor, target: torch.Tensor,
               alpha: int = 2, beta: int = 4) -> torch.Tensor:
    """
    CenterNet-style Focal Loss.
    Nhận RAW LOGITS — tự gọi sigmoid bên trong.
    Model head KHÔNG được có nn.Sigmoid().
    """
    predict = torch.clamp(torch.sigmoid(predict_logits), min=1e-4, max=1.0 - 1e-4)

    pos_inds    = target.eq(1).float()
    neg_inds    = target.lt(1).float()
    neg_weights = torch.pow(1.0 - target, beta)

    pos_loss = torch.log(predict)       * torch.pow(1.0 - predict, alpha) * pos_inds
    neg_loss = torch.log(1.0 - predict) * torch.pow(predict, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        loss = -neg_loss.sum()
    else:
        loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos

    return loss


def train_deepball(root_path: str, weight_path: str = None,
                   epochs: int = 50, batch: int = 8, learning_rate: float = 1e-5):

    # ── DEVICE ──────────────────────────────────────────────────────────────
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print(f"Đang sử dụng thiết bị: {device.type.upper()}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── WEIGHT PATH ──────────────────────────────────────────────────────────
    weight_path = Path(weight_path) if weight_path else Path("weights")
    weight_path.mkdir(parents=True, exist_ok=True)
    last_weight = weight_path / "last.pt"
    best_weight = weight_path / "best.pt"
    print(f"Checkpoint sẽ được lưu tại: {weight_path.resolve()}")

    if not os.path.isdir(root_path):
        print(f"Đường dẫn gốc bị lỗi: {root_path}. Đã tạm dừng.")
        return

    # ── DATASET ──────────────────────────────────────────────────────────────
    print("Đang xây dựng annotation cache (train)...")
    train_dataset = DeepballDataset(data_root=root_path, is_train=True,
                                    down_ratio=4, sigma=4)
    print(f"  Train: {len(train_dataset)} samples")

    print("Đang xây dựng annotation cache (val)...")
    test_dataset  = DeepballDataset(data_root=root_path, is_train=False,
                                    down_ratio=4, sigma=4)
    print(f"  Val  : {len(test_dataset)} samples")

    # num_workers=2 an toàn trên Colab (Linux); =0 nếu chạy Windows
    num_workers = 2 if os.name != "nt" else 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                                  num_workers=num_workers, pin_memory=use_cuda,
                                  drop_last=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch, shuffle=False,
                                  num_workers=num_workers, pin_memory=use_cuda,
                                  drop_last=False)

    # ── MODEL ────────────────────────────────────────────────────────────────
    model     = DeepBall().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                      factor=0.5, patience=3)

    # FIX: GradScaler chỉ khởi tạo khi có CUDA.
    # Nếu dùng CPU → scaler=None, training vẫn chạy bình thường (float32).
    scaler = torch.amp.GradScaler("cuda") if use_cuda else None

    best_loss   = float("inf")
    start_epoch = 0

    # ── RESUME ───────────────────────────────────────────────────────────────
    if last_weight.exists():
        print(f"Tìm thấy checkpoint tại {last_weight} — đang resume...")
        checkpoint  = torch.load(last_weight, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_loss   = checkpoint.get("best_loss", float("inf"))
        print(f"  Resume từ epoch {start_epoch}, best_loss = {best_loss:.4f}")

    # ── TRAINING LOOP ────────────────────────────────────────────────────────
    print(f"\nBắt đầu huấn luyện từ epoch {start_epoch + 1}/{epochs}")

    for epoch in range(start_epoch, epochs):

        # ── TRAIN ──
        model.train()
        train_running_loss = 0.0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for images, heatmaps in train_bar:
            images   = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            optimizer.zero_grad()

            # FIX: autocast và scaler chỉ dùng khi có CUDA
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predictions = model(images)
                    loss = focal_loss(predict_logits=predictions, target=heatmaps)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(images)
                loss = focal_loss(predict_logits=predictions, target=heatmaps)
                loss.backward()
                optimizer.step()

            train_running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_epoch_loss = train_running_loss / len(train_dataloader)
        print(f"  Train loss: {train_epoch_loss:.4f}")

        # ── VALIDATION ──
        model.eval()
        test_running_loss = 0.0
        val_bar = tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")

        with torch.no_grad():
            for images, heatmaps in val_bar:
                images      = images.to(device, non_blocking=True)
                heatmaps    = heatmaps.to(device, non_blocking=True)
                predictions = model(images)
                loss        = focal_loss(predict_logits=predictions, target=heatmaps)
                test_running_loss += loss.item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        test_epoch_loss = test_running_loss / len(test_dataloader)
        print(f"  Val   loss: {test_epoch_loss:.4f}")

        scheduler.step(test_epoch_loss)

        # ── LƯU CHECKPOINT ──
        # last.pt: luôn lưu sau mỗi epoch (dùng để resume)
        torch.save({
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "current_val_loss":     test_epoch_loss,
            "best_loss":            best_loss,
        }, last_weight)
        print(f"  Đã lưu last.pt (epoch {epoch + 1})")

        # best.pt: chỉ lưu khi val_loss cải thiện
        if test_epoch_loss < best_loss:
            best_loss = test_epoch_loss
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss":            best_loss,
            }, best_weight)
            print(f"  ★ Đã lưu best.pt — best_loss = {best_loss:.4f}")


if __name__ == "__main__":
    pass