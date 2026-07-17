import os
os.environ['TQDM_ASCII'] = '1'

import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from football_tracking_station.core.deepball_architecture import DeepBallMobileOne
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

def validate_fused_fp16(original_model: torch.nn.Module, 
                        dataloader: DataLoader, 
                        device: torch.device, 
                        use_cuda: bool, 
                        epoch_info: str) -> float:
    """
    Tạo Shadow Model, gộp lớp, ép FP16 và tính Loss.
    Tuyệt đối không làm ảnh hưởng đến luồng Gradient của original_model.
    """
    # 1. Khởi tạo và nạp trọng số (Shadow Model)
    shadow_model = DeepBallMobileOne().to(device)
    shadow_model.load_state_dict(original_model.state_dict())
    shadow_model.eval()

    # 2. Tái tham số hóa (Ép phẳng)
    for module in shadow_model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
            
    # 3. Ép kiểu giới hạn vật lý
    if use_cuda:
        shadow_model = shadow_model.half()

    test_running_loss = 0.0
    val_bar = tqdm(dataloader, desc=f"{epoch_info} [Val Fused FP16]")

    with torch.no_grad():
        for images, heatmaps in val_bar:
            images   = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            
            # Ép input sang FP16
            if use_cuda:
                images = images.half() 
            
            # Inference tốc độ cao
            predictions = shadow_model(images)
            
            # Khôi phục FP32 để chống tràn số cho Focal Loss
            if use_cuda:
                predictions = predictions.float()
            
            loss = focal_loss(predict_logits=predictions, target=heatmaps)
            
            test_running_loss += loss.item()
            val_bar.set_postfix(loss=f"{loss.item():.4f}")

    # 4. Tiêu hủy Shadow Model để chống tràn VRAM
    del shadow_model
    if use_cuda:
        torch.cuda.empty_cache()
        
    if len(dataloader) == 0:
        raise ValueError("Validation dataloader rỗng. Hãy kiểm tra dataset.")
    return test_running_loss / len(dataloader)

def save_fused_deepball_checkpoint(
    original_model: torch.nn.Module,
    save_path: Path,
    device: torch.device,
    epoch: int,
    best_loss: float
):
    """
    Tạo bản DeepBallMobileOne đã reparameterize để dùng cho inference/export.
    Không ảnh hưởng đến model đang training.
    """
    fused_model = DeepBallMobileOne().to(device)
    fused_model.load_state_dict(original_model.state_dict())
    fused_model.eval()

    fused_model.switch_to_deploy()

    torch.save({
        "epoch": epoch,
        "model_state_dict": fused_model.state_dict(),
        "best_loss": best_loss,
        "model_name": "deepball_mobileone",
        "checkpoint_type": "fused_deploy",
        "reparameterized": True,
    }, save_path)

    del fused_model

    if device.type == "cuda":
        torch.cuda.empty_cache()


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
    last_train_weight = weight_path / "deepball_last_train.pt"
    best_train_weight = weight_path / "deepball_best_train.pt"
    best_fused_weight = weight_path / "deepball_best_fused.pt"
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
    model     = DeepBallMobileOne().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                      factor=0.5, patience=3)

    # FIX: GradScaler chỉ khởi tạo khi có CUDA.
    # Nếu dùng CPU → scaler=None, training vẫn chạy bình thường (float32).
    scaler = torch.amp.GradScaler("cuda") if use_cuda else None

    best_loss   = float("inf")
    start_epoch = 0

    # ── RESUME ───────────────────────────────────────────────────────────────
    if last_train_weight.exists():
        print(f"Tìm thấy checkpoint tại {last_train_weight} — đang resume...")

        checkpoint = torch.load(last_train_weight, map_location=device)

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

                loss = focal_loss(
                    predict_logits=predictions.float(),
                    target=heatmaps.float()
                )

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

        if len(train_dataloader) == 0:
            raise ValueError("Train dataloader rỗng. Hãy giảm batch size hoặc kiểm tra dataset.")
        train_epoch_loss = train_running_loss / len(train_dataloader)
        print(f"  Train loss: {train_epoch_loss:.4f}")

        # ── VALIDATION ──
        epoch_str = f"Epoch {epoch+1}/{epochs}"
        test_epoch_loss = validate_fused_fp16(
            original_model=model, 
            dataloader=test_dataloader, 
            device=device, 
            use_cuda=use_cuda,
            epoch_info=epoch_str
        )

        print(f"  Val   loss: {test_epoch_loss:.4f}")

        scheduler.step(test_epoch_loss)

        # ── LƯU CHECKPOINT ──
        is_best = test_epoch_loss < best_loss

        if is_best:
            best_loss = test_epoch_loss

        # 1. Luôn lưu checkpoint train-time mới nhất để resume
        torch.save({
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "current_val_loss":     test_epoch_loss,
            "best_loss":            best_loss,
            "model_name":           "deepball_mobileone",
            "checkpoint_type":      "last_train",
            "reparameterized":      False,
        }, last_train_weight)

        print(f"  Đã lưu deepball_last_train.pt (epoch {epoch + 1})")

        # 2. Nếu tốt nhất, lưu cả bản train-time và bản fused deploy
        if is_best:
            # 2.1. Lưu model gốc chưa fuse
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss":            best_loss,
                "model_name":           "deepball_mobileone",
                "checkpoint_type":      "best_train",
                "reparameterized":      False,
            }, best_train_weight)

            print(f"  ★ Đã lưu deepball_best_train.pt — best_loss = {best_loss:.4f}")

            # 2.2. Lưu model đã fuse để inference/export
            save_fused_deepball_checkpoint(
                original_model=model,
                save_path=best_fused_weight,
                device=device,
                epoch=epoch + 1,
                best_loss=best_loss
            )

            print(f"  ★ Đã lưu deepball_best_fused.pt — dùng cho inference/export")


if __name__ == "__main__":
    pass