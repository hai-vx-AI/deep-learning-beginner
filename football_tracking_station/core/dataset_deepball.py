import cv2
import numpy as np
import random
import torch
from pathlib import Path
from torch.utils.data import Dataset

# Các hàm gaussian_2d và build_annotation_cache được gộp trực tiếp vào file này
# vì chúng chỉ được dùng duy nhất tại đây — không cần tách ra utils.


def gaussian_2d(heatmap: np.ndarray, center: tuple, sigma: int = 3) -> np.ndarray:
    """Vẽ một nhân Gaussian 2D lên heatmap tại vị trí center."""
    x, y = center
    h, w = heatmap.shape
    radius = int(3 * sigma)
    x1, y1 = max(0, x - radius), max(0, y - radius)
    x2, y2 = min(w, x + radius + 1), min(h, y + radius + 1)
    X, Y = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    gaussian = np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))
    heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], gaussian)
    return heatmap


def build_annotation_cache(root: str, is_train: bool) -> dict:
    """
    Duyệt toàn bộ dataset, xây dựng dict ánh xạ:
      key  : "{seq_id}_{frame_name}"
      value: {"image_paths": [t-2, t-1, t], "coords": [[x, y], ...]}

    Cửa sổ thời gian: [quá khứ xa, quá khứ gần, hiện tại].
    Zero-velocity padding cho 2 frame đầu tiên (dùng lại frame đầu).
    """
    root_path = Path(root) / ("train" if is_train else "test")
    annotation_dict = {}

    for seq in sorted(root_path.iterdir()):
        if not seq.is_dir() or "-" not in seq.name:
            continue

        pre_name = seq.name.strip().split("-")[1]
        images_path = seq / "images"
        labels_path = seq / "labels"

        # BẮT BUỘC sort để bảo toàn trục thời gian
        image_files = sorted(images_path.glob("*.jpg"), key=lambda x: x.name)

        for i, img_t in enumerate(image_files):
            img_t1 = image_files[max(0, i - 1)]   # quá khứ gần (zero-pad nếu i<1)
            img_t2 = image_files[max(0, i - 2)]   # quá khứ xa  (zero-pad nếu i<2)

            name = img_t.stem
            label_path = labels_path / f"{name}.txt"
            coords = []

            # Nhãn chỉ gắn với tọa độ của frame cuối (frame t)
            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            coords.append([float(parts[0]), float(parts[1])])

            annotation_dict[f"{pre_name}_{name}"] = {
                "image_paths": [str(img_t2), str(img_t1), str(img_t)],
                "coords": coords,
            }

    return annotation_dict


class DeepballDataset(Dataset):
    def __init__(self, data_root: str, is_train: bool = True,
                 crop_size: int = 256, down_ratio: int = 4, sigma: int = 3):
        self.data_root = data_root
        self.crop_size = crop_size
        self.down_ratio = down_ratio
        self.sigma = sigma

        self.annotations = build_annotation_cache(self.data_root, is_train)
        self.keys = list(self.annotations.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index: int):
        data = self.annotations[self.keys[index]]
        image_paths = data["image_paths"]   # [t-2, t-1, t]
        raw_coords  = data["coords"]        # tọa độ chuẩn hóa của frame t

        # 1. Load chuỗi 3 frame
        imgs = []
        for path in image_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        h, w = imgs[0].shape[:2]
        heatmap_size = self.crop_size // self.down_ratio
        target_heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)

        # 2. Tính crop window dựa trên frame t
        if len(raw_coords) == 0:
            crop_x = random.randint(0, max(0, w - self.crop_size))
            crop_y = random.randint(0, max(0, h - self.crop_size))
        else:
            pixel_coords = [(int(x * w), int(y * h)) for x, y in raw_coords]
            anchor_x, anchor_y = random.choice(pixel_coords)

            min_cx = max(0, anchor_x - self.crop_size + 10)
            max_cx = min(w - self.crop_size, anchor_x - 10)
            min_cy = max(0, anchor_y - self.crop_size + 10)
            max_cy = min(h - self.crop_size, anchor_y - 10)

            if min_cx > max_cx:
                min_cx = max_cx = max(0, min(anchor_x - self.crop_size // 2, w - self.crop_size))
            if min_cy > max_cy:
                min_cy = max_cy = max(0, min(anchor_y - self.crop_size // 2, h - self.crop_size))

            crop_x = random.randint(int(min_cx), int(max_cx))
            crop_y = random.randint(int(min_cy), int(max_cy))

            # Render heatmap chỉ từ tọa độ frame t
            for px, py in pixel_coords:
                hx = int((px - crop_x) / self.down_ratio)
                hy = int((py - crop_y) / self.down_ratio)
                if 0 <= hx < heatmap_size and 0 <= hy < heatmap_size:
                    target_heatmap = gaussian_2d(target_heatmap, (hx, hy), self.sigma)

        # 3. Crop đồng bộ cả 3 frame với CÙNG crop_x, crop_y
        crop_imgs = [
            img[crop_y: crop_y + self.crop_size, crop_x: crop_x + self.crop_size]
            for img in imgs
        ]

        # 4. Stack kênh: 3 × (H, W, 3) -> (H, W, 9)
        stacked_crop = np.concatenate(crop_imgs, axis=-1)

        # 5. Chuyển tensor: (H, W, 9) -> (9, H, W), chuẩn hóa [0, 1]
        tensor_image   = torch.from_numpy(stacked_crop).permute(2, 0, 1).float() / 255.0
        tensor_heatmap = torch.from_numpy(target_heatmap).unsqueeze(0)

        return tensor_image, tensor_heatmap