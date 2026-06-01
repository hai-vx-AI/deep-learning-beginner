import os
import sys
import cv2
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import build_annotation_cache, gaussian_2d


class DeepballDataset(Dataset):
    def __init__(self, data_root, is_train = True, crop_size = 256, down_ratio = 1, sigma = 3):
        self.data_root = data_root
        self.crop_size = crop_size
        self.down_ratio = down_ratio
        self.sigma = sigma

        self.annotations = build_annotation_cache(self.data_root, is_train)
        self.keys = list(self.annotations.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        unique_key = self.keys[index]
        data = self.annotations[unique_key]
        image_path = data["image_path"]
        raw_coords = data["coords"]

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        heatmap_size = self.crop_size // self.down_ratio
        target_heatmap = np.zeros((heatmap_size, heatmap_size), dtype = np.float32)
        if len(raw_coords) == 0:
            max_x = max(0, w - self.crop_size)
            max_y = max(0, h - self.crop_size)
            crop_x = random.randint(0, max_x)
            crop_y = random.randint(0, max_y)
        else:
            pixel_coords = [(int(x * w), int(y * h)) for x, y in raw_coords]
            anchor_x, anchor_y = random.choice(pixel_coords)
            min_crop_x = max(0, anchor_x - self.crop_size + 10)
            max_crop_x = min(w - self.crop_size, anchor_x - 10)
            min_crop_y = max(0, anchor_y - self.crop_size + 10)
            max_crop_y = min(h - self.crop_size, anchor_y - 10)

            if min_crop_x > max_crop_x:
                min_crop_x = max_crop_x = max(0, min(anchor_x - self.crop_size // 2, w - self.crop_size))
            if min_crop_y > max_crop_y:
                min_crop_y = max_crop_y = max(0, min(anchor_y - self.crop_size // 2, h - self.crop_size))

            crop_x = random.randint(int(min_crop_x), int(max_crop_x))
            crop_y = random.randint(int(min_crop_y), int(max_crop_y))

            for px, py in pixel_coords:
                new_x = int((px - crop_x) / self.down_ratio)
                new_y = int((py - crop_y) / self.down_ratio)
                if 0 <= new_x < heatmap_size and 0 <= new_y < heatmap_size:
                    target_heatmap = gaussian_2d(target_heatmap, (new_x, new_y), self.sigma)
        crop_img = img[crop_y: crop_y + self.crop_size, crop_x: crop_x + self.crop_size]
        tensor_image = torch.from_numpy(crop_img).permute(2, 0, 1).float() / 255.0
        tensor_heatmap = torch.from_numpy(target_heatmap).unsqueeze(0)
        return tensor_image, tensor_heatmap


if __name__ == "__main__":
    data_root = "deepball_data"


    dataset = DeepballDataset(data_root)
    idx = 40
    image_tensor, heatmap_tensor = dataset[idx] 