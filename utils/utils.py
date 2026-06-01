import shutil
import numpy as np
import torch
import cv2
import os
import torchvision.transforms as T
from pathlib import Path


# SOME BENEFIT FUNCTION OF YOLO AND DEEPBALL

def mapping(info_path):
    class_mapping = {}
    rules = {
        "ball": 0,
        "player": 1,
        "referee": 2,
        "goalkeeper": 3
    }
    with open(info_path, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("trackletID_"):
                parts = line.split("=")
                if len(parts) == 2:
                    tracklet_id = int(parts[0].replace("trackletID_", "").strip())
                    class_name = parts[1].strip().lower()
                    for key, val in rules.items():
                        if key in class_name:
                            class_mapping[tracklet_id] = val
                            break
    return class_mapping

def normalize(xtl, ytl, w, h, img_w, img_h):
    xtl, ytl, w, h = float(xtl), float(ytl), abs(float(w)), abs(float(h))
    x_center = max(0.0, min(1.0, (xtl + w / 2) / img_w))
    y_center = max(0.0, min(1.0, (ytl + h / 2) / img_h))
    width = max(0.0, min(1.0, w / img_w))
    height = max(0.0, min(1.0, h / img_h))
    return round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)

def image_size_from_seqinfo(seqinfo_path):
    if not seqinfo_path.exists():
        return None, None
    img_w, img_h = None, None
    with open(seqinfo_path, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("imWidth="):
                parts = line.split("=")
                if len(parts) == 2:
                    img_w = int(parts[1].strip())
            elif line.startswith("imHeight="):
                parts = line.split("=")
                if len(parts) == 2:
                    img_h = int(parts[1].strip())
            if img_w is not None and img_h is not None:
                break
        if img_w is None or img_h is None:
            print(f"--Không tìm thấy kích thước ảnh trong {seqinfo_path}--")
    return img_w, img_h

def mapping_frame_and_bbox(gt_txt_file, class_mapping, wi, he, stride = 2):
    frame_dict = {}
    if not gt_txt_file.exists():
        print(f"--File gt.txt không tồn tại: {gt_txt_file}--")
        return frame_dict
    with open(gt_txt_file, "r", encoding = "utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            num_img, tracklet_id, xtl, ytl, w, h = parts[:6]
            num_img = int(num_img.strip())
            if num_img % stride != 0:
                continue

            tracklet_id = int(tracklet_id.strip())
            if tracklet_id not in class_mapping:
                continue
            class_id = class_mapping[tracklet_id]
            if class_id == 0:
                xtl, ytl, w, h = float(xtl), float(ytl), abs(float(w)), abs(float(h))
                x_cen = xtl + w / 2
                y_cen = ytl + h / 2
                xtl = x_cen - 20
                ytl = y_cen - 20
                w = 40
                h = 40

            if num_img not in frame_dict:
                frame_dict[num_img] = []
            x_center, y_center, width, height = normalize(xtl, ytl, w, h, wi, he)
            yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
            frame_dict[num_img].append(yolo_line)
    return frame_dict

def create_yolo_label(frame_dict, path, source_img):
    if frame_dict is None:
        return
    image_path = path / "images"
    label_path = path / "labels"
    for key, value in frame_dict.items(): 
        label_name = f"{key:06d}.txt"
        dest_image_path = image_path / f"{key:06d}.jpg"
        source_img_path = source_img / f"{key:06d}.jpg"

        if not source_img_path.exists():
            print(f"--Ảnh {source_img_path} không tồn tại. Bỏ qua--")
            continue
        if not dest_image_path.exists():
            shutil.copy2(source_img_path, dest_image_path)
        lb_path = label_path / label_name
        with open(lb_path, "w", encoding = "utf-8") as f:
            f.write("\n".join(value))


# SOME BENEFIT FUNCTION OF DEEPBALL

def ball_256(gt_txt, class_mapping, width, height, stride = 5):
    if not gt_txt.exists():
        print(f"Không tìm thấy đường dẫn {gt_txt}. Đã dừng chạy chương trình")
        return {}
    with open(gt_txt, "r", encoding = "utf-8") as f:
        ball_dict = {}
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame, id, xtl, ytl, w, h = parts[:6]
            frame, id = int(frame), int(id)
            if frame % stride != 0:
                continue

            if id not in class_mapping:
                continue

            if frame not in ball_dict:
                ball_dict[frame] = []

            if class_mapping[id] == 0:
                xtl, ytl, w, h = float(xtl), float(ytl), float(w), float(h)
                x_center = (xtl + w / 2) / width
                y_center = (ytl + h / 2) / height
                ball_dict[frame].append(f"{x_center:.6f} {y_center:.6f}")
        return ball_dict
    
def move_images_and_labels_deepball(source_images, output_dir, ball_dict):
    if not source_images.exists():
        print(f"Không tìm thấy folder ảnh tại {source_images}")
        return
    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    if not ball_dict:
        print(f"ball_dict trống")
        return
    for key, val in ball_dict.items():
        name = f"{key:06d}"
        image_sou_path = source_images / (name + ".jpg")
        image_out_path = output_images / (name + ".jpg")
        if not image_sou_path.exists():
            print(f"Ảnh không tồn tại tại {image_sou_path}. Đã bỏ qua")
            continue
        if not image_out_path.exists():
            shutil.copy2(image_sou_path, image_out_path)
        label_path = output_labels / (name + ".txt")
        with open(label_path, "w", encoding = "utf-8") as f:
            f.write("\n".join(val))

# SOME DEEPBALL FUNCTION

def build_annotation_cache(root, is_train):
    root_path = Path(root)
    root_path = (root_path / "train") if is_train else (root_path / "test")
    annotation_dict = {}
    for seq in root_path.iterdir():
        if not seq.is_dir():
            continue
        if "-" not in seq.name:
            continue
        pre_name = seq.name.strip().split("-")[1]
        images_path = seq / "images"
        labels_path = seq / "labels"
        for image_path in images_path.glob("*.jpg"):
            name = image_path.name.split(".")[0]
            label_path = labels_path / f"{name}.txt"
            coords = []
            if label_path.exists():
                with open(label_path, "r", encoding = "utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            coords.append([float(parts[0]), float(parts[1])])
            annotation_dict[f"{pre_name}_{name}"] = {
                "image_path": str(image_path),
                "coords": coords
            }
    return annotation_dict

def gaussian_2d(heatmap, center, sigma = 3):
    x, y = center
    h, w = heatmap.shape
    radius = int(3 * sigma)
    x1, y1 = max(0, x - radius), max(0, y - radius)
    x2, y2 = min(w, x + radius + 1), min(h, y + radius + 1)

    X, Y = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    gaussian = np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))
    heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], gaussian)
    return heatmap

def focal_loss(predict, target, alpha = 2, beta = 4) -> torch.Tensor:
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, beta)
    loss = 0
    pos_loss = torch.log(predict + 1e-12) * torch.pow(1 - predict, alpha) * pos_inds
    neg_loss = torch.log(1 - predict + 1e-12) * torch.pow(predict, alpha) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

# def predict_image_deepball(model, frame, x_center, y_center, width, height, size = 256, threshold = 0.3, down_ratio = 4):
#     half_size = int(size / 2)
#     y1 = max(0, y_center - half_size)
#     y2 = min(height, y_center + half_size)
#     x1 = max(0, x_center - half_size)
#     x2 = min(width, x_center + half_size)
#     img_crop = frame[y1:y2, x1:x2]

#     if img_crop.shape[0] != size or img_crop.shape[1] != size:
#         tem_image = np.zeros((size, size, 3), dtype = np.uint8)
#         tem_image[0: (y2 - y1), 0: (x2 - x1)] = img_crop
#         img_crop = tem_image

#     img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
#     device = next(model.parameters()).device
#     transform = T.ToTensor()
#     img_tensor = transform(img_rgb).unsqueeze(0).to(device)
#     if img_tensor.device.type == "cuda":
#         img_tensor = img_tensor.half()

#     with torch.no_grad():
#         predict = model(img_tensor)
#         heatmap = predict[0][0]
#         max_val = torch.max(heatmap).item()
#         if max_val > threshold:
#             max_index = torch.argmax(heatmap).item()
#             heatmap_w = int(size / down_ratio)
#             x = max_index % heatmap_w
#             y = int(max_index / heatmap_w)
#             crop_x = x * down_ratio
#             crop_y = y * down_ratio
#             return int(x1 + crop_x), int(y1 + crop_y)
#         else:
#             return None, None
        
def predict_deepball_trt(frame, x_center, y_center, width, height, d_input, d_output, deepball_context, threshold=0.5):
    # Cắt khung hình 256x256 quanh quả bóng
    crop_size = 256
    x1 = max(0, x_center - crop_size // 2)
    y1 = max(0, y_center - crop_size // 2)
    x2 = min(width, x_center + crop_size // 2)
    y2 = min(height, y_center + crop_size // 2)

    crop_img = frame[y1:y2, x1:x2]
    if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
        return None, None

    # Tiền xử lý: Resize -> BGR to RGB -> Chuẩn hóa về [0, 1]
    img_resized = cv2.resize(crop_img, (256, 256))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

    # Đẩy ảnh vào vùng nhớ GPU đã cọc sẵn
    d_input.copy_(torch.from_numpy(img_tensor))

    # Bấm nút cho TensorRT chạy
    bindings = [int(d_input.data_ptr()), int(d_output.data_ptr())]
    deepball_context.execute_v2(bindings=bindings)

    # Rút kết quả Heatmap ra
    heatmap = d_output.cpu().numpy()[0, 0]

    # Tìm tọa độ bóng
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
    if max_val > threshold:
        scale_x = crop_img.shape[1] / 256.0
        scale_y = crop_img.shape[0] / 256.0
        ball_x_final = x1 + int(max_loc[0] * scale_x)
        ball_y_final = y1 + int(max_loc[1] * scale_y)
        return ball_x_final, ball_y_final

    return None, None