"""
utils/utils.py — Các hàm tiện ích DÙNG CHUNG cho bước chuẩn bị dữ liệu.

Các hàm đã được chuyển đi:
  - gaussian_2d          → core/dataset_deepball.py  (chỉ dùng ở đó)
  - build_annotation_cache → core/dataset_deepball.py (chỉ dùng ở đó)
  - focal_loss           → core/train_deepball.py    (chỉ dùng trong training loop)
  - predict_deepball_trt → ĐÃ XÓA (lỗi thời — dùng input 3-channel cũ,
                           cần viết lại cho kiến trúc 9-channel mới khi
                           cập nhật module post_propressing_yolo/)
"""

import shutil
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# DÙNG CHUNG: prepare_data.py + prepare_deepball_data.py
# ─────────────────────────────────────────────────────────────────────────────

def mapping(info_path: Path) -> dict:
    """Đọc gameinfo.ini, trả về {tracklet_id: class_id}."""
    class_mapping = {}
    rules = {"ball": 0, "player": 1, "referee": 2, "goalkeeper": 3}
    with open(info_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("trackletID_"):
                parts = line.split("=")
                if len(parts) == 2:
                    tracklet_id = int(parts[0].replace("trackletID_", "").strip())
                    class_name  = parts[1].strip().lower()
                    for key, val in rules.items():
                        if key in class_name:
                            class_mapping[tracklet_id] = val
                            break
    return class_mapping


def image_size_from_seqinfo(seqinfo_path: Path):
    """Đọc seqinfo.ini, trả về (width, height). Trả về (None, None) nếu lỗi."""
    if not seqinfo_path.exists():
        return None, None
    img_w, img_h = None, None
    with open(seqinfo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("imWidth="):
                img_w = int(line.split("=")[1].strip())
            elif line.startswith("imHeight="):
                img_h = int(line.split("=")[1].strip())
            if img_w is not None and img_h is not None:
                break
    if img_w is None or img_h is None:
        print(f"--Không tìm thấy kích thước ảnh trong {seqinfo_path}--")
    return img_w, img_h


# ─────────────────────────────────────────────────────────────────────────────
# DÙNG BỞI: prepare_data.py (YOLO pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def normalize(xtl, ytl, w, h, img_w, img_h) -> tuple:
    """Chuyển bbox (xtl, ytl, w, h) sang định dạng YOLO chuẩn hóa."""
    xtl, ytl, w, h = float(xtl), float(ytl), abs(float(w)), abs(float(h))
    x_center = max(0.0, min(1.0, (xtl + w / 2) / img_w))
    y_center = max(0.0, min(1.0, (ytl + h / 2) / img_h))
    width    = max(0.0, min(1.0, w / img_w))
    height   = max(0.0, min(1.0, h / img_h))
    return round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)


def mapping_frame_and_bbox(gt_txt_file: Path, class_mapping: dict,
                           wi: int, he: int, stride: int = 2) -> dict:
    """
    Đọc gt.txt, lọc theo stride, chuẩn hóa bbox.
    Với class bóng (id=0): cố định bbox thành hình vuông 40×40 xung quanh tâm.
    Trả về {frame_idx: [yolo_line, ...]}.
    """
    frame_dict = {}
    if not gt_txt_file.exists():
        print(f"--File gt.txt không tồn tại: {gt_txt_file}--")
        return frame_dict

    with open(gt_txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            num_img, tracklet_id, xtl, ytl, w, h = parts[:6]
            num_img     = int(num_img.strip())
            tracklet_id = int(tracklet_id.strip())

            if num_img % stride != 0 or tracklet_id not in class_mapping:
                continue

            class_id = class_mapping[tracklet_id]
            if class_id == 0:
                xtl, ytl, w, h = float(xtl), float(ytl), abs(float(w)), abs(float(h))
                cx = xtl + w / 2
                cy = ytl + h / 2
                xtl, ytl, w, h = cx - 20, cy - 20, 40.0, 40.0

            if num_img not in frame_dict:
                frame_dict[num_img] = []
            xc, yc, nw, nh = normalize(xtl, ytl, w, h, wi, he)
            frame_dict[num_img].append(f"{class_id} {xc} {yc} {nw} {nh}")

    return frame_dict


def create_yolo_label(frame_dict: dict, path: Path, source_img: Path):
    """Copy ảnh + ghi label YOLO cho từng frame trong frame_dict."""
    if not frame_dict:
        return
    image_path = path / "images"
    label_path = path / "labels"

    for key, value in frame_dict.items():
        dest_img_path = image_path / f"{key:06d}.jpg"
        src_img_path  = source_img  / f"{key:06d}.jpg"

        if not src_img_path.exists():
            print(f"--Ảnh {src_img_path} không tồn tại. Bỏ qua--")
            continue
        if not dest_img_path.exists():
            shutil.copy2(src_img_path, dest_img_path)

        lb_path = label_path / f"{key:06d}.txt"
        with open(lb_path, "w", encoding="utf-8") as f:
            f.write("\n".join(value))


# ─────────────────────────────────────────────────────────────────────────────
# DÙNG BỞI: prepare_deepball_data.py (DeepBall pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def ball_256(gt_txt: Path, class_mapping: dict, width: int, height: int,
             stride: int = 1) -> dict:
    """
    Đọc gt.txt, lọc chỉ lấy bóng (class 0).
    Trả về {frame_idx: ["x_norm y_norm", ...]}.
    """
    if not gt_txt.exists():
        print(f"Không tìm thấy đường dẫn {gt_txt}. Đã dừng chạy chương trình")
        return {}

    ball_dict = {}
    with open(gt_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame, obj_id, xtl, ytl, w, h = parts[:6]
            frame, obj_id = int(frame), int(obj_id)

            if frame % stride != 0 or obj_id not in class_mapping:
                continue
            if class_mapping[obj_id] != 0:
                continue

            xtl, ytl, w, h = float(xtl), float(ytl), float(w), float(h)
            x_center = (xtl + w / 2) / width
            y_center = (ytl + h / 2) / height

            if frame not in ball_dict:
                ball_dict[frame] = []
            ball_dict[frame].append(f"{x_center:.6f} {y_center:.6f}")

    return ball_dict


def move_images_and_labels_deepball(source_images: Path, output_dir: Path, ball_dict: dict):
    """
    Copy ảnh từ source_images sang output_dir/images.
    Ghi file label (hoặc file rỗng nếu không có bóng) sang output_dir/labels.
    """
    if not source_images.exists():
        print(f"Không tìm thấy folder ảnh tại {source_images}")
        return
    if not ball_dict:
        print("ball_dict trống")
        return

    output_images = output_dir / "images"
    output_labels = output_dir / "labels"

    for image_sou_path in sorted(source_images.glob("*.jpg")):
        frame_name = image_sou_path.stem
        frame_idx  = int(frame_name)

        image_out_path = output_images / (frame_name + ".jpg")
        if not image_out_path.exists():
            shutil.copy2(image_sou_path, image_out_path)

        label_path = output_labels / (frame_name + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            if frame_idx in ball_dict:
                f.write("\n".join(ball_dict[frame_idx]))