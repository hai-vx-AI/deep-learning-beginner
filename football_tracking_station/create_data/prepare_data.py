from pathlib import Path
from tqdm import tqdm

from football_tracking_station.utils.utils import (
    mapping,
    normalize,
    image_size_from_seqinfo,
    mapping_frame_and_bbox,
    create_yolo_label,
)


def pre_data(root: str, path: str, is_train: bool = True, stride: int = 2):
    """
    Chuẩn bị dữ liệu YOLO từ SoccerNet tracking format.

    Args:
        root:     Thư mục gốc chứa train/ và test/.
        path:     Thư mục output.
        is_train: True để xử lý tập train, False để xử lý tập test.
        stride:   Chỉ lấy 1 frame mỗi `stride` frame.
    """
    root_dir   = Path(root)
    split      = "train" if is_train else "test"
    # output_dir phải có tầng train/ hoặc test/ để create_yaml quét đúng cấu trúc:
    #   data/train/<seq>/images/
    #   data/test/<seq>/images/
    output_dir = Path(path) / split

    if not root_dir.exists() or not root_dir.is_dir():
        print(f"--Thư mục không hợp lệ: {root_dir}--")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    data_split = root_dir / split

    if not data_split.exists():
        print(f"--Thư mục dữ liệu không tồn tại: {data_split}--")
        return

    sequence_dirs = [d for d in data_split.iterdir() if d.is_dir()]
    if not sequence_dirs:
        print(f"--Không tìm thấy thư mục con trong {data_split}--")
        return

    for ff_dir in tqdm(sequence_dirs, desc="Tiến trình xử lí video", unit="video"):
        cre_dir        = output_dir / ff_dir.name
        ori_images_dir = cre_dir / "images"
        ori_labels_dir = cre_dir / "labels"
        ori_images_dir.mkdir(parents=True, exist_ok=True)
        ori_labels_dir.mkdir(parents=True, exist_ok=True)

        gt_txt_file = ff_dir / "gt" / "gt.txt"
        images_dir  = ff_dir / "img1"
        info_file   = ff_dir / "gameinfo.ini"
        seq_file    = ff_dir / "seqinfo.ini"

        if not all(p.exists() for p in [gt_txt_file, images_dir, info_file, seq_file]):
            print(f"--Thiếu file hoặc folder trong {ff_dir}. Đã bỏ qua--")
            continue

        class_mapping = mapping(info_file)
        w, h = image_size_from_seqinfo(seq_file)
        if w is None or h is None:
            print(f"--Không xác định được kích thước ảnh cho {ff_dir}. Đã bỏ qua--")
            continue

        frame_dict = mapping_frame_and_bbox(gt_txt_file, class_mapping, w, h, stride=stride)
        create_yolo_label(frame_dict, cre_dir, images_dir)