from pathlib import Path
from utils import mapping, normalize, image_size_from_seqinfo, mapping_frame_and_bbox, create_yolo_label
from tqdm import tqdm

train = True
path = ""
stride = 2

# FUNCTION: CREATE DATA PREPARE FOR TRAIN MODEL
def pre_data(root, path, is_train = True, stride = 2):
    root_dir = Path(root)
    output_dir = Path(path)
    if not root_dir.exists() or not root_dir.is_dir():
        print(f"--Thư mục không hợp lệ: {root_dir}--")
        return
    output_dir.mkdir(parents = True, exist_ok = True)
    data_split = root_dir / ("train" if is_train else "test")

    if not data_split.exists():
        print(f"--Thư mục dữ liệu không tồn tại: {data_split}--")
        return
    
    sequence_dirs = [d for d in data_split.iterdir() if d.is_dir()]
    if not sequence_dirs:
        print(f"--Không tìm thấy thư mục con trong {data_split}--")
        return

    for ff_dir in tqdm(sequence_dirs, desc = "Tiến trình xử lí video", unit = "video"):

        cre_dir = output_dir / ff_dir.name
        ori_images_dir = cre_dir / "images"
        ori_labels_dir = cre_dir / "labels"
        ori_images_dir.mkdir(parents = True, exist_ok = True)
        ori_labels_dir.mkdir(parents = True, exist_ok = True)

        gt_txt_file = ff_dir / "gt" / "gt.txt"
        images_dir = ff_dir / "img1"
        info_file = ff_dir / "gameinfo.ini"
        seq_file = ff_dir / "seqinfo.ini"
        if not (gt_txt_file.exists() and images_dir.exists() and info_file.exists() and seq_file.exists()):
            print(f"--Thiếu file hoặc folder trong {ff_dir}. Đã bỏ qua--")
            continue
        class_mapping = mapping(info_file)
        w, h = image_size_from_seqinfo(seq_file)
        if w is None or h is None:
            print(f"--Không xác định được kích thước ảnh cho {ff_dir}. Đã bỏ qua--")
            continue
        frame_dict = mapping_frame_and_bbox(gt_txt_file, class_mapping, w, h, stride = stride)
        create_yolo_label(frame_dict, cre_dir, images_dir)