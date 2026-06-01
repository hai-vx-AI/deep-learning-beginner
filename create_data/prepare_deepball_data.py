from pathlib import Path
from tqdm import tqdm
from utils import mapping, image_size_from_seqinfo, ball_256, move_images_and_labels_deepball


def prepare_deepball(input_path, output_path, is_train = True, stride = 5):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    output = (output_path / "train") if is_train else (output_path / "test")
    output.mkdir(parents = True, exist_ok = True)

    if not input_path.exists():
        print(f"Lỗi đường dẫn {input_path}. Đã dừng chạy chương trình")
        return
    data_path = (input_path / "train") if is_train else (input_path / "test")
    if not data_path.exists():
        print(f"Lỗi đường dẫn {data_path}. Đã dừng chạy chương trình")
        return
    data_dir = [a for a in data_path.iterdir() if a.is_dir()]
    if not data_dir:
        print(f"Không tìm thấy thư mục con trong {data_dir}")
        return
    progress_bar = tqdm(data_dir, desc = "Tiến trình: ", unit = "video")
    for ff in progress_bar:
        ff_path = output / str(ff.name)
        cre_images = ff_path / "images"
        cre_labels = ff_path / "labels"
        ff_path.mkdir(parents = True, exist_ok = True)
        cre_images.mkdir(parents = True, exist_ok = True)
        cre_labels.mkdir(parents = True, exist_ok = True)

        images_path = ff / "img1"
        labels_path = ff / "gt" / "gt.txt"
        game_info = ff / "gameinfo.ini"
        seqinfo = ff / "seqinfo.ini"
        class_mapping = mapping(game_info)
        width, height = image_size_from_seqinfo(seqinfo)
        ball_dict = ball_256(gt_txt = labels_path, class_mapping = class_mapping,
                              width = width, height = height, stride = stride)
        move_images_and_labels_deepball(images_path, ff_path, ball_dict)