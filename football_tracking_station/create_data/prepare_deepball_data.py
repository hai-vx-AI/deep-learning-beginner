from pathlib import Path
from tqdm import tqdm

from football_tracking_station.utils.utils import mapping, image_size_from_seqinfo, ball_256, move_images_and_labels_deepball


def prepare_deepball(input_path: str, output_path: str,
                     is_train: bool = True, stride: int = 1):
    """
    Chuẩn bị dữ liệu DeepBall từ SoccerNet tracking format.

    Args:
        input_path:  Thư mục gốc nguồn (chứa train/ và test/).
        output_path: Thư mục output.
        is_train:    True để xử lý tập train, False để xử lý tập test.
        stride:      Chỉ lấy 1 frame mỗi `stride` frame.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output      = output_path / ("train" if is_train else "test")
    output.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Lỗi đường dẫn {input_path}. Đã dừng chạy chương trình")
        return

    data_path = input_path / ("train" if is_train else "test")
    if not data_path.exists():
        print(f"Lỗi đường dẫn {data_path}. Đã dừng chạy chương trình")
        return

    data_dir = [a for a in data_path.iterdir() if a.is_dir()]
    if not data_dir:
        print(f"Không tìm thấy thư mục con trong {data_path}")
        return

    for ff in tqdm(data_dir, desc="Tiến trình: ", unit="video"):
        ff_path    = output / ff.name
        cre_images = ff_path / "images"
        cre_labels = ff_path / "labels"
        ff_path.mkdir(parents=True, exist_ok=True)
        cre_images.mkdir(parents=True, exist_ok=True)
        cre_labels.mkdir(parents=True, exist_ok=True)

        images_path = ff / "img1"
        labels_path = ff / "gt" / "gt.txt"
        game_info   = ff / "gameinfo.ini"
        seqinfo     = ff / "seqinfo.ini"

        class_mapping  = mapping(game_info)
        width, height  = image_size_from_seqinfo(seqinfo)
        ball_dict      = ball_256(gt_txt=labels_path, class_mapping=class_mapping,
                                  width=width, height=height, stride=stride)
        move_images_and_labels_deepball(images_path, ff_path, ball_dict)