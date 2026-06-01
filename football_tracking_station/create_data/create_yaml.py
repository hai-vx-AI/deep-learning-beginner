import yaml
from pathlib import Path


def create_yaml(dataset_root: str, yaml_output_path: str):
    """
    Tạo file YAML cho YOLOv8 từ cấu trúc thư mục dataset.

    Cấu trúc thư mục mong đợi:
        dataset_root/
            train/
                <seq_name>/
                    images/
            test/
                <seq_name>/
                    images/
    """
    root_path  = Path(dataset_root).resolve()
    train_path = root_path / "train"
    val_path   = root_path / "test"

    if not root_path.exists():
        print(f"Dataset không tồn tại: {root_path}")
        return

    train_dirs = [
        f"train/{seq.name}/images"
        for seq in sorted(train_path.iterdir())
        if seq.is_dir() and (seq / "images").exists()
    ]
    val_dirs = [
        f"test/{seq.name}/images"
        for seq in sorted(val_path.iterdir())
        if seq.is_dir() and (seq / "images").exists()
    ]

    if not train_dirs:
        print("Không tìm thấy thư mục video hợp lệ trong train/")
        return

    yaml_data = {
        "path":  root_path.as_posix(),
        "train": train_dirs,
        "val":   val_dirs,
        "names": {0: "ball", 1: "player", 2: "referee", 3: "goalkeeper"},
    }

    Path(yaml_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
    print(f"Đã tạo file YAML thành công: {yaml_output_path}")


if __name__ == "__main__":
    create_yaml(
        dataset_root="data",
        yaml_output_path="yolov8.yaml",
    )