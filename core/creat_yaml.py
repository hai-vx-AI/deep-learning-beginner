import yaml
from pathlib import Path

dataset_root = r"data"
yaml_output_path = r"D:\.vscode\football_distance\core\yolov8.yaml"

def multi_yaml(dataset_root, yaml_output_path):
    root_path = Path(dataset_root).resolve()
    train = root_path / "train"
    val = root_path / "test"

    if not root_path.exists():
        print(f"Dataset không tồn tại: {root_path}")
        return
    train_path = []
    for sequence in train.iterdir():
        if sequence.is_dir() and (sequence / "images").exists():
            train_path.append(f"train/{sequence.name}/images")
    val_path = []
    for sequence in val.iterdir():
        if sequence.is_dir() and (sequence / "images").exists():
            val_path.append(f"test/{sequence.name}/images")

    if not train_path:
        print("Không tìm thấy thư mục video hợp lệ")
        return
    yaml_data = {
        "path": root_path.as_posix(),
        "train": train_path,
        "val": val_path,
        "names": {
            0: "ball",
            1: "player",
            2: "referee",
            3: "goalkeeper"
        }
    }
    with open(yaml_output_path, "w", encoding = "utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys = False, default_flow_style = False)
    print("Đã tạo file YAML thành công")
    
if __name__ == "__main__":
    multi_yaml(dataset_root, yaml_output_path)