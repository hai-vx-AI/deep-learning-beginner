from football_tracking_station.create_data.prepare_data import pre_data
from football_tracking_station.create_data.prepare_deepball_data import prepare_deepball
from football_tracking_station.create_data.create_yaml import create_yaml
from football_tracking_station.core.train_deepball import train_deepball
from football_tracking_station.yolo.yolo_model import train_model
from football_tracking_station.post_processing.pipeline import processing_yolo

# Module xử lý video — chưa cập nhật cho kiến trúc 9-channel mới
# from post_propressing_yolo.main_processing import processing_yolo


if __name__ == "__main__":

    # ── BƯỚC 1: Chuẩn bị dữ liệu YOLO ──────────────────────────────────────
    # source = "SoccerNet/tracking-2023"
    # for is_train in [True, False]:
    #     pre_data(root=source, path="data", is_train=is_train, stride=5)

    # ── BƯỚC 2: Tạo YAML cho YOLO ───────────────────────────────────────────
    # create_yaml(dataset_root="data", yaml_output_path="yolov8.yaml")

    # ── BƯỚC 3: Chuẩn bị dữ liệu DeepBall ──────────────────────────────────
    # source = "SoccerNet/tracking-2023"
    # for is_train in [True, False]:
    #     prepare_deepball(input_path=source, output_path="deepball_data",
    #                      is_train=is_train, stride=1)

    # ── BƯỚC 4: Xử lý video (chưa cập nhật) ────────────────────────────────
    # TODO: Cập nhật processing_yolo để truyền 3 frame liên tiếp vào DeepBall
    #       (stacked 9-channel input thay vì 3-channel cũ)
    # video_path       = "video.mp4"
    # yolo_weight      = "weights/yolo_best.pt"
    # deepball_weight  = "weights/deepball_best.pt"
    # processing_yolo(video_path, yolo_weight=yolo_weight, deepball_weight=deepball_weight)

    # ── BƯỚC 5: Train DeepBall ───────────────────────────────────────────────
    # train_deepball(
    #     root_path     = "deepball_data",
    #     weight_path   = "weights",
    #     epochs        = 50,
    #     batch         = 2,
    #     learning_rate = 1e-3,
    # )

    # train_model(yaml_path="yolov8.yaml", pretrain="")


    # MAIN PROGRAM
    video_path = "local_video.mp4"
    yolo_weight = "football_tracking_station/weights/best.pt"
    deepball_weight = "football_tracking_station/weights/best_deepball_1.pt"
    processing_yolo(video_path, yolo_weight=yolo_weight, deepball_weight=deepball_weight, n_warmup_colors=100)