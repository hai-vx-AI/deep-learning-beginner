from pathlib import Path
from create_data.prepare_data import pre_data
from create_data.prepare_deepball_data import prepare_deepball
from post_propressing_yolo.main_processing import processing_yolo
from core.train_deepball import train_deepball


if __name__ == "__main__":
    # root = r"SoccerNet/tracking-2023"
    # path = Path("data")
    # is_train = False
    # path = path / ("train" if is_train else "test")
    # stride = 5
    # pre_data(root, path, is_train, stride)

    video_path = r"D:\.vscode\football_distance\SNMOT-120.mp4"
    yolo_weight =  r"D:\.vscode\football_distance\best_yolo_new.pt"
    deepball_weight = r"D:\.vscode\football_distance\best (1).pt"

    processing_yolo(video_path, yolo_weight = yolo_weight, deepball_weight = deepball_weight)

    # source_data = r"D:\.vscode\football_distance\SoccerNet\tracking-2023"
    # output_data = "deepball_data"
    # is_train = True
    # prepare_deepball(input_path = source_data, output_path = output_data, is_train = is_train, stride = 5)


    # root = "deepball_data"
    # weight_path = r"D:\.vscode\football_distance\weight"
    # epochs = 50
    # batch = 2
    # learning_rate = 1e-4
    # train_deepball(root_path = root, weight_path = weight_path, epochs = epochs,
    #                 batch = batch, learning_rate = learning_rate)