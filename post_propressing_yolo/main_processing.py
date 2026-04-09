from core.deepball_arkitecture import DeepBall
from utils import predict_image_deepball

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
import time
from collections import defaultdict
from queue import Queue
from threading import Thread
from utils import predict_deepball_trt
import tensorrt as trt

class CodeProfiler:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self.start_times = {}

    def start(self, name):
        self.start_times[name] = time.time()

    def stop(self, name):
        if name in self.start_times:
            elapsed_ms = (time.time() - self.start_times[name]) * 1000
            self.times[name] += elapsed_ms
            self.counts[name] += 1

    def report(self):
        print("\n" + "="*50)
        print(" BÁO CÁO HIỆU NĂNG (Trung bình ms / frame)")
        print(" (Mục tiêu để đạt 30 FPS: Tổng thời gian < 33.3 ms)")
        print("="*50)
        total_time = 0
        for name in self.times:
            avg_time = self.times[name] / self.counts[name]
            total_time += avg_time
            print(f"{name:<25}: {avg_time:>6.2f} ms")
        print("-" * 50)
        print(f"{'TỔNG CỘNG':<25}: {total_time:>6.2f} ms")
        print("="*50 + "\n")

def get_shirt_color(frame, x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    chest_y1 = y1 + int(h * 0.2)
    chest_y2 = y1 + int(h * 0.5)
    chest_x1 = x1 + int(w * 0.3)
    chest_x2 = x1 + int(w * 0.7)

    chest_crop = frame[chest_y1:chest_y2, chest_x1:chest_x2]
    if chest_crop.size == 0:
        return None

    hsv_crop = cv2.cvtColor(chest_crop, cv2.COLOR_BGR2HSV)
    mean_color = cv2.mean(hsv_crop)[:2]
    return mean_color

def euclid_between_ball_players(euclid_players, euclid_ball):
    if not euclid_players or not euclid_ball:
        return None, -1, -1
    players_arr = np.array(euclid_players)
    ball_pos = np.array([euclid_ball[0], euclid_ball[1]])
    distances = np.linalg.norm(players_arr[:,:2] - ball_pos, axis = 1)
    min_idx = np.argmin(distances)
    return distances[min_idx], int(players_arr[min_idx, 2]), euclid_ball[2]

def processing_yolo(video_path, yolo_weight, deepball_weight, num_frame = 100, dis_ball_player = 100, down_ratio = 4):
    if not Path(video_path).is_file():
        print(f"--Lỗi đường dẫn video: {video_path}--")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLO(yolo_weight)

    print("Đang khởi động động cơ TensorRT cho DeepBall...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(deepball_weight, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        deepball_engine = runtime.deserialize_cuda_engine(f.read())
    deepball_context = deepball_engine.create_execution_context()

    d_input = torch.empty((1, 3, 256, 256), dtype=torch.float32, device="cuda")
    d_output = torch.empty((1, 1, 256, 256), dtype=torch.float32, device="cuda")
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"--Không thể mở video: {video_path}--")
        return

    kmeans = KMeans(n_clusters = 2, n_init = 10, random_state = 0)
    is_kmeans_trained = False
    collected_colors = []

    frame_count = 0
    team_0 = 0
    team_1 = 0

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
    save_ball = []

    team_cache = {}
    color_cache = {}
    id_last_seen = {}
    MAX_LOST_FRAMES = 75

    start_time = time.time()
    profiler = CodeProfiler()
    frame_queue = Queue(maxsize=500)

    def video_writer_worker():
        while True:
            frame_to_write = frame_queue.get()
            if frame_to_write is None: 
                break
            output_video.write(frame_to_write)
            frame_queue.task_done()

    writer_thread = Thread(target=video_writer_worker, daemon=True)
    writer_thread.start()

    try:
        while True:
            profiler.start("1. Đọc Frame (cv2.read)")
            flag, frame = video.read()
            profiler.stop("1. Đọc Frame (cv2.read)")
            if not flag:
                print("\n--Kết thúc video--")
                break

            frame_count += 1
            profiler.start("2. YOLO Tracking (Engine)")
            results = model.track(
                frame,
                tracker = "bytetrack.yaml",
                conf = 0.25,
                iou = 0.5,
                persist = True,
                verbose = False,
                half = True
            )
            profiler.stop("2. YOLO Tracking (Engine)")
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                xyxy_array = boxes.xyxy.cpu().numpy()
                id_array = boxes.id.cpu().numpy()
                class_array = boxes.cls.cpu().numpy()
                conf_array = boxes.conf.cpu().numpy()

                current_players = []
                euclid_players = []
                euclid_balls = []
                confident_ball = -1
                best_ball_bbox = None

                profiler.start("3. Logic Cầu Thủ & DeepBall")
                for i in range(len(id_array)):
                    x1, y1, x2, y2 = xyxy_array[i]
                    track_id = int(id_array[i])
                    class_id = int(class_array[i])

                    if class_id == 1:
                        x_pcenter = int((x1 + x2) / 2)
                        y_pdown = int(y2)
                        euclid_players.append([x_pcenter, y_pdown, track_id])

                        if not is_kmeans_trained:
                            color = get_shirt_color(frame, x1, y1, x2, y2)
                            if color is not None:
                                collected_colors.append(color)
                                current_players.append((track_id, color, int(x1), int(y1), int(x2), int(y2)))
                            else:
                                current_players.append((track_id, None, int(x1), int(y1), int(x2), int(y2)))
                        else:
                            if track_id not in team_cache:
                                color = get_shirt_color(frame, x1, y1, x2, y2)
                                if color is not None:
                                  color_cache[track_id] = color
                                current_players.append((track_id, color, int(x1), int(y1), int(x2), int(y2)))
                            else:
                                cached_color = color_cache.get(track_id)
                                current_players.append((track_id, cached_color, int(x1), int(y1), int(x2), int(y2)))

                    if class_id == 0 and conf_array[i] > confident_ball:
                        confident_ball = conf_array[i]
                        x_center = int((x1 + x2) / 2)
                        y_center = int((y1 + y2) / 2)
                        euclid_balls = [x_center, y_center, class_id]
                        save_ball = [x_center, y_center]
                        best_ball_bbox = (int(x1), int(y1), int(x2), int(y2))

                if not is_kmeans_trained and len(collected_colors) > num_frame:
                    print(f"\nĐã gom đủ {len(collected_colors)} áo. Bắt đầu huấn luyện Kmeans...")
                    kmeans.fit(collected_colors)
                    is_kmeans_trained = True
                    print("Đã huấn luyện xong!")

                if is_kmeans_trained:
                    team_player_dict = {}
                    new_ids = []
                    new_colors = []

                    for player in current_players:
                        p_id, p_color, px1, py1, px2, py2 = player
                        if p_id not in team_cache:
                            if p_color is not None:
                                new_ids.append(p_id)
                                new_colors.append(p_color)

                    if len(new_ids) > 0:
                        color_batch = np.array(new_colors)
                        team_predictions = kmeans.predict(color_batch)
                        for idx, p_id in enumerate(new_ids):
                            team_cache[p_id] = team_predictions[idx]

                    for player in current_players:
                        p_id, _, px1, py1, px2, py2 = player
                        if p_id in team_cache:
                            team_id = team_cache[p_id]
                            team_player_dict[p_id] = team_id
                            id_last_seen[p_id] = frame_count

                            box_color = (0, 0, 255) if team_id == 0 else (255, 255, 255)
                            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
                            cv2.putText(frame, f"ID: {p_id}", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    if best_ball_bbox is not None:
                        bx1, by1, bx2, by2 = best_ball_bbox
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                    else:
                        if len(save_ball) == 2:
                            x_deepball, y_deepball = predict_deepball_trt(
                                frame=frame,
                                x_center=save_ball[0],
                                y_center=save_ball[1],
                                width=width,
                                height=height,
                                d_input=d_input,              # Truyền thêm vùng nhớ vào
                                d_output=d_output,            # Truyền thêm vùng nhớ vào
                                deepball_context=deepball_context, # Truyền thêm động cơ vào
                                threshold=0.5
                            )

                            if x_deepball is not None:
                                cv2.circle(frame, (x_deepball, y_deepball), radius = 10, color = (255, 255, 255), thickness = 2)
                                euclid_balls = [x_deepball, y_deepball, 0]
                                save_ball = [x_deepball, y_deepball]

                    distance, player_closest, ball_closest = euclid_between_ball_players(euclid_players, euclid_balls)
                    team_keep_ball = -1
                    if distance is not None and distance < dis_ball_player:
                        if player_closest in team_player_dict:
                            ball_x, ball_y = euclid_balls[0], euclid_balls[1]

                            for p in euclid_players:
                                if p[2] == player_closest:
                                    foot_x, foot_y = p[0], p[1]
                                    cv2.line(frame, (ball_x, ball_y), (foot_x, foot_y), (0, 255, 0), 2)
                                    cv2.ellipse(frame, center = (foot_x, foot_y), axes = (40, 15), angle = 0, startAngle = 0, endAngle = 360, color = (255, 255, 0), thickness = 3)
                                    cv2.ellipse(frame, center = (foot_x, foot_y), axes = (48, 18), angle = 0, startAngle = 0, endAngle = 360, color = (255, 255, 0), thickness = 1)
                                    cv2.putText(frame, "Keep ball", (foot_x + 30, foot_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                                    break
                            team_keep_ball = team_player_dict[player_closest]

                            if team_keep_ball == 0:
                                team_0 += 1
                            else:
                                team_1 += 1
                profiler.stop("3. Logic Cầu Thủ & DeepBall")

            profiler.start("4. Ghi Video (cv2.write)")
            cv2.putText(frame, f"Team 0: {team_0} frames", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, f"Team 1: {team_1} frames", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            print(f"\rĐang xử lý frame {frame_count} | Tốc độ: {current_fps:.1f} FPS", end="")

            if frame_count % 30 == 0:
                dead_ids = [p_id for p_id, last_frame in id_last_seen.items() if (frame_count - last_frame) > MAX_LOST_FRAMES]
                for p_id in dead_ids:
                    team_cache.pop(p_id, None)
                    id_last_seen.pop(p_id, None)

            profiler.start("4. Đẩy Frame & Cập nhật UI")
            cv2.putText(frame, f"Team 0: {team_0} frames", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, f"Team 1: {team_1} frames", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            print(f"\rĐang xử lý frame {frame_count} | Tốc độ: {current_fps:.1f} FPS", end="")

            if frame_count % 30 == 0:
                dead_ids = [p_id for p_id, last_frame in id_last_seen.items() if (frame_count - last_frame) > MAX_LOST_FRAMES]
                for p_id in dead_ids:
                    team_cache.pop(p_id, None)
                    id_last_seen.pop(p_id, None)
            if not frame_queue.full():
                frame_queue.put(frame.copy())
            
            profiler.stop("4. Đẩy Frame & Cập nhật UI")

            if frame_count % 100 == 0:
                profiler.report()

    except KeyboardInterrupt:
        print("\n\n--Đã dừng tiến trình thủ công--")

    finally:
        if 'video' in locals() and video.isOpened():
            video.release()
            
        print("\nĐang chờ luồng ngầm xả nốt hàng đợi ghi ra ổ cứng...")
        frame_queue.put(None)
        if 'writer_thread' in locals():
            writer_thread.join()
            
        if 'output_video' in locals():
            output_video.release()
            
        cv2.destroyAllWindows()
        print(f"Đã đóng gói an toàn. Tổng số frame: {frame_count}")