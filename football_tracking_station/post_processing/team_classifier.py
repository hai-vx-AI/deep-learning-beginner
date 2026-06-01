from typing import Optional, List, Dict, Tuple
from collections import defaultdict, deque
import numpy as np
import cv2
from sklearn.cluster import KMeans

from football_tracking_station.post_processing.detector import Detection


class TeamClassifier:
    """
    Phân loại cầu thủ vào 2 đội dựa trên màu áo bằng KMeans + vote buffer.

    Cơ chế vote buffer (chống flicker hoàn toàn):
      - Mỗi cầu thủ có 1 deque lưu N lần predict gần nhất
      - Team hiển thị = đa số phiếu trong buffer (majority vote)
      - Không bao giờ clear team_map → không bao giờ flash
      - Kể cả khi predict sai 1-2 lần → buffer hấp thụ noise

    Giai đoạn WARMUP:
      - Thu thập màu áo → train KMeans 1 lần
      - Lưu anchor để detect label swap về sau

    Giai đoạn ACTIVE:
      - Mỗi frame: predict team → đẩy vào vote buffer
      - Team hiển thị = majority vote → ổn định, không flash
    """

    TEAM_COLORS: Dict[int, Tuple[int, int, int]] = {
        0: (0,   0,   255),   # BGR: Đỏ
        1: (255, 255, 255),   # BGR: Trắng
    }

    def __init__(self, n_warmup_colors:   int = 100,
                 max_lost_frames:         int = 30,
                 cleanup_interval:        int = 30,
                 vote_window:             int = 15):
        """
        Args:
            n_warmup_colors : số mẫu cần thu thập trước khi train KMeans
            max_lost_frames : xóa ID không xuất hiện quá N frame
            cleanup_interval: tần suất chạy cleanup (frame)
            vote_window     : số lần predict gần nhất dùng để vote
                              (lớn hơn = ổn định hơn nhưng chậm thích nghi hơn)
        """
        self.n_warmup_colors  = n_warmup_colors
        self.max_lost_frames  = max_lost_frames
        self.cleanup_interval = cleanup_interval
        self.vote_window      = vote_window

        # Recheck riêng từng ID để sửa cầu thủ bị phân loại sai lâu dài.
        # Không dùng để đảo màu toàn bộ 2 đội.
        self.recheck_window = 12
        self.recheck_min_votes = 9

        self.recheck_buffer: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.recheck_window)
        )

        self.kmeans     = KMeans(n_clusters=2, n_init=10, random_state=0)
        self.is_trained = False
        self.anchor:    Dict[int, np.ndarray] = {}

        # Mapping cố định từ nhãn cluster của KMeans sang team hiển thị.
        # Quan trọng: mapping này chỉ được quyết định 1 lần sau warmup,
        # không được đảo qua lại theo từng frame.
        self.cluster_to_team: Dict[int, int] = {0: 0, 1: 1}

        self.collected_colors: List[List[float]]               = []
        # vote_buffer[track_id] = deque([0,1,0,1,1,...], maxlen=vote_window)
        self.vote_buffer:      Dict[int, deque]                = defaultdict(
            lambda: deque(maxlen=self.vote_window)
        )
        # team_map = kết quả majority vote — chỉ dùng để đọc, không clear
        self.team_map:    Dict[int, int]   = {}
        self.color_cache: Dict[int, List[float]] = {}
        self.last_seen:   Dict[int, int]   = {}
        self.frame_count: int              = 0

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, players: List[Detection]) -> None:
        self.frame_count += 1

        if not self.is_trained:
            self._warmup(frame, players)
        else:
            self._predict_and_vote(frame, players)

        if self.frame_count % self.cleanup_interval == 0:
            self._cleanup()

    def get_team(self, track_id: int) -> Optional[int]:
        return self.team_map.get(track_id)

    def get_box_color(self, track_id: int) -> Tuple[int, int, int]:
        team = self.team_map.get(track_id)
        if team is None:
            return (128, 128, 128)
        return self.TEAM_COLORS.get(team, (128, 128, 128))

    # ── WARMUP ────────────────────────────────────────────────────────────────

    def _warmup(self, frame: np.ndarray, players: List[Detection]) -> None:
        for det in players:
            color = _extract_shirt_color(frame, det)
            if color is not None:
                self.collected_colors.append(color)

        if len(self.collected_colors) < self.n_warmup_colors:
            return

        print(f"\n[TeamClassifier] Đủ {len(self.collected_colors)} mẫu — "
              f"train KMeans 1 lần...")
        X = np.array(self.collected_colors)
        self.kmeans.fit(X)

        # Lưu anchor để debug/tham khảo, nhưng KHÔNG dùng để đảo nhãn theo từng frame.
        labels = self.kmeans.labels_
        for team_id in [0, 1]:
            mask = labels == team_id
            self.anchor[team_id] = X[mask].mean(axis=0)

        # Khóa mapping cluster -> team sau warmup.
        # Với video bóng đá thường gặp: đội áo trắng có saturation thấp hơn.
        # COLOR_TEAM trong Visualizer đang để Team 0 = đỏ, Team 1 = trắng,
        # nên map cluster có saturation thấp hơn sang Team 1.
        centers = self.kmeans.cluster_centers_  # HSV feature: [H, S]
        white_cluster = int(np.argmin(centers[:, 1]))
        color_cluster = 1 - white_cluster
        self.cluster_to_team = {
            color_cluster: 0,
            white_cluster: 1,
        }

        self.is_trained = True
        self.collected_colors.clear()
        print(f"[TeamClassifier] Train xong. cluster_to_team khóa cứng: {self.cluster_to_team}")

    # ── PREDICT + VOTE ────────────────────────────────────────────────────────

    def _predict_and_vote(self, frame: np.ndarray,
                          players: List[Detection]) -> None:
        """
        Mỗi frame:
          1. Predict team cho từng cầu thủ
          2. Đẩy kết quả vào vote_buffer
          3. team_map[id] = majority vote của buffer
             → thay đổi chậm, mượt, không flash
        """
        if not players:
            return

        # Batch: extract màu tất cả cầu thủ cùng lúc
        ids_to_predict:    List[int]         = []
        colors_to_predict: List[List[float]] = []

        for det in players:
            self.last_seen[det.track_id] = self.frame_count

            color = _extract_shirt_color(frame, det)
            if color is not None:
                self.color_cache[det.track_id] = color
                ids_to_predict.append(det.track_id)
                colors_to_predict.append(color)
            elif det.track_id in self.color_cache:
                # Dùng màu cache nếu frame này không extract được
                ids_to_predict.append(det.track_id)
                colors_to_predict.append(self.color_cache[det.track_id])

        if not ids_to_predict:
            return

        # Batch predict.
        # KMeans đã fit 1 lần thì nhãn cluster là ổn định trong suốt vòng đời model.
        # Không detect/đảo nhãn global theo từng frame, vì thao tác đó làm 2 đội đổi màu hàng loạt.
        raw_predictions = self.kmeans.predict(np.array(colors_to_predict))

        # Đẩy vào vote buffer và cập nhật team_map
        for idx, track_id in enumerate(ids_to_predict):
            raw_label = int(raw_predictions[idx])
            predicted_team = self.cluster_to_team.get(raw_label, raw_label)

            current_team = self.team_map.get(track_id)

            # Nếu ID chưa có team thì gán bình thường qua vote buffer.
            if current_team is None:
                self.vote_buffer[track_id].append(predicted_team)
            else:
                # Nếu dự đoán trùng team hiện tại, giữ ổn định và reset nghi ngờ.
                if predicted_team == current_team:
                    self.vote_buffer[track_id].append(predicted_team)
                    self.recheck_buffer[track_id].clear()
                else:
                    # Nếu dự đoán khác team hiện tại, chưa đổi ngay.
                    # Lưu vào buffer kiểm tra lại riêng cho ID này.
                    self.recheck_buffer[track_id].append(predicted_team)

                    # Chỉ đổi nếu đủ nhiều frame liên tiếp/nghiêng mạnh về team mới.
                    votes_new_team = sum(
                        1 for t in self.recheck_buffer[track_id]
                        if t == predicted_team
                    )

                    if votes_new_team >= self.recheck_min_votes:
                        # Đổi riêng ID này, không đổi toàn bộ đội.
                        self.vote_buffer[track_id].clear()

                        for _ in range(self.vote_window):
                            self.vote_buffer[track_id].append(predicted_team)

                        self.recheck_buffer[track_id].clear()

            buf = self.vote_buffer[track_id]
            votes_team_1 = sum(buf)
            self.team_map[track_id] = 1 if votes_team_1 > len(buf) / 2 else 0
            
    def _detect_swap(self, X: np.ndarray,
                     predictions: np.ndarray) -> Dict[int, int]:
        """
        Kiểm tra xem KMeans có bị swap label không bằng cách so
        centroid thực tế của batch hiện tại với anchor đã lưu.
        Trả về label_map: {0:0, 1:1} hoặc {0:1, 1:0}.
        """
        # Không đảo nhãn theo từng frame.
        # KMeans.predict() đã dùng centroid cố định sau fit(), nên label không tự swap.
        # Hàm này giữ lại để tránh lỗi nếu nơi khác còn gọi, nhưng luôn trả mapping identity.
        return {0: 0, 1: 1}

    # ── CLEANUP ───────────────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        dead = [
            tid for tid, last in self.last_seen.items()
            if (self.frame_count - last) > self.max_lost_frames
        ]
        for tid in dead:
            self.team_map.pop(tid, None)
            self.color_cache.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.vote_buffer.pop(tid, None)
            self.recheck_buffer.pop(tid, None)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _extract_shirt_color(frame: np.ndarray,
                         det: Detection) -> Optional[List[float]]:
    """
    Trích xuất màu áo từ vùng ngực: 20-50% chiều cao, 30-70% chiều rộng.
    Dùng median để tránh bị kéo bởi outlier (tóc, cổ áo, tay).
    """
    if det.width <= 0 or det.height <= 0:
        return None

    cx1 = det.x1 + int(det.width  * 0.3)
    cx2 = det.x1 + int(det.width  * 0.7)
    cy1 = det.y1 + int(det.height * 0.2)
    cy2 = det.y1 + int(det.height * 0.5)

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    return [float(np.median(pixels[:, 0])),
            float(np.median(pixels[:, 1]))]