from typing import Optional, List, Dict, Tuple
import numpy as np
import cv2

from football_tracking_station.post_processing.detector import Detection, BallPosition


class Visualizer:
    """
    Toàn bộ logic vẽ lên frame — tách hoàn toàn khỏi logic nghiệp vụ.

    Nguyên tắc:
    - Không đọc/ghi bất kỳ state nào ngoài những gì được truyền vào
    - Mọi hàm draw*() đều vẽ trực tiếp lên frame (in-place)
    - Vẽ theo thứ tự: players → ball → possession line → HUD
      (HUD luôn vẽ sau cùng để không bị che)
    """

    # Màu BGR
    COLOR_BALL_YOLO    = (0,   255, 255)   # vàng
    COLOR_BALL_DEEP    = (255, 255, 255)   # trắng
    COLOR_BALL_PREDICT = (0,   165, 255)   # cam — dự đoán từ last_position
    COLOR_KEEP_BALL    = (0,   255, 0)     # xanh lá
    COLOR_POSSESSION   = (0,   255, 255)   # vàng
    COLOR_FPS          = (0,   255, 0)
    COLOR_TEAM = {
        0: (0,   0,   255),   # đỏ
        1: (255, 255, 255),   # trắng
    }
    COLOR_UNCLASSIFIED = (128, 128, 128)   # xám

    def draw_all(self,
                 frame:       np.ndarray,
                 players:     List[Detection],
                 ball_pos:    Optional[BallPosition],
                 team_map:    Dict[int, int],
                 closest_pid: Optional[int],
                 possession:  Dict[int, int],
                 fps:         float,
                 frame_count: int) -> None:
        """
        Entry point duy nhất. Vẽ toàn bộ nội dung lên frame theo đúng thứ tự.
        """
        self.draw_players(frame, players, team_map)
        self.draw_ball(frame, ball_pos)
        if ball_pos is not None and closest_pid is not None:
            self.draw_possession_indicator(frame, players, ball_pos, closest_pid)
        self.draw_hud(frame, possession, fps, frame_count)

    # ── PLAYERS ───────────────────────────────────────────────────────────────

    def draw_players(self, frame: np.ndarray,
                     players: List[Detection],
                     team_map: Dict[int, int]) -> None:
        """Vẽ bbox + track_id cho từng cầu thủ, màu theo team."""
        for det in players:
            team_id   = team_map.get(det.track_id)
            box_color = self.COLOR_TEAM.get(team_id, self.COLOR_UNCLASSIFIED)

            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), box_color, 2)
            cv2.putText(
                frame, f"ID:{det.track_id}",
                (det.x1, det.y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA
            )

    # ── BALL ──────────────────────────────────────────────────────────────────

    def draw_ball(self, frame: np.ndarray,
                  ball_pos: Optional[BallPosition]) -> None:
        """
        Vẽ vị trí bóng với màu khác nhau theo source:
          - yolo     → hình chữ nhật vàng (bbox thô từ YOLO)
          - deepball → hình tròn trắng (đã được tinh chỉnh)
          - predicted → hình tròn cam (dự đoán từ last_position, YOLO mất bóng)
        """
        if ball_pos is None:
            return

        if ball_pos.source == "yolo":
            # Vẽ điểm tâm
            cv2.circle(frame, (ball_pos.x, ball_pos.y), 8,
                       self.COLOR_BALL_YOLO, 2, cv2.LINE_AA)
            cv2.circle(frame, (ball_pos.x, ball_pos.y), 2,
                       self.COLOR_BALL_YOLO, -1)
        elif ball_pos.source == "deepball":
            cv2.circle(frame, (ball_pos.x, ball_pos.y), 10,
                       self.COLOR_BALL_DEEP, 2, cv2.LINE_AA)
            cv2.circle(frame, (ball_pos.x, ball_pos.y), 3,
                       self.COLOR_BALL_DEEP, -1)
        else:   # predicted / fallback
            cv2.circle(frame, (ball_pos.x, ball_pos.y), 8,
                       self.COLOR_BALL_PREDICT, 1, cv2.LINE_AA)

        # Label confidence nhỏ bên cạnh
        label = f"{ball_pos.source[0].upper()} {ball_pos.confidence:.2f}"
        cv2.putText(
            frame, label,
            (ball_pos.x + 12, ball_pos.y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            self.COLOR_BALL_DEEP if ball_pos.source == "deepball" else self.COLOR_BALL_YOLO,
            1, cv2.LINE_AA
        )

    # ── POSSESSION ────────────────────────────────────────────────────────────

    def draw_possession_indicator(self,
                                   frame:       np.ndarray,
                                   players:     List[Detection],
                                   ball_pos:    BallPosition,
                                   closest_pid: int) -> None:
        """
        Vẽ đường nối từ bóng → chân cầu thủ đang giữ bóng + ellipse highlight.
        FIX so với gốc: không cần duyệt lại O(n), dùng dict lookup O(1).
        """
        player_map = {det.track_id: det for det in players}
        det = player_map.get(closest_pid)
        if det is None:
            return

        bx, by = ball_pos.x, ball_pos.y
        fx, fy = det.foot_x, det.foot_y

        # Đường nối bóng → chân
        cv2.line(frame, (bx, by), (fx, fy), self.COLOR_KEEP_BALL, 2, cv2.LINE_AA)

        # Ellipse highlight dưới chân
        cv2.ellipse(frame, (fx, fy), (40, 15), 0, 0, 360,
                    self.COLOR_POSSESSION, 3)
        cv2.ellipse(frame, (fx, fy), (48, 18), 0, 0, 360,
                    self.COLOR_POSSESSION, 1)

        cv2.putText(
            frame, "Keep ball",
            (fx + 30, fy - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_POSSESSION, 2, cv2.LINE_AA
        )

    # ── HUD ───────────────────────────────────────────────────────────────────

    def draw_hud(self, frame: np.ndarray,
                 possession: Dict[int, int],
                 fps: float,
                 frame_count: int) -> None:
        """
        Vẽ HUD góc trên-trái:
          Line 1: Team 0 possession
          Line 2: Team 1 possession
          Line 3: FPS
          Line 4: Frame count
        Vẽ duy nhất 1 lần (không bị nhân đôi như code gốc).
        """
        t0 = possession.get(0, 0)
        t1 = possession.get(1, 0)

        lines = [
            (f"Team 0: {t0} frames", self.COLOR_TEAM[0]),
            (f"Team 1: {t1} frames", self.COLOR_TEAM[1]),
            (f"FPS: {fps:.1f}",      self.COLOR_FPS),
            (f"Frame: {frame_count}", (200, 200, 200)),
        ]
        for i, (text, color) in enumerate(lines):
            cv2.putText(
                frame, text,
                (20, 40 + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA
            )