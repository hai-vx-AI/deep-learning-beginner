import time
import cv2
import numpy as np
from queue import Queue, Full
from threading import Thread
from typing import Dict

from football_tracking_station.post_processing.detector        import YoloDetector, BallPosition
from football_tracking_station.post_processing.tracker         import DeepBallTracker
from football_tracking_station.post_processing.team_classifier import TeamClassifier
from football_tracking_station.post_processing.visualizer      import Visualizer
from football_tracking_station.post_processing.profiler        import CodeProfiler
from football_tracking_station.post_processing.frame_source    import FileFrameSource, ScreenFrameSource


class VideoProcessor:
    """
    Điều phối pipeline inference.

    Nguồn frame có thể là:
      - video file
      - screen capture region

    Nguyên tắc tốc độ:
      - UI queue đầy -> drop preview frame, không block AI thread.
      - File writer queue đầy -> drop output frame, không block AI thread.
      - Screen mode mặc định không ghi output để giữ FPS ổn định.
    """

    def __init__(self,
                 yolo_weight:     str,
                 trt_weight:      str,
                 dis_ball_player: int        = 100,
                 n_warmup_colors: int        = 100,
                 deepball_thresh: float      = 0.5,
                 report_interval: int        = 100,
                 vote_window:     int        = 15,
                 ui_queue:        Queue      = None):

        self.dis_ball_player = dis_ball_player
        self.report_interval = report_interval
        self.ui_queue        = ui_queue

        self.detector   = YoloDetector(yolo_weight)
        self.tracker    = DeepBallTracker(trt_weight, threshold=deepball_thresh)
        self.team_clf   = TeamClassifier(
            n_warmup_colors=n_warmup_colors,
            vote_window=vote_window,
        )
        self.visualizer = Visualizer()
        self.profiler   = CodeProfiler()

        self.possession:    Dict[int, int] = {0: 0, 1: 0}
        self.frame_count:   int            = 0
        self._current_fps:  float          = 0.0
        self._stop_flag:    bool           = False

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_flag = True

    def get_stats(self) -> dict:
        total = sum(self.possession.values()) or 1
        return {
            "possession":  self.possession.copy(),
            "possession_pct": {
                0: round(self.possession.get(0, 0) / total * 100),
                1: round(self.possession.get(1, 0) / total * 100),
            },
            "frame_count": self.frame_count,
            "fps":         round(self._current_fps, 1),
        }

    def process(self, video_path: str, output_path: str = "output.mp4") -> None:
        """Video file mode."""
        source = FileFrameSource(video_path)
        self.process_source(
            source=source,
            output_path=output_path,
            save_output=True,
            source_name=video_path,
        )

    def process_screen(self,
                       region=None,
                       fps: int = 30,
                       output_path: str = "screen_output.mp4",
                       save_output: bool = False) -> None:
        """Realtime screen capture mode."""
        source = ScreenFrameSource(region=region, fps=fps)
        self.process_source(
            source=source,
            output_path=output_path,
            save_output=save_output,
            source_name=f"screen region={region}",
        )

    def process_source(self,
                       source,
                       output_path: str = "output.mp4",
                       save_output: bool = True,
                       source_name: str = "source") -> None:
        frame_w = int(source.width)
        frame_h = int(source.height)
        fps_src = float(source.fps or 30.0)
        total   = int(getattr(source, "total", 0) or 0)

        writer = None
        file_queue = None
        writer_thread = None

        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_src, (frame_w, frame_h))
            file_queue = Queue(maxsize=4)
            writer_thread = Thread(
                target=self._writer_worker,
                args=(writer, file_queue),
                daemon=True,
            )
            writer_thread.start()

        self._stop_flag  = False
        self.frame_count = 0
        self.profiler.reset()
        start_time = time.perf_counter()

        print(f"Bắt đầu: {source_name} [{frame_w}×{frame_h} @ {fps_src:.0f}fps]")

        try:
            while not self._stop_flag:
                self.profiler.start("1. Read frame")
                ok, frame = source.read()
                self.profiler.stop("1. Read frame")
                if not ok or frame is None:
                    # Screen mode có thể thiếu frame tạm thời; không kết thúc luôn.
                    if total == 0:
                        time.sleep(0.001)
                        continue
                    break

                self.frame_count += 1

                self.profiler.start("2. YOLO")
                detections = self.detector.run(frame)
                players    = YoloDetector.get_players(detections)
                yolo_ball  = YoloDetector.best_ball(detections)
                self.profiler.stop("2. YOLO")

                self.profiler.start("3. DeepBall")
                ball_pos = self.tracker.update(frame, yolo_ball, frame_w, frame_h)
                if ball_pos is None and yolo_ball is not None:
                    ball_pos = BallPosition(
                        x=yolo_ball.cx,
                        y=yolo_ball.cy,
                        source="yolo",
                        confidence=float(yolo_ball.conf),
                    )
                self.profiler.stop("3. DeepBall")

                self.profiler.start("4. Team")
                self.team_clf.update(frame, players)
                self.profiler.stop("4. Team")

                self.profiler.start("5. Possession")
                closest_pid = self._find_closest_player(players, ball_pos)
                if closest_pid is not None:
                    team = self.team_clf.get_team(closest_pid)
                    if team is not None:
                        self.possession[team] = self.possession.get(team, 0) + 1
                self.profiler.stop("5. Possession")

                self.profiler.start("6. Draw")
                elapsed = time.perf_counter() - start_time
                self._current_fps = self.frame_count / elapsed if elapsed > 0 else 0.0

                self.visualizer.draw_all(
                    frame=frame,
                    players=players,
                    ball_pos=ball_pos,
                    team_map=self.team_clf.team_map,
                    closest_pid=closest_pid,
                    possession=self.possession,
                    fps=self._current_fps,
                    frame_count=self.frame_count,
                )
                self.profiler.stop("6. Draw")

                # Preview UI: không được block AI thread.
                if self.ui_queue is not None:
                    try:
                        self.ui_queue.put_nowait(frame)
                    except Full:
                        pass

                # Ghi file: không được block AI thread.
                if save_output and file_queue is not None:
                    try:
                        # Writer chạy thread khác; copy để tránh buffer frame bị thay đổi.
                        file_queue.put_nowait(frame.copy())
                    except Full:
                        pass

                if self.report_interval > 0 and self.frame_count % self.report_interval == 0:
                    self.profiler.report()

        finally:
            source.release()
            if save_output and file_queue is not None:
                file_queue.put(None)
                file_queue.join()
                if writer_thread is not None:
                    writer_thread.join(timeout=2.0)
                if writer is not None:
                    writer.release()
            cv2.destroyAllWindows()
            print(f"Xong. {self.frame_count}/{total or '?'} frames")
            self.profiler.report()

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _find_closest_player(self, players, ball_pos):
        if ball_pos is None or not players:
            return None
        ball_arr  = np.array([ball_pos.x, ball_pos.y])
        best_dist = float("inf")
        best_id   = None
        for det in players:
            dist = float(np.linalg.norm(
                np.array([det.foot_x, det.foot_y]) - ball_arr
            ))
            if dist < best_dist:
                best_dist = dist
                best_id   = det.track_id
        return best_id if best_dist <= self.dis_ball_player else None

    @staticmethod
    def _writer_worker(writer, queue):
        while True:
            frame = queue.get()
            if frame is None:
                queue.task_done()
                break
            writer.write(frame)
            queue.task_done()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def processing_yolo(video_path, yolo_weight, deepball_weight,
                    output_path="output.mp4", dis_ball_player=100,
                    n_warmup_colors=100, deepball_thresh=0.5):
    processor = VideoProcessor(
        yolo_weight=yolo_weight,
        trt_weight=deepball_weight,
        dis_ball_player=dis_ball_player,
        n_warmup_colors=n_warmup_colors,
        deepball_thresh=deepball_thresh,
    )
    processor.process(video_path, output_path)
