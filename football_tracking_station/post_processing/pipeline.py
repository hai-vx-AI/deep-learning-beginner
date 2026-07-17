import time
import cv2
import numpy as np
from queue import Queue, Full
from threading import Thread
from typing import Callable, Dict, Optional

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

    Screen Capture mới:
      - Không gửi processed frame về QLabel preview nữa.
      - Chỉ gửi overlay_data nhẹ gồm bbox/ball/team/fps về UI thread.
      - OverlayWindow trong suốt sẽ vẽ trực tiếp lên màn hình laptop.

    Nguyên tắc tốc độ:
      - UI queue đầy -> thay frame cũ bằng frame mới, không block AI thread.
      - File writer queue đầy -> drop output frame, không block AI thread.
      - Screen mode mặc định không ghi output để giữ FPS ổn định.
    """

    def __init__(self,
                 yolo_weight:      str,
                 trt_weight:       str,
                 dis_ball_player:  int        = 100,
                 n_warmup_colors:  int        = 100,
                 deepball_thresh:  float      = 0.5,
                 report_interval:  int        = 100,
                 vote_window:      int        = 15,
                 ui_queue:         Queue      = None,
                 stats_callback:   Optional[Callable[[dict], None]] = None,
                 overlay_callback: Optional[Callable[[dict], None]] = None):

        self.dis_ball_player  = dis_ball_player
        self.report_interval  = report_interval
        self.ui_queue         = ui_queue
        self.stats_callback   = stats_callback
        self.overlay_callback = overlay_callback

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

    def process(self, video_path: str, output_path: str = "output.mp4",
                save_output: bool = False) -> None:
        """Video file mode: draw into preview frame as before."""
        source = FileFrameSource(video_path)
        self.process_source(
            source=source,
            output_path=output_path,
            save_output=save_output,
            source_name=video_path,
            use_overlay=False,
        )

    def process_screen(self,
                       region=None,
                       fps: int = 30,
                       output_path: str = "screen_output.mp4",
                       save_output: bool = False) -> None:
        """Realtime screen capture mode: send overlay data to transparent window."""
        source = ScreenFrameSource(region=region, fps=fps)
        self.process_source(
            source=source,
            output_path=output_path,
            save_output=save_output,
            source_name=f"screen region={region}",
            use_overlay=True,
        )

    def process_source(self,
                       source,
                       output_path: str = "output.mp4",
                       save_output: bool = True,
                       source_name: str = "source",
                       use_overlay: bool = False) -> None:
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
        last_stats_emit = start_time
        last_overlay_emit = start_time

        print(f"Bắt đầu: {source_name} [{frame_w}×{frame_h} @ {fps_src:.0f}fps]")

        try:
            while not self._stop_flag:
                self.profiler.start("1. Read frame")
                ok, frame = source.read()
                self.profiler.stop("1. Read frame")
                if not ok or frame is None:
                    # Chỉ realtime source mới được chờ frame tiếp; file video thì kết thúc.
                    if getattr(source, "is_realtime", False):
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

                elapsed = time.perf_counter() - start_time
                self._current_fps = self.frame_count / elapsed if elapsed > 0 else 0.0

                self.profiler.start("6. Overlay/Draw")

                # Screen mode: gửi dữ liệu nhẹ để overlay vẽ trực tiếp lên màn hình.
                now = time.perf_counter()
                if use_overlay and self.overlay_callback is not None:
                    # Không cần spam Qt signal quá dày; 30 Hz là đủ cho overlay.
                    if now - last_overlay_emit >= (1.0 / 30.0):
                        self.overlay_callback(
                            self._make_overlay_data(
                                frame_w=frame_w,
                                frame_h=frame_h,
                                players=players,
                                ball_pos=ball_pos,
                                closest_pid=closest_pid,
                            )
                        )
                        last_overlay_emit = now

                # Video mode vẫn preview frame đã vẽ trong UI.
                # Screen mode chỉ vẽ vào frame nếu người dùng bật save_output.
                should_draw_frame = (not use_overlay) or save_output
                if should_draw_frame:
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

                self.profiler.stop("6. Overlay/Draw")

                # Preview UI chỉ dùng cho Video File. Screen Capture không preview
                # để tránh capture ngược chính UI chính.
                if (not use_overlay) and self.ui_queue is not None:
                    self._push_latest_preview(frame)

                # Ghi file: không được block AI thread.
                if save_output and file_queue is not None:
                    try:
                        file_queue.put_nowait(frame.copy())
                    except Full:
                        pass

                if self.stats_callback is not None and now - last_stats_emit >= 0.5:
                    self.stats_callback(self.get_stats())
                    last_stats_emit = now

                if self.report_interval > 0 and self.frame_count % self.report_interval == 0:
                    self.profiler.report()

        finally:
            # Clear overlay visually before the UI closes it.
            if use_overlay and self.overlay_callback is not None:
                try:
                    self.overlay_callback({"players": [], "ball": None, "closest_pid": None})
                except Exception:
                    pass

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

    def _make_overlay_data(self, frame_w, frame_h, players, ball_pos, closest_pid) -> dict:
        payload_players = []
        for det in players:
            payload_players.append({
                "track_id": int(det.track_id),
                "class_id": int(det.class_id),
                "x1": int(det.x1),
                "y1": int(det.y1),
                "x2": int(det.x2),
                "y2": int(det.y2),
                "foot_x": int(det.foot_x),
                "foot_y": int(det.foot_y),
                "team": self.team_clf.get_team(det.track_id),
                "conf": float(det.conf),
            })

        payload_ball = None
        if ball_pos is not None:
            payload_ball = {
                "x": int(ball_pos.x),
                "y": int(ball_pos.y),
                "source": str(ball_pos.source),
                "confidence": float(ball_pos.confidence),
            }

        stats = self.get_stats()
        return {
            "frame_w": int(frame_w),
            "frame_h": int(frame_h),
            "players": payload_players,
            "ball": payload_ball,
            "closest_pid": int(closest_pid) if closest_pid is not None else None,
            "possession_pct": stats.get("possession_pct", {0: 50, 1: 50}),
            "fps": float(self._current_fps),
            "frame_count": int(self.frame_count),
        }

    def _push_latest_preview(self, frame) -> None:
        try:
            self.ui_queue.put_nowait(frame)
        except Full:
            try:
                self.ui_queue.get_nowait()
            except Exception:
                pass
            try:
                self.ui_queue.put_nowait(frame)
            except Full:
                pass

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
    processor.process(video_path, output_path, save_output=True)
