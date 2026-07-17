"""
ui/screen_overlay.py — Transparent overlay window for Screen Capture mode.

This window draws detections directly over the captured screen region.
It does NOT display the captured frame; it only paints lightweight shapes
(bbox, ball, possession indicator, HUD) on a transparent always-on-top window.
"""

from __future__ import annotations

import sys
import ctypes
from typing import Dict, Any, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget


Region = Tuple[int, int, int, int]  # left, top, right, bottom

class ScreenOverlay(QWidget):
    """Transparent click-through overlay aligned with the capture region."""

    COLOR_TEAM_0 = QColor(255, 40, 70)       # red-ish
    COLOR_TEAM_1 = QColor(255, 255, 255)     # white
    COLOR_UNKNOWN = QColor(150, 150, 150)
    COLOR_BALL = QColor(0, 255, 255) #cyan
    COLOR_BALL_DEEP = QColor(255, 255, 255) #white
    COLOR_POSSESSION = QColor(0, 255, 80) #green
    COLOR_HUD = QColor(0, 212, 255) #blue
    COLOR_SHADOW = QColor(0, 0, 0)

    def __init__(self, region: Region, parent=None):
        super().__init__(parent)
        self.region = tuple(int(v) for v in region)
        self.overlay_data: Dict[str, Any] = {
            "frame_w": max(1, self.region[2] - self.region[0]),
            "frame_h": max(1, self.region[3] - self.region[1]),
            "players": [],
            "ball": None,
            "closest_pid": None,
            "possession_pct": {0: 50, 1: 50},
            "fps": 0.0,
            "frame_count": 0,
        }
        self._build_window()

    def _build_window(self) -> None:
        left, top, right, bottom = self.region
        self.setGeometry(left, top, max(1, right - left), max(1, bottom - top))

        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)

    def update_data(self, data: Dict[str, Any]) -> None:
        """Called from the GUI thread through a Qt signal."""
        if not data:
            return
        self.overlay_data = data
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt naming convention
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setFont(QFont("Consolas", 10))

        data = self.overlay_data or {}
        frame_w = max(1, int(data.get("frame_w", self.width())))
        frame_h = max(1, int(data.get("frame_h", self.height())))
        sx = self.width() / frame_w
        sy = self.height() / frame_h

        def px(x: float) -> int:
            return int(round(float(x) * sx))

        def py(y: float) -> int:
            return int(round(float(y) * sy))

        players = data.get("players", []) or []
        closest_pid = data.get("closest_pid")
        ball = data.get("ball")

        # Players.
        for player in players:
            team = player.get("team")
            color = self._team_color(team)
            width = 3 if player.get("track_id") == closest_pid else 2
            painter.setPen(QPen(color, width))

            x1 = px(player.get("x1", 0))
            y1 = py(player.get("y1", 0))
            x2 = px(player.get("x2", 0))
            y2 = py(player.get("y2", 0))
            painter.drawRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))

            self._draw_text(
                painter,
                f"ID:{player.get('track_id', '?')}",
                x1,
                max(14, y1 - 6),
                color,
            )

        # Possession line from ball to closest player's foot.
        if ball is not None and closest_pid is not None:
            closest = next((p for p in players if p.get("track_id") == closest_pid), None)
            if closest is not None:
                painter.setPen(QPen(self.COLOR_POSSESSION, 2))
                painter.drawLine(
                    px(ball.get("x", 0)),
                    py(ball.get("y", 0)),
                    px(closest.get("foot_x", closest.get("x1", 0))),
                    py(closest.get("foot_y", closest.get("y2", 0))),
                )

        # Ball.
        if ball is not None:
            bcolor = self.COLOR_BALL_DEEP if ball.get("source") == "deepball" else self.COLOR_BALL
            bx, by = px(ball.get("x", 0)), py(ball.get("y", 0))
            painter.setPen(QPen(bcolor, 2))
            painter.drawEllipse(bx - 7, by - 7, 14, 14)
            self._draw_text(
                painter,
                f"{str(ball.get('source', '?'))[:1].upper()} {float(ball.get('confidence', 0.0)):.2f}",
                bx + 10,
                by - 8,
                bcolor,
            )

        # Lightweight HUD.
        possession = data.get("possession_pct", {}) or {}
        fps = float(data.get("fps", 0.0) or 0.0)
        frame_count = int(data.get("frame_count", 0) or 0)
        hud = f"AI Overlay | FPS {fps:.1f} | T0 {possession.get(0, 0)}% / T1 {possession.get(1, 0)}% | F {frame_count}"
        self._draw_text(painter, hud, 10, 22, self.COLOR_HUD)

        painter.end()

    def _team_color(self, team: Optional[int]) -> QColor:
        if team == 0:
            return self.COLOR_TEAM_0
        if team == 1:
            return self.COLOR_TEAM_1
        return self.COLOR_UNKNOWN

    def _draw_text(self, painter: QPainter, text: str, x: int, y: int, color: QColor) -> None:
        painter.setPen(QPen(self.COLOR_SHADOW, 3))
        painter.drawText(x + 1, y + 1, text)
        painter.setPen(QPen(color, 1))
        painter.drawText(x, y, text)


def exclude_window_from_capture(widget: QWidget) -> bool:
    """
    Best-effort Windows-only protection so screen capture APIs do not capture
    this overlay/floating control window again.

    Returns True when the OS call succeeds. On unsupported systems it returns False.
    """
    if not sys.platform.startswith("win"):
        return False

    try:
        hwnd = int(widget.winId())
        # Windows 10 2004+: WDA_EXCLUDEFROMCAPTURE. Fallback is harmless if unsupported.
        WDA_EXCLUDEFROMCAPTURE = 0x00000011
        ok = ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        return bool(ok)
    except Exception:
        return False
