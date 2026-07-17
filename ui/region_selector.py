"""
ui/region_selector.py — Select a screen capture region by clicking/dragging.

Usage:
  - Click the top-left corner, then click the bottom-right corner, or
  - Drag from one corner to the opposite corner.

The selector returns two regions:
  - logical_region : Qt logical pixels, used by the transparent overlay window.
  - physical_region: physical screen pixels, used by dxcam/mss screen capture.

Keeping these two coordinate systems separate is important on Windows when
Display Scaling is 125%, 150%, etc. Using one region for both capture and
Qt overlay is a common reason bbox appears shifted to a wrong screen corner.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QWidget


Region = Tuple[int, int, int, int]  # left, top, right, bottom

class ScreenRegionSelector(QWidget):
    """Fullscreen transparent selector for choosing a capture region."""

    region_selected = pyqtSignal(object, object)  # logical_region, physical_region
    selection_cancelled = pyqtSignal()

    MIN_SIZE = 40

    def __init__(self, parent=None):
        super().__init__(parent)
        self._screen = QApplication.primaryScreen()
        self._screen_geom = self._screen.geometry()
        self._dpr = float(self._screen.devicePixelRatio() or 1.0)

        self._start_global: Optional[QPoint] = None
        self._current_global: Optional[QPoint] = None
        self._dragging = False

        self._build_window()

    def _build_window(self) -> None:
        self.setGeometry(self._screen_geom)
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── EVENTS ───────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):  # noqa: N802 - Qt naming convention
        if event.button() == Qt.MouseButton.RightButton:
            self._cancel()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = self._event_global_pos(event)

        # First click: set top-left/first corner.
        if self._start_global is None:
            self._start_global = pos
            self._current_global = pos
            self._dragging = False
            self.update()
            return

        # Second click: finish selection.
        self._finish(pos)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._start_global is None:
            self._current_global = self._event_global_pos(event)
            self.update()
            return

        pos = self._event_global_pos(event)
        self._current_global = pos

        if event.buttons() & Qt.MouseButton.LeftButton:
            if (pos - self._start_global).manhattanLength() > 20:
                self._dragging = True

        self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802
        # Drag mode: press -> move -> release finishes immediately.
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._start_global is not None
            and self._dragging
        ):
            self._finish(self._event_global_pos(event))

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._start_global is not None and self._current_global is not None:
                self._finish(self._current_global)
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Light dim layer so the selected region is easy to see.
        painter.fillRect(self.rect(), QColor(0, 0, 0, 75))

        painter.setFont(QFont("Consolas", 11))
        painter.setPen(QPen(QColor(0, 212, 255), 1))
        painter.drawText(
            24,
            32,
            "Select Capture Region: click top-left, then click bottom-right | drag also works | ESC cancels",
        )
        painter.drawText(
            24,
            54,
            f"DPI scale used for capture mapping: {self._dpr:.2f}x",
        )

        if self._start_global is not None and self._current_global is not None:
            rect = self._logical_rect(self._start_global, self._current_global)
            local_rect = QRect(
                rect.left() - self.geometry().left(),
                rect.top() - self.geometry().top(),
                rect.width(),
                rect.height(),
            )

            # Clear/darken selected area less by drawing a translucent fill.
            painter.fillRect(local_rect, QColor(0, 212, 255, 30))
            painter.setPen(QPen(QColor(0, 212, 255), 2))
            painter.drawRect(local_rect)

            label = f"{rect.left()}, {rect.top()}  →  {rect.left() + rect.width()}, {rect.top() + rect.height()}   ({rect.width()}×{rect.height()} logical px)"
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(local_rect.left() + 8, max(76, local_rect.top() - 10), label)

        painter.end()

    # ── SELECTION ────────────────────────────────────────────────────────────

    def _event_global_pos(self, event) -> QPoint:
        p = event.position().toPoint()
        return self.mapToGlobal(p)

    def _logical_rect(self, p1: QPoint, p2: QPoint) -> QRect:
        left = min(p1.x(), p2.x())
        top = min(p1.y(), p2.y())
        right = max(p1.x(), p2.x())
        bottom = max(p1.y(), p2.y())
        return QRect(left, top, right - left, bottom - top)

    def _finish(self, end_pos: QPoint) -> None:
        if self._start_global is None:
            return

        rect = self._logical_rect(self._start_global, end_pos)
        if rect.width() < self.MIN_SIZE or rect.height() < self.MIN_SIZE:
            # Too small: treat as a mistaken first click and restart selection.
            self._start_global = end_pos
            self._current_global = end_pos
            self._dragging = False
            self.update()
            return

        logical_region = (
            int(rect.left()),
            int(rect.top()),
            int(rect.left() + rect.width()),
            int(rect.top() + rect.height()),
        )
        physical_region = self._logical_to_physical_region(logical_region)

        self.region_selected.emit(logical_region, physical_region)
        self.close()

    def _logical_to_physical_region(self, region: Region) -> Region:
        left, top, right, bottom = region
        return (
            int(round(left * self._dpr)),
            int(round(top * self._dpr)),
            int(round(right * self._dpr)),
            int(round(bottom * self._dpr)),
        )

    def _cancel(self) -> None:
        self.selection_cancelled.emit()
        self.close()
