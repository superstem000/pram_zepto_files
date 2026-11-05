from PyQt6 import QtCore, QtGui, QtWidgets
import math

class JoystickZWidget(QtWidgets.QWidget):
    """
    A 1D, center-origin joystick-like widget for Z jogging:
      - Click/drag up/down to command vz.
      - Emits continuous vz at fixed rate while pressed.
      - vz sign: up on screen => positive vz (widget convention).

    Signals:
      speedCommand(float vz)
      stopCommand()
    """
    speedCommand = QtCore.pyqtSignal(float)
    stopCommand  = QtCore.pyqtSignal()

    def __init__(self,
                 parent=None,
                 max_speed_z: float = 500.0,
                 update_hz: int = 30,
                 dead_zone_px: int = 8,
                 response: str = "linear"  # "linear" | "exp"
                 ):
        super().__init__(parent)
        self.setMinimumSize(90, 240)
        self.setMouseTracking(True)

        self._pressed = False
        self._cursor_pos = None
        self._dead_zone_px = int(dead_zone_px)
        self._max_speed_z = float(max_speed_z)
        self._response = response

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(1000 / max(1, update_hz)))
        self._timer.timeout.connect(self._emit_speed)

        # cosmetic
        self._bg = self.palette().base().color()
        self._axis_pen   = QtGui.QPen(QtGui.QColor(130, 130, 130), 1)
        self._box_pen    = QtGui.QPen(QtGui.QColor(120, 120, 120), 1)
        self._center_brush = QtGui.QBrush(QtGui.QColor(60, 150, 255, 180))
        self._handle_brush = QtGui.QBrush(QtGui.QColor(60, 150, 255, 120))

    def set_max_speed(self, max_speed_z: float):
        """Dynamically update max Z speed (e.g. when switching speed profiles)."""
        self._max_speed_z = float(max_speed_z)

    # -------- geometry helpers --------
    def _track_rect(self) -> QtCore.QRect:
        r = self.rect()
        # a tall inner rectangle with small margins
        return r.adjusted(10, 10, -10, -10)

    def _center_y(self) -> int:
        return self._track_rect().center().y()

    def _clamp_to_track(self, p: QtCore.QPoint) -> QtCore.QPoint:
        tr = self._track_rect()
        x = tr.center().x()
        y = max(tr.top(), min(p.y(), tr.bottom()))
        return QtCore.QPoint(x, y)

    # -------- input handling --------
    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pressed = True
            self._cursor_pos = self._clamp_to_track(e.position().toPoint())
            self._timer.start()
            self.update()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        if self._pressed:
            self._cursor_pos = self._clamp_to_track(e.position().toPoint())
            self._emit_speed()
            self.update()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.MouseButton.LeftButton and self._pressed:
            self._pressed = False
            self._cursor_pos = None
            self._timer.stop()
            self.stopCommand.emit()
            self.update()
        super().mouseReleaseEvent(e)

    # -------- mapping logic --------
    def _emit_speed(self):
        if not self._pressed or self._cursor_pos is None:
            return

        tr = self._track_rect()
        cy = tr.center().y()
        half = tr.height() / 2.0

        # Qt coords: +y down. We want "up" => +vz (widget convention),
        # so use dy = cy - y.
        dy_px = cy - self._cursor_pos.y()

        # dead zone
        if abs(dy_px) < self._dead_zone_px:
            self.speedCommand.emit(0.0)
            return

        # normalize to [-1, 1]
        nz = max(-1.0, min(1.0, dy_px / half))

        if self._response == "exp":
            gamma = 2.0
            nz = math.copysign(abs(nz) ** gamma, nz)

        vz = nz * self._max_speed_z
        self.speedCommand.emit(vz)

    # -------- painting --------
    def paintEvent(self, e: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        tr = self._track_rect()
        p.fillRect(tr, self._bg)
        p.setPen(self._box_pen)
        p.drawRect(tr.adjusted(0, 0, -1, -1))

        # axis line
        p.setPen(self._axis_pen)
        cx = tr.center().x()
        cy = tr.center().y()
        p.drawLine(cx, tr.top(), cx, tr.bottom())
        p.drawLine(tr.left(), cy, tr.right(), cy)

        # dead-zone band (small line segment)
        p.setPen(QtGui.QPen(QtGui.QColor(140, 140, 140), 2))
        p.drawLine(cx - 10, cy - self._dead_zone_px, cx + 10, cy - self._dead_zone_px)
        p.drawLine(cx - 10, cy + self._dead_zone_px, cx + 10, cy + self._dead_zone_px)

        # center disc
        p.setBrush(self._center_brush)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawEllipse(QtCore.QPoint(cx, cy), 4, 4)

        # handle
        if self._cursor_pos is not None:
            p.setBrush(self._handle_brush)
            p.setPen(QtGui.QPen(QtGui.QColor(60, 150, 255, 220), 2))
            p.drawEllipse(self._cursor_pos, 10, 10)

            # HUD
            dy_px = cy - self._cursor_pos.y()
            half = tr.height() / 2.0
            nz = dy_px / half
            vz = nz * self._max_speed_z
            text = f"nz={nz:+.2f}  |  vz={vz:+.0f}"
            p.setPen(QtGui.QColor(80, 80, 80))
            p.setFont(QtGui.QFont("Arial", 10))
            p.drawText(tr.adjusted(6, 6, -6, -6),
                       QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft,
                       text)

        p.end()