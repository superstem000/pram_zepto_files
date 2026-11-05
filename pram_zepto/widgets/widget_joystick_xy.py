from PyQt6 import QtCore, QtGui, QtWidgets
import math

class JoystickXYWidget(QtWidgets.QWidget):
    """
    A square, center-origin joystick-like widget:
      - Click/drag anywhere inside the box to set an (x, y) command.
      - Outputs continuous speed commands at a fixed rate while pressed.
      - Speed ∝ distance from center; sign encodes direction.

    Signals:
      speedCommand(float vx, float vy)  # emitted every tick while pressed
      stopCommand()                      # emitted once on release
    """
    speedCommand = QtCore.pyqtSignal(float, float)
    stopCommand = QtCore.pyqtSignal()

    def __init__(self,
                 parent=None,
                 max_speed_x: float = 500.0,   # steps/s or mm/s
                 max_speed_y: float = 500.0,
                 update_hz: int = 30,
                 dead_zone_px: int = 8,
                 response: str = "linear"  # "linear" | "exp"
                 ):
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self.setMouseTracking(True)
        self._pressed = False
        self._cursor_pos = None
        self._dead_zone_px = dead_zone_px
        self._max_speed_x = float(max_speed_x)
        self._max_speed_y = float(max_speed_y)
        self._response = response

        # timer for continuous output while held
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(1000 / max(1, update_hz)))
        self._timer.timeout.connect(self._emit_speed)

        # cosmetic
        self._bg = self.palette().base().color()
        self._axis_pen = QtGui.QPen(QtGui.QColor(130, 130, 130), 1, QtCore.Qt.PenStyle.SolidLine)
        self._box_pen = QtGui.QPen(QtGui.QColor(120, 120, 120), 1)
        self._center_brush = QtGui.QBrush(QtGui.QColor(60, 150, 255, 180))
        self._handle_brush = QtGui.QBrush(QtGui.QColor(60, 150, 255, 120))

    def set_max_speed(self, max_speed_x: float, max_speed_y: float):
        """Dynamically update max speeds (e.g. when switching speed profiles)."""
        self._max_speed_x = float(max_speed_x)
        self._max_speed_y = float(max_speed_y)

    # ----------------- geometry helpers -----------------
    def _square_rect(self) -> QtCore.QRect:
        r = self.rect()
        margin = 0  # small padding inside the widget
        side = min(r.width(), r.height()) - 2 * margin
        side = max(10, side)
        return QtCore.QRect(r.left() + margin, r.top() + margin, side, side)

    def _center(self) -> QtCore.QPoint:
        return self._square_rect().center()

    def _clamp_to_square(self, p: QtCore.QPoint) -> QtCore.QPoint:
        sr = self._square_rect()
        x = max(sr.left(), min(p.x(), sr.right()))
        y = max(sr.top(),  min(p.y(), sr.bottom()))
        return QtCore.QPoint(x, y)

    # ----------------- input handling -----------------
    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pressed = True
            self._cursor_pos = self._clamp_to_square(e.position().toPoint())
            self._timer.start()
            self.update()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        if self._pressed:
            self._cursor_pos = self._clamp_to_square(e.position().toPoint())
            # emit immediately for reactive feel; timer also maintains rate
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

    # ----------------- mapping logic -----------------
    def _emit_speed(self):
        if not self._pressed or self._cursor_pos is None:
            return
        center = self._center()
        sr = self._square_rect()
        half = sr.width() / 2.0

        # vector from center (pixels); +x to right, +y down (Qt coords)
        dx_px = self._cursor_pos.x() - center.x()
        dy_px = self._cursor_pos.y() - center.y()

        # dead zone
        if math.hypot(dx_px, dy_px) < self._dead_zone_px:
            self.speedCommand.emit(0.0, 0.0)
            return

        # normalize to [-1, 1] relative to square half-size
        nx = max(-1.0, min(1.0, dx_px / half))
        ny = max(-1.0, min(1.0, dy_px / half))

        # Optional non-linear response for finer center control
        if self._response == "exp":
            # smooth S-shaped curve preserving sign; gamma in (1,3]
            gamma = 2.0
            nx = math.copysign(abs(nx) ** gamma, nx)
            ny = math.copysign(abs(ny) ** gamma, ny)

        # Map to speeds (invert Y if your stage Y+ is up)
        vx = nx * self._max_speed_x
        vy = -ny * self._max_speed_y  # minus to convert screen-down to stage-up
        self.speedCommand.emit(vx, vy)

    # ----------------- painting -----------------
    def paintEvent(self, e: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        r = self._square_rect()
        p.fillRect(r, self._bg)
        p.setPen(self._box_pen)
        p.drawRect(r.adjusted(0, 0, -1, -1))

        # axes
        p.setPen(self._axis_pen)
        c = r.center()
        p.drawLine(r.left(), c.y(), r.right(), c.y())
        p.drawLine(c.x(), r.top(), c.x(), r.bottom())

        # dead-zone circle
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, self._dead_zone_px, self._dead_zone_px)

        # center disc
        p.setBrush(self._center_brush)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawEllipse(c, 4, 4)

        # handle at cursor
        if self._cursor_pos is not None:
            p.setBrush(self._handle_brush)
            p.setPen(QtGui.QPen(QtGui.QColor(60, 150, 255, 220), 2))
            p.drawEllipse(self._cursor_pos, 10, 10)

        # textual HUD: normalized and speed values
        if self._cursor_pos is not None:
            dx_px = self._cursor_pos.x() - c.x()
            dy_px = self._cursor_pos.y() - c.y()
            half = r.width() / 2.0
            nx = dx_px / half
            ny = dy_px / half
            vx = nx * self._max_speed_x
            vy = -ny * self._max_speed_y
            text = f"nx={nx:+.2f}, ny={ny:+.2f}  |  vx={vx:+.0f}, vy={vy:+.0f}"
            p.setPen(QtGui.QColor(80, 80, 80))
            p.setFont(QtGui.QFont("Arial", 10))
            p.drawText(r.adjusted(6, 6, -6, -6), QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft, text)

        p.end()