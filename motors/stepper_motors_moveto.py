from __future__ import annotations
import math, time
from PyQt6.QtCore import QObject, QTimer, pyqtSignal


class StepperMotorsMoveto(QObject):
    """GOTO planner with safe Z sequencing for precision stages.

    When a move involves BOTH XY and Z changes AND Z delta > Z_HOME_THRESHOLD:
      1. Z homes to true zero (via photoelectric switch)
      2. XY moves to target
      3. Z moves to target position

    Small Z deltas (e.g. spiral scan tilt corrections) do XY first,
    then Z, without Z homing.

    Pure XY or pure Z moves skip the irrelevant phases.
    Speed is dynamic: read from the hardware controller's current profile.
    """
    finished = pyqtSignal(bool, str)
    statusChanged = pyqtSignal(str, str)

    Z_HOME_THRESHOLD = 5.0

    def __init__(self, motion_controller, motors_model, mc=None, homing=None,
                 vmax: float = 400.0, parent=None):
        super().__init__(parent)
        self.motion = motion_controller
        self.MOTORS = motors_model
        self.mc = mc
        self.homing = homing
        self._default_vmax = float(vmax)
        self._active = False

        self._phase = "idle"
        self._target = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._final_target = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._saved_profile = None
        self._z_home_connected = False

        self._tol = 0.5
        self._slow_radius = 5.0
        self._vmin = 30.0

        self._stall_timeout_s = 3.0
        self._progress_eps = 0.02

        self._last_dist = None
        self._last_progress_t = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(10)
        self._timer.timeout.connect(self._plan_tick)

    def set_homing(self, homing):
        self.homing = homing

    @property
    def vmax(self) -> float:
        if self.mc is not None:
            try:
                return float(self.mc.get_profile_vmax())
            except Exception:
                pass
        return self._default_vmax

    def is_active(self) -> bool:
        return self._active

    def start(self, tx_fs: float, ty_fs: float, tz_fs: float, profile: str = None):
        """Start a move with safe Z sequencing.

        Combined XY+Z, large Z delta: home Z first, then XY, then Z.
        Combined XY+Z, small Z delta: XY first, then Z (no homing).
        Pure XY: move directly.
        Pure Z to ~0: home Z.
        Pure Z to other: move directly.
        """
        if self._active:
            self.finished.emit(False, "Busy: move-to already active.")
            return

        self._final_target = {"x": float(tx_fs), "y": float(ty_fs), "z": float(tz_fs)}
        cur = self._cur()

        self._saved_profile = None
        if profile is not None and self.mc is not None:
            self._saved_profile = self.mc.get_current_profile()
            self.mc.apply_speed_profile(profile)

        moving_xy = (abs(self._final_target["x"] - cur["x"]) > self._tol or
                     abs(self._final_target["y"] - cur["y"]) > self._tol)
        moving_z = abs(self._final_target["z"] - cur["z"]) > self._tol
        z_delta = abs(self._final_target["z"] - cur["z"])
        large_z = z_delta > self.Z_HOME_THRESHOLD

        if moving_xy and moving_z and large_z:
            if self.homing is not None:
                self._start_z_homing_phase()
            else:
                print("[MoveTo] WARNING: no homing controller, driving Z toward 0")
                self._start_tracking_phase(
                    "z_up", {"x": cur["x"], "y": cur["y"], "z": 0.0}, "Moving...")
        elif moving_xy:
            self._start_tracking_phase(
                "xy",
                {"x": self._final_target["x"], "y": self._final_target["y"], "z": cur["z"]},
                "Moving...")
        elif moving_z:
            if self._final_target["z"] < 1.0 and self.homing is not None:
                self._start_z_homing_phase()
            else:
                self._start_tracking_phase("z_only", self._final_target, "Moving...")
        else:
            self.finished.emit(True, "")
            return

    def _start_tracking_phase(self, phase: str, target: dict, status_text: str):
        """Begin a MoveTo tick-loop phase: set state and start the timer."""
        self._active = True
        self._phase = phase
        self._target = target
        self._last_dist = self._calc_dist(self._cur(), self._target)
        self._last_progress_t = time.monotonic()
        self.statusChanged.emit(status_text, "red")
        self.motion.set_mode("goto")
        self._timer.start()

    def _start_z_homing_phase(self):
        """Begin a Z-only homing phase (no tick loop; homing controller
        drives Z and signals back via _on_z_home_done)."""
        self._active = True
        self._phase = "z_homing"
        self.statusChanged.emit("Homing Z...", "red")
        self._connect_z_home_signal()
        self.homing.home_z_only()

    def stop(self, reason="Move stopped by user."):
        if not self._active:
            return
        if self._phase == "z_homing" and self.homing is not None:
            self.homing.stop(reason)
        self._timer.stop()
        self._active = False
        self._phase = "idle"
        self.motion.stop_immediate()
        self._disconnect_z_home_signal()
        self._restore_profile()
        self.statusChanged.emit("Idle", "gray")
        self.finished.emit(False, reason)

    def _connect_z_home_signal(self):
        if not self._z_home_connected and self.homing is not None:
            self.homing.z_home_finished.connect(self._on_z_home_done)
            self._z_home_connected = True

    def _disconnect_z_home_signal(self):
        if self._z_home_connected and self.homing is not None:
            try:
                self.homing.z_home_finished.disconnect(self._on_z_home_done)
            except Exception:
                pass
            self._z_home_connected = False

    def _on_z_home_done(self, success: bool, message: str):
        self._disconnect_z_home_signal()

        if not self._active or self._phase != "z_homing":
            return

        if not success:
            print(f"[MoveTo] Z homing failed: {message}")
            self._timer.stop()
            self._active = False
            self._phase = "idle"
            self._restore_profile()
            self.statusChanged.emit("Idle", "gray")
            self.finished.emit(False, f"Z homing failed: {message}")
            return

        cur = self._cur()
        moving_xy = (abs(self._final_target["x"] - cur["x"]) > self._tol or
                     abs(self._final_target["y"] - cur["y"]) > self._tol)

        if moving_xy:
            self._phase = "xy"
            self._target = {"x": self._final_target["x"],
                            "y": self._final_target["y"],
                            "z": cur["z"]}
            self._last_dist = self._calc_dist(cur, self._target)
            self._last_progress_t = time.monotonic()
            self.statusChanged.emit("Moving XY...", "red")
            self.motion.set_mode("goto")
            self._timer.start()
        else:
            self._finish()

    def _restore_profile(self):
        if self._saved_profile is not None and self.mc is not None:
            try:
                self.mc.apply_speed_profile(self._saved_profile)
            except Exception:
                pass
            self._saved_profile = None

    def _cur(self):
        return {
            "x": float(getattr(self.MOTORS, "x_current_position_full_step", 0.0)),
            "y": float(getattr(self.MOTORS, "y_current_position_full_step", 0.0)),
            "z": float(getattr(self.MOTORS, "z_current_position_full_step", 0.0)),
        }

    def _calc_dist(self, p1, p2):
        return math.sqrt((p1["x"]-p2["x"])**2 + (p1["y"]-p2["y"])**2 + (p1["z"]-p2["z"])**2)

    def _finish(self):
        self._timer.stop()
        self._active = False
        self._phase = "idle"
        self.motion.stop_immediate()
        self._restore_profile()
        self.statusChanged.emit("Idle", "gray")
        self.finished.emit(True, "")

    def _advance_phase(self):
        """Transition to the next phase after arriving at current target."""
        cur = self._cur()
        now = time.monotonic()

        if self._phase == "z_up":
            # Z-retract fallback path done → XY, then Z target.
            moving_xy = (abs(self._final_target["x"] - cur["x"]) > self._tol or
                         abs(self._final_target["y"] - cur["y"]) > self._tol)
            if moving_xy:
                self._phase = "xy"
                self._target = {"x": self._final_target["x"],
                                "y": self._final_target["y"],
                                "z": cur["z"]}
                self._last_dist = self._calc_dist(cur, self._target)
                self._last_progress_t = now
                self.statusChanged.emit("Moving XY...", "red")
                return
            dz = abs(self._final_target["z"] - cur["z"])
            if dz > self._tol:
                self._phase = "z_down"
                self._target = self._final_target
                self._last_dist = self._calc_dist(cur, self._target)
                self._last_progress_t = now
                self.statusChanged.emit("Moving Z...", "red")
                return
            self._finish()

        elif self._phase == "xy":
            dz = abs(self._final_target["z"] - cur["z"])
            if dz > self._tol:
                self._phase = "z_down"
                self._target = self._final_target
                self._last_dist = self._calc_dist(cur, self._target)
                self._last_progress_t = now
                self.statusChanged.emit("Moving Z...", "red")
                return
            self._finish()

        else:
            self._finish()

    def _plan_tick(self):
        if not self._active:
            return

        if self._phase == "z_homing":
            return  # homing controller handles its own motion

        cur = self._cur()
        dist = self._calc_dist(cur, self._target)
        now = time.monotonic()

        # Stall detection
        if self._last_dist is not None:
            if dist < (self._last_dist - self._progress_eps):
                self._last_dist = dist
                self._last_progress_t = now
            elif (now - self._last_progress_t) > self._stall_timeout_s:
                print(f"[MoveTo] Stall detected: dist={dist:.3f}, "
                      f"last_dist={self._last_dist:.3f}, phase={self._phase}")
                self.stop("Move blocked (Stalled).")
                return

        if dist <= self._tol:
            self._advance_phase()
            return

        ex = self._target["x"] - cur["x"]
        ey = self._target["y"] - cur["y"]
        ez = self._target["z"] - cur["z"]
        ux, uy, uz = ex/dist, ey/dist, ez/dist

        current_vmax = self.vmax
        speed = current_vmax
        if dist < self._slow_radius:
            vmin = min(self._vmin, current_vmax)
            alpha = max(0.0, min(1.0, dist / self._slow_radius))
            speed = vmin + (current_vmax - vmin) * alpha

        self.motion.set_desired_velocity(ux*speed, uy*speed, uz*speed, mode="goto")
