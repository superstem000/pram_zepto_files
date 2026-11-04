from __future__ import annotations
import time
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from motors.stepper_motors_motion import MotionMode

class StepperMotorsHoming(QObject):
    """Homing state machine.

    Uses PhotoSwitchMonitor edges and commands velocities via MotionController.
    Modes: full (Z first, then XY) or Z-only.
    """
    finished = pyqtSignal(bool, str)  # success, message
    statusChanged = pyqtSignal(str, str)
    axisHomed = pyqtSignal(str)
    z_home_finished = pyqtSignal(bool, str)  # separate signal for z-only homing

    def __init__(self, mc, motion_controller, photo_monitor, config, motors_model, parent=None):
        super().__init__(parent)
        self.mc = mc
        self.motion = motion_controller
        self.photo = photo_monitor
        self.config = config
        self.MOTORS = motors_model

        self._active = False
        self._pending = set()
        self._state = {}
        self._home_clear_min_s = 0.6
        self._home_seek_speed = 200.0
        self._home_clear_speed = 120.0
        self._timeout_ms = 100000
        self._saved_profile = "normal"  # profile to restore after homing
        self._homing_phase = ""  # "z_first", "xy", or "z_only"

        self._watchdog = QTimer(self)
        self._watchdog.setSingleShot(True)
        self._watchdog.timeout.connect(self._on_timeout)

        self.photo.entered_home.connect(self._on_entered_home)
        self.photo.left_home.connect(self._on_left_home)

    def is_active(self) -> bool:
        return self._active

    def start(self):
        """Full homing: Z first, then XY."""
        if self._active:
            return

        # stop any motion first
        self.motion.stop_immediate()

        # Build the list of axes that actually have switches configured
        self._pending = set()
        for ax in ("x", "y", "z"):
            gpio = getattr(self.config.axes[ax], "photoelectric_switch", None)
            if gpio is not None:
                self._pending.add(ax)

        if not self._pending:
            self.finished.emit(False, "No photoelectric switches configured.")
            return

        self._active = True
        self._state = {}
        now = time.monotonic()

        # Homing ALWAYS uses max speed
        try:
            self.mc.set_speed_for_homing()
        except Exception:
            pass

        # Initialize state records for all axes
        for ax in self._pending:
            self._state[ax] = {"phase": None, "t0": now, "seek_timer": None}

        # SAFETY: Home Z first (move to 0) before XY, to avoid Z being
        # too high while XY moves through unsafe regions.
        if "z" in self._pending:
            self._homing_phase = "z_first"
            hd = int(getattr(self.config.axes["z"], "home_dir", -1))
            if self.photo.is_home("z"):
                self._state["z"]["phase"] = "clear"
                vz = -hd * self._home_clear_speed
            else:
                self._state["z"]["phase"] = "seek"
                vz = +hd * self._home_seek_speed
            self.motion.set_desired_velocity(0.0, 0.0, vz, mode=MotionMode.HOMING)
        else:
            # No Z axis — go straight to XY
            self._start_xy_homing()

        self.statusChanged.emit("Homing Z first...", "orange")
        self._watchdog.start(int(self._timeout_ms))

    def home_z_only(self):
        """Home Z axis only — retract to true zero without touching XY.
        Emits z_home_finished(success, message) when done."""
        if self._active:
            self.z_home_finished.emit(False, "Homing already active.")
            return

        # Check Z has a switch configured
        gpio = getattr(self.config.axes["z"], "photoelectric_switch", None)
        if gpio is None:
            self.z_home_finished.emit(False, "No Z photoelectric switch configured.")
            return

        self.motion.stop_immediate()

        self._active = True
        self._pending = {"z"}
        self._homing_phase = "z_only"
        now = time.monotonic()

        # Use max speed for homing
        try:
            self.mc.set_speed_for_homing()
        except Exception:
            pass

        self._state = {"z": {"phase": None, "t0": now, "seek_timer": None}}

        hd = int(getattr(self.config.axes["z"], "home_dir", -1))
        if self.photo.is_home("z"):
            self._state["z"]["phase"] = "clear"
            vz = -hd * self._home_clear_speed
        else:
            self._state["z"]["phase"] = "seek"
            vz = +hd * self._home_seek_speed

        # Only move Z — leave XY at zero velocity
        self.motion.set_desired_velocity(0.0, 0.0, vz, mode=MotionMode.HOMING)
        self.statusChanged.emit("Homing Z...", "orange")
        self._watchdog.start(int(self._timeout_ms))

    def _start_xy_homing(self):
        """Start homing X and Y axes (called after Z is homed)."""
        now = time.monotonic()
        v = {"x": 0.0, "y": 0.0, "z": 0.0}
        xy_pending = False
        for ax in ("x", "y"):
            if ax not in self._pending:
                continue
            xy_pending = True
            hd = int(getattr(self.config.axes[ax], "home_dir", -1))
            self._state[ax] = {"phase": None, "t0": now, "seek_timer": None}
            if self.photo.is_home(ax):
                self._state[ax]["phase"] = "clear"
                v[ax] = -hd * self._home_clear_speed
            else:
                self._state[ax]["phase"] = "seek"
                v[ax] = +hd * self._home_seek_speed

        if xy_pending:
            self._homing_phase = "xy"
            self.statusChanged.emit("Homing XY...", "orange")
            self.motion.set_desired_velocity(v["x"], v["y"], 0.0, mode=MotionMode.HOMING)
        else:
            self._finish(True, "")

    def stop(self, reason="Homing stopped by user."):
        if not self._active:
            return
        self.motion.stop_immediate()
        self._finish(False, reason)

    def _on_entered_home(self, axis: str):
        axis = axis.lower()
        if not self._active or axis not in self._pending:
            return

        st = self._state.get(axis, {})
        if st.get("phase") != "seek":
            return

        # Hard stop axis and latch logical home position.
        self.mc.set_speed_fullstep(axis, 0.0)
        self.motion.v_des[axis] = 0.0
        self.mc.set_logical_position_micro(axis, 0)
        setattr(self.MOTORS, f"{axis}_current_position_full_step", 0.0)

        self._pending.discard(axis)

        if not self._pending:
            self.motion.stop_immediate()
            self._finish(True, "")
        elif axis == "z" and self._homing_phase == "z_first":
            self.motion.stop_immediate()
            self._start_xy_homing()

    def _on_left_home(self, axis: str):
        axis = axis.lower()
        if not self._active or axis not in self._pending:
            return
        st = self._state.get(axis)
        if not st or st.get("phase") != "clear":
            return

        elapsed = time.monotonic() - float(st.get("t0", time.monotonic()))
        remaining = float(self._home_clear_min_s) - elapsed

        if remaining <= 0:
            self._switch_clear_to_seek(axis)
        else:
            if st.get("seek_timer") is None:
                t = QTimer(self)
                t.setSingleShot(True)
                t.timeout.connect(lambda ax=axis: self._switch_clear_to_seek(ax))
                st["seek_timer"] = t
                t.start(int(remaining * 1000))

    def _switch_clear_to_seek(self, axis: str):
        axis = axis.lower()
        if not self._active or axis not in self._pending:
            return

        st = self._state.get(axis)
        if not st or st.get("phase") != "clear":
            return

        st["phase"] = "seek"
        st["t0"] = time.monotonic()

        hd = int(getattr(self.config.axes[axis], "home_dir", -1))
        self.motion.v_des[axis] = +float(hd) * float(self._home_seek_speed)

    def _on_timeout(self):
        if not self._active:
            return
        self.motion.stop_immediate()
        self._finish(False, "Homing timed out. Check switch polarity/direction.")

    def _finish(self, success: bool, message: str):
        was_z_only = (self._homing_phase == "z_only")
        self._active = False
        self._homing_phase = ""
        try:
            if self._watchdog.isActive():
                self._watchdog.stop()
        except Exception:
            pass
        self.motion.stop_immediate()
        # Restore user's speed multiplier
        try:
            self.mc.restore_speed_multiplier()
        except Exception:
            pass
        self.statusChanged.emit("Idle" if success else "Idle", "gray" if success else "red")

        if was_z_only:
            self.z_home_finished.emit(success, message)
        else:
            self.finished.emit(success, message)