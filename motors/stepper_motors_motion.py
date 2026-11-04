from __future__ import annotations
from dataclasses import dataclass
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

@dataclass(frozen=True)
class AABB:
    x0: float; x1: float
    y0: float; y1: float
    z0: float; z1: float

@dataclass(frozen=True)
class AxisLimits:
    x0: float; x1: float
    y0: float; y1: float
    z0: float; z1: float

class MotionMode:
    IDLE = "idle"
    JOYSTICK = "joystick"
    GOTO = "goto"
    HOMING = "homing"

class StepperMotorsMotion(QObject):
    """Single authoritative velocity loop.

    Joystick, move-to and homing set a desired velocity (fullsteps/s).
    Each tick: apply limits and safety policies, command the hardware.
    vmax is read dynamically from the hardware's current profile.
    """
    statusChanged = pyqtSignal(str, str)  # text, color

    def __init__(self, mc, config, motors_model, parent=None):
        super().__init__(parent)
        self.mc = mc
        self.config = config
        self.MOTORS = motors_model

        m = getattr(config, "motor", None)
        self.limits = AxisLimits(
            x0=float(getattr(m, "x_full_step_min", 0.0)),
            x1=float(getattr(m, "x_full_step_max", 0.0)),
            y0=float(getattr(m, "y_full_step_min", 0.0)),
            y1=float(getattr(m, "y_full_step_max", 0.0)),
            z0=float(getattr(m, "z_full_step_min", 0.0)),
            z1=float(getattr(m, "z_full_step_max", 0.0)),
        )

        fz = getattr(config, "forbidden_zone", None)
        self.forbidden = AABB(
            x0=float(getattr(fz, "x_forbidden_start", 0.0)),
            x1=float(getattr(fz, "x_forbidden_end", 0.0)),
            y0=float(getattr(fz, "y_forbidden_start", 0.0)),
            y1=float(getattr(fz, "y_forbidden_end", 0.0)),
            z0=float(getattr(fz, "z_forbidden_start", 0.0)),
            z1=float(getattr(fz, "z_forbidden_end", 0.0)),
        )

        # ── Safety bounds ──
        sb = getattr(config, "safety_bounds", None)
        self._sb_z_global_max = float(getattr(sb, "z_global_max", 80.0)) if sb else 80.0
        self._sb_xy_forbidden_x = float(getattr(sb, "xy_forbidden_x", 3800.0)) if sb else 3800.0
        self._sb_xy_forbidden_y = float(getattr(sb, "xy_forbidden_y", 850.0)) if sb else 850.0

        zsr = getattr(sb, "z_safe_rect", None) if sb else None
        self._zsr_x_min = float(getattr(zsr, "x_min", 4050.0)) if zsr else 4050.0
        self._zsr_x_max = float(getattr(zsr, "x_max", 4450.0)) if zsr else 4450.0
        self._zsr_y_min = float(getattr(zsr, "y_min", 260.0)) if zsr else 260.0
        self._zsr_y_max = float(getattr(zsr, "y_max", 670.0)) if zsr else 670.0
        self._zsr_z_max = float(getattr(zsr, "z_max", 300.0)) if zsr else 300.0

        motion = getattr(config, "motion", None)
        self._fallback_vmax = float(getattr(motion, "max_speed_fullstep", 400.0))
        self.tick_ms = int(getattr(motion, "control_tick_ms", 10))

        # Margin to reduce tunneling through the forbidden zone (fullsteps)
        self.forbidden_margin_fs = float(getattr(motion, "forbidden_margin_fullstep", 10.0))
        # Escape speed if we detect we're inside forbidden and commanded velocity is 0
        self.escape_speed_fs = float(getattr(motion, "escape_speed_fullstep", 200.0))

        self.mode = MotionMode.IDLE
        self.v_des = {"x": 0.0, "y": 0.0, "z": 0.0}

        self._timer = QTimer(self)
        self._timer.setInterval(self.tick_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # -------- public API --------
    @property
    def vmax(self) -> float:
        """Dynamic vmax: read from the hardware controller's current profile."""
        try:
            return float(self.mc.get_profile_vmax())
        except Exception:
            return self._fallback_vmax

    def set_mode(self, mode: str):
        self.mode = mode

    def set_desired_velocity(self, vx: float, vy: float, vz: float, mode: str | None = None):
        self.v_des["x"] = float(vx)
        self.v_des["y"] = float(vy)
        self.v_des["z"] = float(vz)
        if mode is not None:
            self.mode = mode

    def stop_immediate(self):
        self.v_des = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.mode = MotionMode.IDLE
        try:
            self.mc.halt_all_speed()
        except Exception:
            for ax in ("x", "y", "z"):
                try:
                    self.mc.set_speed_fullstep(ax, 0.0)
                except Exception:
                    pass

    # -------- internals --------
    def _get_cur_fs(self):
        return {
            "x": float(getattr(self.MOTORS, "x_current_position_full_step", 0.0)),
            "y": float(getattr(self.MOTORS, "y_current_position_full_step", 0.0)),
            "z": float(getattr(self.MOTORS, "z_current_position_full_step", 0.0)),
        }

    def _tick(self):
        dt = max(0.001, float(self.tick_ms) / 1000.0)
        cur = self._get_cur_fs()
        vm = self.vmax

        v = {
            "x": max(-vm, min(vm, float(self.v_des["x"]))),
            "y": max(-vm, min(vm, float(self.v_des["y"]))),
            "z": max(-vm, min(vm, float(self.v_des["z"]))),
        }

        v_safe = self._apply_policies(cur, v, dt)

        EPS = 1e-6
        try:
            self.mc.set_speed_fullstep("x", 0.0 if abs(v_safe["x"]) < EPS else v_safe["x"])
            self.mc.set_speed_fullstep("y", 0.0 if abs(v_safe["y"]) < EPS else v_safe["y"])
            self.mc.set_speed_fullstep("z", 0.0 if abs(v_safe["z"]) < EPS else v_safe["z"])
        except Exception:
            pass

    def _apply_policies(self, cur, v, dt):
        """
        Policy layers applied in order:
          0) Safety bounds (Z limits, XY forbidden, Z safe rectangle)
          1) Per-axis travel limits (hard min/max)
          2) XY forbidden zone (2D) with escape
        """
        # ---------- 0) Safety bounds ----------
        if self.mode != MotionMode.HOMING:
            v = self._apply_safety_bounds(cur, v, dt)

        # ---------- 1) Travel limits (axis-wise) ----------
        lim = self.limits
        nxt = {ax: cur[ax] + v[ax] * dt for ax in ("x", "y", "z")}

        if self.mode != MotionMode.HOMING:
            if nxt["x"] < lim.x0 and v["x"] < 0: v["x"] = 0.0
            if nxt["x"] > lim.x1 and v["x"] > 0: v["x"] = 0.0
            if nxt["y"] < lim.y0 and v["y"] < 0: v["y"] = 0.0
            if nxt["y"] > lim.y1 and v["y"] > 0: v["y"] = 0.0
            if nxt["z"] < lim.z0 and v["z"] < 0: v["z"] = 0.0
            if nxt["z"] > lim.z1 and v["z"] > 0: v["z"] = 0.0

        # ---------- 2) XY forbidden (2D) with escape ----------
        b = self.forbidden
        m = float(self.forbidden_margin_fs)
        fx0, fx1 = b.x0 - m, b.x1 + m
        fy0, fy1 = b.y0 - m, b.y1 + m

        def xy_forbidden_at(p):
            return (fx0 < p["x"] < fx1) and (fy0 < p["y"] < fy1)

        cur_xy_in = xy_forbidden_at(cur)

        def escape_component(p, lo, hi, vel):
            if not (lo < p < hi):
                return vel
            dist_lo = p - lo
            dist_hi = hi - p
            exit_dir = -1.0 if dist_lo <= dist_hi else +1.0
            return vel if (vel * exit_dir) > 0 else 0.0

        if cur_xy_in:
            v2 = dict(v)
            v2["x"] = escape_component(cur["x"], fx0, fx1, v2["x"])
            v2["y"] = escape_component(cur["y"], fy0, fy1, v2["y"])

            if abs(v2["x"]) < 1e-9 and abs(v2["y"]) < 1e-9:
                sp = float(self.escape_speed_fs)
                dx = min(cur["x"] - fx0, fx1 - cur["x"])
                dy = min(cur["y"] - fy0, fy1 - cur["y"])
                if dx <= dy:
                    exit_dir = -1.0 if (cur["x"] - fx0) <= (fx1 - cur["x"]) else +1.0
                    v2["x"] = exit_dir * sp
                else:
                    exit_dir = -1.0 if (cur["y"] - fy0) <= (fy1 - cur["y"]) else +1.0
                    v2["y"] = exit_dir * sp

            return v2

        nxt = {ax: cur[ax] + v[ax] * dt for ax in ("x", "y", "z")}
        if not xy_forbidden_at(nxt):
            return v

        def would_enter(v_local):
            p = {ax: cur[ax] + v_local[ax] * dt for ax in ("x", "y", "z")}
            return xy_forbidden_at(p)

        for ax in ("x", "y"):
            v_try = dict(v)
            v_try[ax] = 0.0
            if not would_enter(v_try):
                return v_try

        v_out = dict(v)
        v_out["x"] = 0.0
        v_out["y"] = 0.0
        return v_out

    def _apply_safety_bounds(self, cur, v, dt):
        """
        Safety bounds rules:

        1. XY forbidden: cannot have x > 3800 AND y > 850 at the same time.

        2. Z limit depends on XY position:
           - Inside the safe rectangle (x in [4050,4450], y in [260,670]):
             Z can go up to 300
           - Everywhere else: Z must stay below 80

        3. If Z > 80 and inside safe rect, block XY motion that would
           leave the safe rect (since Z would then exceed the global limit).
        """
        nxt = {ax: cur[ax] + v[ax] * dt for ax in ("x", "y", "z")}
        v_out = dict(v)

        # -- Rule 1: XY forbidden (x > 3800 AND y > 850 simultaneously) --
        if nxt["x"] > self._sb_xy_forbidden_x and nxt["y"] > self._sb_xy_forbidden_y:
            if v_out["x"] > 0 and cur["x"] <= self._sb_xy_forbidden_x:
                v_out["x"] = 0.0
            if v_out["y"] > 0 and cur["y"] <= self._sb_xy_forbidden_y:
                v_out["y"] = 0.0
            if cur["x"] > self._sb_xy_forbidden_x and cur["y"] > self._sb_xy_forbidden_y:
                if v_out["x"] > 0: v_out["x"] = 0.0
                if v_out["y"] > 0: v_out["y"] = 0.0

        # -- Rule 2: Z limit based on XY position --
        in_safe_rect = (self._zsr_x_min <= nxt["x"] <= self._zsr_x_max and
                        self._zsr_y_min <= nxt["y"] <= self._zsr_y_max)

        z_limit = self._zsr_z_max if in_safe_rect else self._sb_z_global_max

        if nxt["z"] > z_limit and v_out["z"] > 0:
            v_out["z"] = 0.0

        # -- Rule 3: Trap prevention --
        # If Z > global max and inside safe rect, block XY from leaving rect
        if cur["z"] > self._sb_z_global_max:
            cur_in_safe = (self._zsr_x_min <= cur["x"] <= self._zsr_x_max and
                           self._zsr_y_min <= cur["y"] <= self._zsr_y_max)
            if cur_in_safe:
                nxt_in_safe = (self._zsr_x_min <= nxt["x"] <= self._zsr_x_max and
                               self._zsr_y_min <= nxt["y"] <= self._zsr_y_max)
                if not nxt_in_safe:
                    if nxt["x"] < self._zsr_x_min and v_out["x"] < 0: v_out["x"] = 0.0
                    if nxt["x"] > self._zsr_x_max and v_out["x"] > 0: v_out["x"] = 0.0
                    if nxt["y"] < self._zsr_y_min and v_out["y"] < 0: v_out["y"] = 0.0
                    if nxt["y"] > self._zsr_y_max and v_out["y"] > 0: v_out["y"] = 0.0

        return v_out