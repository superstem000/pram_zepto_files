"""
Auto Focus Module - Laplacian-based Z-sweep autofocus.

Multi-stage cascading sweep: each stage sweeps ± range around the
previous best Z, with decreasing step size. After all stages, moves to
the best Z.

Z moves bypass the MoveTo velocity planner — the AF drives the Z motor
directly via mc.set_speed_fullstep("z", ...) and polls position until
arrival. All internal calculations use full-steps.

The last calibration-AF result is persisted in calibration.json under
'known_focus_fs' and reloaded as the default center on module import.
"""

import json
import os
import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal


KNOWN_FOCUS_FULLSTEP = 170.0


def _try_load_known_focus_from_disk():
    global KNOWN_FOCUS_FULLSTEP
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration.json'),
        os.path.join(os.getcwd(), 'calibration.json'),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    data = json.load(f)
                v = data.get('known_focus_fs')
                if v is not None:
                    KNOWN_FOCUS_FULLSTEP = float(v)
                    print(f"[AutoFocus] Loaded known_focus_fs={KNOWN_FOCUS_FULLSTEP:.2f} from {p}")
                    return
            except Exception as e:
                print(f"[AutoFocus] Could not read known_focus from {p}: {e}")


_try_load_known_focus_from_disk()


# ── Search Profiles (all distances in full-steps) ─────────────────
SEARCH_PROFILES = {
    1: {
        "stages": [
            {"range": 10.0, "step": 4.0},
            {"range": 4.0,  "step": 2.0},
            {"range": 2.0,  "step": 1.0},
        ],
    },
    2: {
        "stages": [
            {"range": 50.0, "step": 10.0},
            {"range": 10.0, "step": 4.0},
            {"range": 4.0,  "step": 2.0},
            {"range": 2.0,  "step": 1.0},
        ],
    },
    4: {
        "stages": [
            {"range": 50.0, "step": 10.0},
            {"range": 10.0, "step": 3.0},
            {"range": 3.0,  "step": 1.0},
            {"range": 1.0,  "step": 0.5},
        ],
    },
    8: {
        "stages": [
            {"range": 10.0, "step": 3.0},
            {"range": 3.0,  "step": 1.0},
            {"range": 1.0,  "step": 0.5},
        ],
    },
    16: {
        "stages": [
            {"range": 10.0, "step": 3.0},
            {"range": 3.0,  "step": 1.0},
            {"range": 1.0,  "step": 0.5},
        ],
    },
    32: {
        "stages": [
            {"range": 10.0, "step": 1.5},
            {"range": 3.0,  "step": 0.5},
            {"range": 0.5,  "step": 0.25},
        ],
    },
}

# Calibration AF: replaces stage 1 with a wide sweep around a fixed Z,
# then uses the regular mres=32 finer stages.
CALIBRATION_AF_STAGE1_CENTER_FS = 175.0
CALIBRATION_AF_STAGE1_RANGE_FS  = 25.0
CALIBRATION_AF_STAGE1_STEP_FS   = 0.5

QUICK_PROFILES = {
    1:  {"stages": [{"range": 4.0, "step": 2.0},  {"range": 2.0, "step": 1.0}]},
    2:  {"stages": [{"range": 4.0, "step": 2.0},  {"range": 2.0, "step": 1.0}]},
    4:  {"stages": [{"range": 3.0, "step": 1.0},  {"range": 1.0, "step": 0.5}]},
    8:  {"stages": [{"range": 3.0, "step": 1.0},  {"range": 1.0, "step": 0.5}]},
    16: {"stages": [{"range": 0.6, "step": 0.2}]},
    32: {"stages": [{"range": 0.6, "step": 0.2}]},
}
QUICK_PROFILES_TIGHT = {
    16: {"stages": [{"range": 0.4, "step": 0.2}]},
    32: {"stages": [{"range": 0.4, "step": 0.2}]},
}
QUICK_PROFILES_MINIMAL = {
    16: {"stages": [{"range": 0.2, "step": 0.2}]},
    32: {"stages": [{"range": 0.4, "step": 0.2}]},
}

# ── Sharpness Metric ──────────────────────────────────────────────


def extract_best_channel(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.float64)
    best_idx = 0
    best_range = 0.0
    for i in range(rgb.shape[2]):
        ch = rgb[:, :, i]
        r = float(ch.max()) - float(ch.min())
        if r > best_range:
            best_range = r
            best_idx = i
    return rgb[:, :, best_idx].astype(np.float64)


def focus_score(gray: np.ndarray) -> float:
    """Variance of Laplacian with flat-field correction to remove vignetting."""
    img = gray.astype(np.float64)
    h, w = img.shape
    ds = 32
    small = img[::ds, ::ds]
    bg = np.repeat(np.repeat(small, ds, axis=0), ds, axis=1)[:h, :w]
    bg = np.maximum(bg, 1.0)
    flat = img / bg
    flat = flat[::4, ::4]
    lap = (flat[:-2, 1:-1] + flat[2:, 1:-1] +
           flat[1:-1, :-2] + flat[1:-1, 2:] -
           4.0 * flat[1:-1, 1:-1])
    return float(np.var(lap))


# ── AutoFocus Controller ──────────────────────────────────────────

class AutoFocusController(QObject):
    """
    Autofocus via direct Z motor control. XY is never touched.

    Signals:
        progress(phase, step, total, z_micro, score)
        finished(success, message)
    """

    progress = pyqtSignal(str, int, int, int, float)
    finished = pyqtSignal(bool, str)

    SETTLE_MS = 200
    DISCARD_FRAMES = 1
    Z_TOL = 0.08
    Z_SPEED = 80.0
    Z_SLOW_RADIUS = 1.0
    Z_MIN_SPEED = 5.0
    POLL_MS = 5

    def __init__(self, move_to_ctrl, camera_preview, mc, motors_config,
                 config=None, parent=None):
        super().__init__(parent)
        self.move_to_ctrl   = move_to_ctrl
        self.camera_preview = camera_preview
        self.mc             = mc
        self.MOTORS         = motors_config
        self.config         = config

        self._active         = False
        self._stop_requested = False
        self._phase          = 'idle'

        self._positions_fs   = []
        self._scores         = []
        self._current_idx    = 0
        self._best_z_fs      = 0.0
        self._camera_settings = {}
        self._mres           = 1

        self._coarse_positions_fs = []
        self._coarse_scores       = []
        self._profile             = {}
        self._saved_speed_mult    = 1.0
        self._stages              = []
        self._stage_idx           = 0

        self._is_calibration_af   = False

        self._z_target        = 0.0
        self._z_min_fs        = 0.0
        self._z_max_fs        = 1200.0

        self._z_timer = QTimer(self)
        self._z_timer.setInterval(self.POLL_MS)
        self._z_timer.timeout.connect(self._z_poll)
        self._z_arrived_cb = None

    # ── Helpers ──

    def _get_z_mres(self) -> int:
        try:
            return int(self.mc.get_axis_microstepping("z"))
        except Exception:
            return 1

    def _fs_to_micro(self, fs: float) -> int:
        return int(round(fs * self._mres))

    def _cur_z(self) -> float:
        return float(getattr(self.MOTORS, "z_current_position_full_step", 0.0))

    # ── Direct Z movement ──

    def _move_z_direct(self, z_target_fs: float, callback):
        self._z_target = z_target_fs
        self._z_arrived_cb = callback
        self._z_timer.start()
        self._z_poll()

    def _z_poll(self):
        if self._stop_requested or not self._active:
            self._z_stop()
            return

        cur_z = self._cur_z()
        error = self._z_target - cur_z
        dist = abs(error)

        if dist <= self.Z_TOL:
            self._z_stop()
            cb = self._z_arrived_cb
            self._z_arrived_cb = None
            if cb:
                cb()
            return

        if dist < self.Z_SLOW_RADIUS:
            alpha = dist / self.Z_SLOW_RADIUS
            speed = self.Z_MIN_SPEED + (self.Z_SPEED - self.Z_MIN_SPEED) * alpha
        else:
            speed = self.Z_SPEED

        if error < 0:
            speed = -speed

        try:
            self.mc.set_speed_fullstep("z", speed)
        except Exception as e:
            print(f"[AutoFocus] Z speed error: {e}")
            self._z_stop()

    def _z_stop(self):
        self._z_timer.stop()
        try:
            self.mc.set_speed_fullstep("z", 0.0)
        except Exception:
            pass

    # ── Public API ──

    def is_active(self) -> bool:
        return self._active

    def start(self, camera_settings: dict):
        """Full autofocus centered on KNOWN_FOCUS_FULLSTEP."""
        if self._active:
            return
        if not self.camera_preview.is_running():
            self.finished.emit(False, "Start live preview before autofocusing.")
            return

        self._active         = True
        self._stop_requested = False
        self._camera_settings = camera_settings
        self._mres = self._get_z_mres()
        self._is_calibration_af = False

        try:
            self._saved_speed_mult = self.mc.get_speed_multiplier()
        except Exception:
            pass

        profile = SEARCH_PROFILES.get(self._mres, SEARCH_PROFILES[32])
        self._stages = profile["stages"]
        self._stage_idx = 0

        self._z_min_fs = float(getattr(self.MOTORS, "z_full_step_min", 0.0))
        self._z_max_fs = float(getattr(self.MOTORS, "z_full_step_max", 1200.0))

        self._best_z_fs = KNOWN_FOCUS_FULLSTEP

        total_positions = sum(
            max(3, int(2 * s["range"] / s["step"]) + 1)
            for s in self._stages
        )
        print(f"[AutoFocus] mres={self._mres}, {len(self._stages)} stages, "
              f"~{total_positions} total positions, center={self._best_z_fs:.2f}fs")

        self._begin_stage()

    def start_calibration_af(self, camera_settings: dict):
        """Calibration-style full AF: wide fixed-center stage 1, then
        the regular finer stages. On success, persists the best Z as
        the new known_focus_fs.
        """
        if self._active:
            return
        if not self.camera_preview.is_running():
            self.finished.emit(False, "Start live preview before autofocusing.")
            return

        self._active         = True
        self._stop_requested = False
        self._camera_settings = camera_settings
        self._mres = self._get_z_mres()
        self._is_calibration_af = True

        try:
            self._saved_speed_mult = self.mc.get_speed_multiplier()
        except Exception:
            pass

        finer_stages = SEARCH_PROFILES.get(self._mres, SEARCH_PROFILES[32])["stages"][1:]
        if not finer_stages:
            finer_stages = SEARCH_PROFILES[32]["stages"][1:]

        self._stages = [
            {"range": CALIBRATION_AF_STAGE1_RANGE_FS,
             "step":  CALIBRATION_AF_STAGE1_STEP_FS},
        ] + list(finer_stages)
        self._stage_idx = 0

        self._z_min_fs = float(getattr(self.MOTORS, "z_full_step_min", 0.0))
        self._z_max_fs = float(getattr(self.MOTORS, "z_full_step_max", 1200.0))

        self._best_z_fs = CALIBRATION_AF_STAGE1_CENTER_FS

        total_positions = sum(
            max(3, int(2 * s["range"] / s["step"]) + 1)
            for s in self._stages
        )
        print(f"[AutoFocus] CALIBRATION AF: mres={self._mres}, "
              f"{len(self._stages)} stages, ~{total_positions} positions, "
              f"stage1 sweeps Z={CALIBRATION_AF_STAGE1_CENTER_FS - CALIBRATION_AF_STAGE1_RANGE_FS:.0f}"
              f"..{CALIBRATION_AF_STAGE1_CENTER_FS + CALIBRATION_AF_STAGE1_RANGE_FS:.0f}fs "
              f"step={CALIBRATION_AF_STAGE1_STEP_FS}fs")

        self._begin_stage()

    def start_quick(self, camera_settings: dict, mode='normal', custom_range=None, center_z=None):
        """Quick refocus around current (or given) Z position."""
        if self._active:
            return
        if not self.camera_preview.is_running():
            self.finished.emit(False, "Start live preview before autofocusing.")
            return

        self._active         = True
        self._stop_requested = False
        self._camera_settings = camera_settings
        self._mres = self._get_z_mres()
        self._is_calibration_af = False

        try:
            self._saved_speed_mult = self.mc.get_speed_multiplier()
        except Exception:
            pass

        if custom_range is not None:
            profile = {"stages": [{"range": custom_range, "step": 0.2}]}
        elif mode == 'tight':
            profile = QUICK_PROFILES_TIGHT.get(self._mres, QUICK_PROFILES_TIGHT[16])
        elif mode == 'minimal':
            profile = QUICK_PROFILES_MINIMAL.get(self._mres, QUICK_PROFILES_MINIMAL[16])
        else:
            profile = QUICK_PROFILES.get(self._mres, QUICK_PROFILES[16])
        self._stages = profile["stages"]
        self._stage_idx = 0

        self._z_min_fs = float(getattr(self.MOTORS, "z_full_step_min", 0.0))
        self._z_max_fs = float(getattr(self.MOTORS, "z_full_step_max", 1200.0))
        if center_z is not None:
            self._best_z_fs = center_z
        else:
            self._best_z_fs = self._cur_z()

        print(f"[AutoFocus] Quick refocus around Z={self._best_z_fs:.2f}fs, "
              f"{len(self._stages)} stages, mode={mode}")

        self._begin_stage()

    def stop(self, reason: str = "Cancelled by user"):
        if not self._active:
            return
        self._stop_requested = True
        self._active = False
        self._phase  = 'idle'
        self._z_stop()
        self._restore_speed()
        self.finished.emit(False, reason)

    def _restore_speed(self):
        try:
            self.mc.set_speed_multiplier(self._saved_speed_mult)
        except Exception:
            pass

    # ── Internal Flow ──

    def _begin_stage(self):
        if self._stop_requested:
            return

        if self._stage_idx >= len(self._stages):
            self._phase = 'move_to_best'
            print(f"[AutoFocus] Finalizing. Moving to best Z={self._best_z_fs:.2f}fs")
            self._move_z_direct(self._best_z_fs, self._on_final_arrival)
            return

        stage = self._stages[self._stage_idx]
        sweep_min = max(self._z_min_fs, self._best_z_fs - stage["range"])
        sweep_max = min(self._z_max_fs, self._best_z_fs + stage["range"])

        n_positions = max(3, int((sweep_max - sweep_min) / stage["step"]) + 1)
        self._positions_fs = np.linspace(sweep_min, sweep_max, n_positions).tolist()
        self._scores = []
        self._current_idx = 0

        self._phase = f"stage_{self._stage_idx + 1}"
        print(f"[AutoFocus] Stage {self._stage_idx+1}: {n_positions} positions, "
              f"Z={sweep_min:.2f}..{sweep_max:.2f}fs")
        self._move_to_next_z()

    def _move_to_next_z(self):
        if self._stop_requested:
            return

        if self._current_idx >= len(self._positions_fs):
            self._on_sweep_complete()
            return

        z_target_fs = self._positions_fs[self._current_idx]
        self._move_z_direct(z_target_fs, self._on_z_arrived)

    def _on_z_arrived(self):
        if self._stop_requested:
            return
        QTimer.singleShot(self.SETTLE_MS, self._capture_and_score)

    def _on_final_arrival(self):
        self._active = False
        self._phase  = 'idle'
        self._restore_speed()

        if self._is_calibration_af:
            self._is_calibration_af = False
            self._persist_known_focus(self._best_z_fs)

        best_micro = self._fs_to_micro(self._best_z_fs)
        best_score = max(self._scores) if self._scores else 0.0
        msg = (f"Focus at Z = {best_micro} µsteps "
               f"({self._best_z_fs:.2f} fs, score {best_score:.0f})")
        print(f"[AutoFocus] Done — {msg}")
        self.finished.emit(True, msg)

    def _persist_known_focus(self, z_fs: float):
        global KNOWN_FOCUS_FULLSTEP
        KNOWN_FOCUS_FULLSTEP = float(z_fs)
        path = self._resolve_calibration_path()
        if not path:
            return
        data = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f) or {}
            except Exception as e:
                print(f"[AutoFocus] Could not read existing {path}: {e}")
                data = {}
        data['known_focus_fs'] = float(z_fs)
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[AutoFocus] Persisted known_focus_fs={z_fs:.2f} to {path}")
        except Exception as e:
            print(f"[AutoFocus] Failed to write {path}: {e}")

    def _resolve_calibration_path(self) -> str:
        cfg_path = getattr(self.config, 'cfg_path', '') if self.config else ''
        if cfg_path:
            return os.path.join(os.path.dirname(cfg_path), 'calibration.json')
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'calibration.json')

    def _capture_and_score(self):
        if self._stop_requested or not self._active:
            return

        try:
            frame = self.camera_preview.flush_and_capture(
                self._camera_settings, discard=self.DISCARD_FRAMES
            )
            if frame is None:
                self.stop("Capture failed — no frame returned")
                return

            gray  = extract_best_channel(frame)
            score = focus_score(gray)

            z_fs    = self._positions_fs[self._current_idx]
            z_micro = self._fs_to_micro(z_fs)
            self._scores.append(score)

            total = len(self._positions_fs)
            self.progress.emit(self._phase,
                               self._current_idx + 1, total,
                               z_micro, score)

            self._current_idx += 1
            self._move_to_next_z()

        except Exception as e:
            self.stop(f"Capture/scoring error: {e}")

    def _on_sweep_complete(self):
        if not self._scores:
            self.stop("No scores collected in sweep stage.")
            return

        best_idx = int(np.argmax(self._scores))
        self._best_z_fs = self._positions_fs[best_idx]
        best_score = self._scores[best_idx]

        print(f"[AutoFocus] stage{self._stage_idx + 1} peak: "
              f"Z={self._best_z_fs:.2f}fs "
              f"({self._fs_to_micro(self._best_z_fs)}µ), score={best_score:.1f}")

        self._stage_idx += 1
        self._begin_stage()
