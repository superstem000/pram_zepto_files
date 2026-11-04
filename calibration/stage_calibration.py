"""
Stage Calibration Module — full 2×2 affine calibration with multi-point
refinement and persistent storage.

Calibration strategy (3 phases):
  Phase 1 — Local probe:
    Move motor X and Y by small amounts, measure pixel shifts.
    Gives initial motor_to_px matrix.
  Phase 2 — Grid pitch:
    Measure pixel spacing between detected markers in one frame.
  Phase 3 — Refinement (iterative & live):
    Move exactly 1 unit X, measure error, instantly recalibrate X.
    Move exactly 1 unit Y, measure error, instantly recalibrate Y.
    Move 2 units diagonally, measure error, apply final distributed fix.

Persistence & Continuous Learning:
    Calibration is saved to calibration.json next to config.json.
    On startup, the last calibration is loaded automatically.
    Every successful `move_to_feature` jump calculates the drift error
    and updates the calibration by a small weight to self-heal over time.

Grid orientation (physical chip):
  - (0, 0) at TOP-RIGHT
  - Increasing col goes LEFT
  - Increasing row goes DOWN
"""

import json
import math
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from vision.feature_detection import detect_markers, save_failed_frame


def measure_pixel_pitch_from_markers(markers):
    """Compute grid pitch in pixels directly from observed marker positions.

    Uses ONLY axis-aligned marker pairs (same row for col pitch, same col
    for row pitch).  Diagonal pairs are skipped because dividing the full
    diagonal pixel displacement by one axis conflates both axes, introducing
    large systematic errors in the cross-term.

    Returns (col_pitch_px, row_pitch_px) as numpy arrays, or (None, None)
    for each axis that has no axis-aligned pairs.
    """
    if len(markers) < 2:
        return None, None
    col_deltas = []
    row_deltas = []
    for i, m1 in enumerate(markers):
        for m2 in markers[i+1:]:
            dcol = m2['col_id'] - m1['col_id']
            drow = m2['row_id'] - m1['row_id']
            dpx = np.array([m2['center'][0] - m1['center'][0],
                            m2['center'][1] - m1['center'][1]])
            if dcol != 0 and drow == 0:
                col_deltas.append(dpx / dcol)
            elif drow != 0 and dcol == 0:
                row_deltas.append(dpx / drow)
            # Diagonal pairs intentionally skipped
    col_pitch = np.mean(col_deltas, axis=0) if col_deltas else None
    row_pitch = np.mean(row_deltas, axis=0) if row_deltas else None
    return col_pitch, row_pitch


# ── Calibration Result ────────────────────────────────────────────

class CalibrationResult:
    """
    Stores the pixel ↔ full-step mapping as a 2×2 matrix.

    motor_to_px: pixel shift per motor full-step
        pixel_delta = motor_to_px @ motor_delta

    px_to_motor: motor full-steps per pixel
        motor_delta = px_to_motor @ pixel_delta
    """

    def __init__(self):
        self.motor_to_px = np.zeros((2, 2), dtype=float)
        self.px_to_motor = np.zeros((2, 2), dtype=float)

        # Grid pitch in pixels: 2D vector per col/row step
        self.grid_pitch_col_px = np.zeros(2, dtype=float)
        self.grid_pitch_row_px = np.zeros(2, dtype=float)

        # Grid pitch in motor full-steps (derived)
        self.grid_pitch_col_fs = np.zeros(2, dtype=float)
        self.grid_pitch_row_fs = np.zeros(2, dtype=float)

        self.image_center_x: float = 0.0
        self.image_center_y: float = 0.0

        # Z-tilt: how Z changes per tile movement (full-steps)
        # dz = dz_per_col * dcol + dz_per_row * drow + dz_per_tile
        self.dz_per_col: float = 0.0
        self.dz_per_row: float = 0.0
        self.dz_per_tile: float = 0.0
        self.z_tilt_valid: bool = False

        self.valid: bool = False

    def pixel_to_motor(self, dpx_x: float, dpx_y: float) -> Tuple[float, float]:
        d = self.px_to_motor @ np.array([dpx_x, dpx_y])
        return (float(d[0]), float(d[1]))

    def motor_to_pixel(self, dfs_x: float, dfs_y: float) -> Tuple[float, float]:
        d = self.motor_to_px @ np.array([dfs_x, dfs_y])
        return (float(d[0]), float(d[1]))

    def grid_delta_to_pixel(self, dcol: int, drow: int) -> Tuple[float, float]:
        d = dcol * self.grid_pitch_col_px + drow * self.grid_pitch_row_px
        return (float(d[0]), float(d[1]))

    def to_dict(self) -> dict:
        return {
            'motor_to_px': self.motor_to_px.tolist(),
            'px_to_motor': self.px_to_motor.tolist(),
            'grid_pitch_col_px': self.grid_pitch_col_px.tolist(),
            'grid_pitch_row_px': self.grid_pitch_row_px.tolist(),
            'grid_pitch_col_fs': self.grid_pitch_col_fs.tolist(),
            'grid_pitch_row_fs': self.grid_pitch_row_fs.tolist(),
            'image_center_x': self.image_center_x,
            'image_center_y': self.image_center_y,
            'dz_per_col': self.dz_per_col,
            'dz_per_row': self.dz_per_row,
            'dz_per_tile': self.dz_per_tile,
            'z_tilt_valid': self.z_tilt_valid,
            'valid': self.valid,
        }

    @staticmethod
    def from_dict(d: dict) -> 'CalibrationResult':
        r = CalibrationResult()
        r.motor_to_px = np.array(d['motor_to_px'], dtype=float)
        r.px_to_motor = np.array(d['px_to_motor'], dtype=float)
        r.grid_pitch_col_px = np.array(d['grid_pitch_col_px'], dtype=float)
        r.grid_pitch_row_px = np.array(d['grid_pitch_row_px'], dtype=float)
        r.grid_pitch_col_fs = np.array(d['grid_pitch_col_fs'], dtype=float)
        r.grid_pitch_row_fs = np.array(d['grid_pitch_row_fs'], dtype=float)
        r.image_center_x = d['image_center_x']
        r.image_center_y = d['image_center_y']
        r.dz_per_col = d.get('dz_per_col', 0.0)
        r.dz_per_row = d.get('dz_per_row', 0.0)
        r.dz_per_tile = d.get('dz_per_tile', 0.0)
        r.z_tilt_valid = d.get('z_tilt_valid', False)
        r.valid = d.get('valid', False)
        return r

    def __repr__(self):
        if not self.valid:
            return "CalibrationResult(invalid)"
        m = self.px_to_motor
        return (f"CalibrationResult(\n"
                f"  px→motor: Xm = {m[0,0]:+.5f}*px_x {m[0,1]:+.5f}*px_y\n"
                f"            Ym = {m[1,0]:+.5f}*px_x {m[1,1]:+.5f}*px_y\n"
                f"  grid/col fs=({self.grid_pitch_col_fs[0]:.2f}, {self.grid_pitch_col_fs[1]:.2f})\n"
                f"  grid/row fs=({self.grid_pitch_row_fs[0]:.2f}, {self.grid_pitch_row_fs[1]:.2f}))")


# ── Persistence ──────────────────────────────────────────────────

def _cal_path(config) -> str:
    cfg_path = getattr(config, 'cfg_path', '')
    if cfg_path:
        return os.path.join(os.path.dirname(cfg_path), 'calibration.json')
    return 'calibration.json'


def save_calibration(result: CalibrationResult, config) -> str:
    path = _cal_path(config)
    # Load existing data to preserve extra keys (e.g. orientation)
    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            pass
    # Merge: calibration result overwrites its keys, extra keys preserved
    existing.update(result.to_dict())
    with open(path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"[StageCal] Saved to {path}")
    return path


def load_calibration(config) -> Optional[CalibrationResult]:
    path = _cal_path(config)
    if not os.path.exists(path):
        print(f"[StageCal] No saved calibration at {path}")
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        r = CalibrationResult.from_dict(d)
        print(f"[StageCal] Loaded from {path}: {r}")
        return r
    except Exception as e:
        print(f"[StageCal] Failed to load {path}: {e}")
        return None


# ── Calibration Controller ────────────────────────────────────────

class StageCalibration(QObject):

    progress = pyqtSignal(str, str)
    finished = pyqtSignal(bool, str, object)
    detection_preview = pyqtSignal(str, object)

    SETTLE_MS = 400
    DISCARD_FRAMES = 3
    PROBE_DISTANCE_FS = 10.0
    DETECTION_THRESHOLD = 20

    PROBE_BY_MRES = {
        1: 30.0, 2: 25.0, 4: 20.0, 8: 18.0, 16: 10.0, 32: 10.0,
    }

    MAX_MOVE_FS = 4000.0

    # Cap-bounded reference-detection: try MAX_REFERENCE_RETRIES AF+recapture
    # rounds before aborting the calibration. Each retry settles + recaptures.
    MAX_REFERENCE_RETRIES = 5
    REFERENCE_RETRY_DELAY_MS = 300

    def __init__(self, move_to_ctrl, camera_preview, mc, motors_config,
                 config=None, detection_threshold: int = 50, af_controller=None, parent=None):
        super().__init__(parent)
        self.move_to_ctrl = move_to_ctrl
        self.camera_preview = camera_preview
        self.mc = mc
        self.MOTORS = motors_config
        self.config = config
        self.DETECTION_THRESHOLD = detection_threshold
        self.af_controller = af_controller

        self._active = False
        self._stop_requested = False
        self._phase = 'idle'

        self._ref_x_fs = 0.0
        self._ref_y_fs = 0.0
        self._ref_markers: List[Dict] = []
        self._camera_settings = {}

        self._x_probe_dpx = np.zeros(2)
        self._y_probe_dpx = np.zeros(2)

        # Refinement
        self._refine_targets = []
        self._refine_idx = 0
        self._refine_start_x = 0.0
        self._refine_start_y = 0.0
        self._anchor_col = 0
        self._anchor_row = 0

        # Continuous learning: record the start detection for comparison
        self._move_start_col = 0
        self._move_start_row = 0
        self._is_first_fine_correction = False

        self._pending_move = None
        self.result = CalibrationResult()

        if config is not None:
            saved = load_calibration(config)
            if saved and saved.valid:
                self.result = saved

        self.move_to_ctrl.finished.connect(self._on_move_finished)

        if self.af_controller:
            self.af_controller.finished.connect(self._on_quick_af_finished)

    # ── Helpers ──

    def _current_z(self) -> float:
        return float(getattr(self.MOTORS, "z_current_position_full_step", 0.0))

    def _get_ideal_target_px(self) -> np.ndarray:
        cx, cy = self.result.image_center_x, self.result.image_center_y
        if np.linalg.norm(self.result.grid_pitch_col_px) < 1.0:
            return np.array([cx, cy])
        return np.array([cx, cy]) - 0.5 * self.result.grid_pitch_col_px - 0.5 * self.result.grid_pitch_row_px

    def _request_capture_with_af(self, next_phase, func_name, msg):
        if self.af_controller:
            self._phase = f'af_wait_{next_phase}'
            self._af_next_func_name = func_name
            self.progress.emit(self._phase, f"Quick AF: {msg}")
            self.af_controller.start_quick(self._camera_settings)
        else:
            self._phase = next_phase
            func = getattr(self, func_name)
            QTimer.singleShot(self.SETTLE_MS, func)

    def _on_quick_af_finished(self, success, message):
        if not self._active:
            return
        if not success and "Cancelled" not in message:
            print(f"[StageCal] Quick AF failed: {message}. Proceeding anyway...")

        # Z-tilt survey AF completed — record Z and advance
        if self._phase == 'z_tilt_af':
            self._on_ztilt_af_done(success)
            return

        if self._phase.startswith('af_wait_'):
            self._phase = self._phase.replace('af_wait_', '')
            func = getattr(self, self._af_next_func_name, None)
            if func:
                QTimer.singleShot(100, func)

    # ── Public API ──

    def is_active(self):
        return self._active

    def get_result(self):
        return self.result

    def start(self, camera_settings, probe_distance_fs=None,
              detection_threshold=None):
        if self._active:
            return
        if not self.camera_preview.is_running():
            self.finished.emit(False, "Start live preview first.", None)
            return

        self._active = True
        self._stop_requested = False
        self._camera_settings = camera_settings

        if probe_distance_fs is not None:
            self.PROBE_DISTANCE_FS = float(probe_distance_fs)
        else:
            try:
                mres_x = int(self.mc.get_axis_microstepping("x"))
            except Exception:
                mres_x = 1
            self.PROBE_DISTANCE_FS = self.PROBE_BY_MRES.get(mres_x, 10.0)

        if detection_threshold is not None:
            self.DETECTION_THRESHOLD = int(detection_threshold)

        self._ref_x_fs = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
        self._ref_y_fs = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))

        print(f"[StageCal] Start at X={self._ref_x_fs:.1f} Y={self._ref_y_fs:.1f}, "
              f"probe={self.PROBE_DISTANCE_FS}fs")

        self._request_capture_with_af('ref_capture', '_capture_reference',
                                      'Capturing reference frame...')

    def stop(self, reason="Cancelled"):
        if not self._active:
            return
        self._stop_requested = True
        self._active = False
        self._phase = 'idle'
        self._pending_move = None
        print(f"[StageCal] Stopped: {reason}")
        self.finished.emit(False, reason, None)

    # ═══════════════════════════════════════════════════════════════
    # Move-to-Feature
    # ═══════════════════════════════════════════════════════════════


    def move_to_feature(self, target_col, target_row, camera_settings,
                        detection_threshold=None):
        if self._active:
            self.finished.emit(False, "Already active.", None)
            return
        if not self.result.valid:
            self.finished.emit(False, "Calibration required first.", None)
            return
        if not self.camera_preview.is_running():
            self.finished.emit(False, "Start live preview first.", None)
            return

        self._active = True
        self._stop_requested = False
        self._camera_settings = camera_settings
        self._move_target_col = target_col
        self._move_target_row = target_row
        if detection_threshold is not None:
            self.DETECTION_THRESHOLD = detection_threshold

        self.progress.emit('detecting',
                           f'Detecting before move to ({target_col}, {target_row})...')

        # CHANGED: Replaced _request_capture_with_af with a direct timer.
        # We assume the camera is already in focus from the previous position.
        self._phase = 'detecting'
        QTimer.singleShot(self.SETTLE_MS, self._do_move_detection)

    def _do_move_detection(self):
        if self._stop_requested:
            return

        target_col = self._move_target_col
        target_row = self._move_target_row

        frame = self.camera_preview.flush_and_capture(
            self._camera_settings, discard=self.DISCARD_FRAMES)
        if frame is None:
            self._active = False
            self.finished.emit(False, "Capture failed", None)
            return

        markers = detect_markers(frame, threshold=self.DETECTION_THRESHOLD, debug=True)
        if not markers:
            p = save_failed_frame(frame, "cal_initial")
            self._initial_retries = getattr(self, '_initial_retries', 0) + 1
            if self._initial_retries < self.MAX_REFERENCE_RETRIES:
                print(f"[StageCal] No markers at initial — retry "
                      f"{self._initial_retries}/{self.MAX_REFERENCE_RETRIES} "
                      f"(saved {p})")
                QTimer.singleShot(self.REFERENCE_RETRY_DELAY_MS, self._do_move_detection)
                return
            print(f"[StageCal] No markers after {self._initial_retries} retries — saved {p}")
            self._initial_retries = 0
            self._active = False
            self.finished.emit(False, "No markers visible.", None)
            return
        self._initial_retries = 0

        markers = self._filter_suspicious_markers(markers)

        cx, cy = self.result.image_center_x, self.result.image_center_y
        ref_m = min(markers, key=lambda m:
                    (m['center'][0] - cx)**2 + (m['center'][1] - cy)**2)

        current_col = ref_m['col_id']
        current_row = ref_m['row_id']

        # Save for continuous learning
        self._move_start_col = current_col
        self._move_start_row = current_row

        dcol = target_col - current_col
        drow = target_row - current_row

        # Use observed pixel pitch from current markers
        obs_col_px, obs_row_px = measure_pixel_pitch_from_markers(markers)
        col_px = obs_col_px if obs_col_px is not None else self.result.grid_pitch_col_px
        row_px = obs_row_px if obs_row_px is not None else self.result.grid_pitch_row_px

        grid_dpx = dcol * col_px + drow * row_px
        ideal_target_px = np.array([cx, cy]) - 0.5 * col_px - 0.5 * row_px
        offset_px = ideal_target_px - np.array([ref_m['center'][0], ref_m['center'][1]])
        total_dpx = -grid_dpx + offset_px

        motor_delta = self.result.px_to_motor @ total_dpx

        cur_x = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
        cur_y = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))

        target_x = cur_x + motor_delta[0]
        target_y = cur_y + motor_delta[1]
        move_mag = float(np.linalg.norm(motor_delta))
        safe = move_mag <= self.MAX_MOVE_FS

        lines = []
        lines.append(f"Detected {len(markers)} marker(s):")
        for m in markers:
            dist = math.sqrt((m['center'][0] - cx)**2 + (m['center'][1] - cy)**2)
            tag = "  [INFERRED]" if m.get('_inferred') else ""
            lines.append(
                f"  ({m['col_id']:2d}, {m['row_id']:2d}) "
                f"px=({m['center'][0]:.0f}, {m['center'][1]:.0f})  "
                f"conf={m['confidence']:.3f}  d={dist:.0f}{tag}")
        lines.append(f"\nRef: ({current_col}, {current_row}) "
                     f"at px ({ref_m['center'][0]:.0f}, {ref_m['center'][1]:.0f})")
        lines.append(f"Target: ({target_col}, {target_row}), Δ=({dcol}, {drow})")
        lines.append(f"Grid px shift: ({grid_dpx[0]:.1f}, {grid_dpx[1]:.1f})")
        lines.append(f"Stage px shift: ({total_dpx[0]:.1f}, {total_dpx[1]:.1f})")
        lines.append(f"Motor: ΔX={motor_delta[0]:.1f}, ΔY={motor_delta[1]:.1f} fs")
        lines.append(f"Now:  X={cur_x:.1f}, Y={cur_y:.1f}")
        lines.append(f"Goal: X={target_x:.1f}, Y={target_y:.1f}")
        lines.append(f"Magnitude: {move_mag:.1f} fs")
        if not safe:
            lines.append(f"\n⚠ Exceeds limit {self.MAX_MOVE_FS:.0f} fs!")

        info_str = "\n".join(lines)
        print(f"\n[StageCal] Preview:\n{info_str}\n")

        self._pending_move = {
            'target_x': target_x, 'target_y': target_y,
            'target_col': target_col, 'target_row': target_row,
            'magnitude': move_mag, 'safe': safe,
        }

        print("[StageCal] Emitting detection_preview...")
        self.detection_preview.emit(info_str, markers)

    def confirm_move(self):
        if not self._active or self._pending_move is None:
            self.finished.emit(False, "No pending move.", None)
            return
        pm = self._pending_move
        if not pm['safe']:
            self._active = False
            self._pending_move = None
            self._phase = 'idle'
            self.finished.emit(False, f"Too large ({pm['magnitude']:.0f}fs).", None)
            return

        print(f"[StageCal] Confirmed → X={pm['target_x']:.1f} Y={pm['target_y']:.1f}")
        self._fine_target_col = pm['target_col']
        self._fine_target_row = pm['target_row']
        self._final_target_x = pm['target_x']
        self._final_target_y = pm['target_y']
        self._pending_move = None
        self._is_first_fine_correction = True

        cur_x = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
        cur_y = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))

        # Move X first, then Y separately to reduce simultaneous-axis errors
        need_x = abs(self._final_target_x - cur_x) > 0.5
        need_y = abs(self._final_target_y - cur_y) > 0.5

        if need_x:
            self._phase = 'move_to_feature_x'
            self.progress.emit('moving',
                               f"Moving X toward ({pm['target_col']}, {pm['target_row']})...")
            self.move_to_ctrl.start(self._final_target_x, cur_y, self._current_z())
        elif need_y:
            self._phase = 'move_to_feature_y'
            self.progress.emit('moving',
                               f"Moving Y toward ({pm['target_col']}, {pm['target_row']})...")
            self.move_to_ctrl.start(cur_x, self._final_target_y, self._current_z())
        else:
            # Already at target, just do fine correction
            self._request_capture_with_af('fine_correction', '_fine_correct_capture',
                                          'Fine-correcting...')

    def cancel_move(self):
        self._pending_move = None
        self._active = False
        self._phase = 'idle'
        self.finished.emit(False, "Move cancelled.", None)

    # ═══════════════════════════════════════════════════════════════
    # Calibration Phase 1: Local probes
    # ═══════════════════════════════════════════════════════════════

    def _capture_reference(self):
        if self._stop_requested:
            return
        frame = self.camera_preview.flush_and_capture(
            self._camera_settings, discard=self.DISCARD_FRAMES)
        if frame is None:
            self.stop("Reference capture failed")
            return

        h, w = frame.shape[:2]
        self.result.image_center_x = w / 2.0
        self.result.image_center_y = h / 2.0

        markers = detect_markers(frame, threshold=self.DETECTION_THRESHOLD, debug=True)
        if not markers:
            p = save_failed_frame(frame, "cal_reference")
            self._ref_retries = getattr(self, '_ref_retries', 0) + 1
            if self._ref_retries < self.MAX_REFERENCE_RETRIES:
                print(f"[StageCal] No markers at reference — retry "
                      f"{self._ref_retries}/{self.MAX_REFERENCE_RETRIES} "
                      f"(saved {p})")
                QTimer.singleShot(self.REFERENCE_RETRY_DELAY_MS, self._capture_reference)
                return
            print(f"[StageCal] No markers at reference after {self._ref_retries} retries — saved {p}")
            self._ref_retries = 0
            self.stop("No markers at reference position.")
            return
        self._ref_retries = 0

        self._ref_markers = markers
        print(f"[StageCal] Reference: {len(markers)} marker(s)")
        for m in markers:
            tag = '  [inf]' if m.get('_inferred') else ''
            print(f"  ({m['col_id']}, {m['row_id']}) "
                  f"px=({m['center'][0]:.1f}, {m['center'][1]:.1f}){tag}")

        self._measure_grid_pitch_px(markers)

        self._phase = 'x_probe_move'
        self.progress.emit('x_probe', f'Probing motor X (+{self.PROBE_DISTANCE_FS:.0f}fs)...')
        self.move_to_ctrl.start(
            self._ref_x_fs + self.PROBE_DISTANCE_FS,
            self._ref_y_fs, self._current_z())

    def _capture_probe(self, axis_label):
        if self._stop_requested:
            return None
        frame = self.camera_preview.flush_and_capture(
            self._camera_settings, discard=self.DISCARD_FRAMES)
        if frame is None:
            self.stop(f"{axis_label} probe capture failed")
            return None

        markers = detect_markers(frame, threshold=self.DETECTION_THRESHOLD, debug=True)
        if not markers:
            p = save_failed_frame(frame, f"cal_probe_{axis_label}")
            self._probe_retries = getattr(self, '_probe_retries', 0) + 1
            if self._probe_retries < self.MAX_REFERENCE_RETRIES:
                print(f"[StageCal] No markers after {axis_label} probe — retry "
                      f"{self._probe_retries}/{self.MAX_REFERENCE_RETRIES} "
                      f"(saved {p})")
                # Re-dispatch this same probe-capture after a settle delay.
                cb = getattr(self, f"_capture_{axis_label.lower()}_probe", None)
                if cb:
                    QTimer.singleShot(self.REFERENCE_RETRY_DELAY_MS, cb)
                    return None
            print(f"[StageCal] No markers after {axis_label} probe "
                  f"({self._probe_retries} retries) — saved {p}")
            self._probe_retries = 0
            self.stop(f"No markers after {axis_label} probe.")
            return None
        self._probe_retries = 0

        matched = self._match_markers(self._ref_markers, markers)
        if not matched:
            self.stop(f"{axis_label} probe: no matching markers.")
            return None

        dx_list, dy_list = [], []
        for ref_m, probe_m in matched:
            dx = probe_m['center'][0] - ref_m['center'][0]
            dy = probe_m['center'][1] - ref_m['center'][1]
            dx_list.append(dx)
            dy_list.append(dy)

        avg = np.array([float(np.mean(dx_list)), float(np.mean(dy_list))])
        mag = float(np.linalg.norm(avg))
        if mag < 5.0:
            self.stop(f"{axis_label} probe: shift too small ({mag:.1f}px)")
            return None

        print(f"[StageCal] Motor +{axis_label} → px ({avg[0]:.1f}, {avg[1]:.1f})")
        return avg

    def _capture_x_probe(self):
        result = self._capture_probe("X")
        if result is None:
            return
        self._x_probe_dpx = result
        self._phase = 'x_return'
        self.progress.emit('x_return', 'Returning...')
        self.move_to_ctrl.start(self._ref_x_fs, self._ref_y_fs, self._current_z())

    def _capture_y_probe(self):
        result = self._capture_probe("Y")
        if result is None:
            return
        self._y_probe_dpx = result
        self._phase = 'y_return'
        self.progress.emit('y_return', 'Returning...')
        self.move_to_ctrl.start(self._ref_x_fs, self._ref_y_fs, self._current_z())

    # ═══════════════════════════════════════════════════════════════
    # Calibration Phase 2: Build initial matrix
    # ═══════════════════════════════════════════════════════════════

    def _build_initial_matrix(self):
        probe = self.PROBE_DISTANCE_FS
        self.result.motor_to_px[:, 0] = self._x_probe_dpx / probe
        self.result.motor_to_px[:, 1] = self._y_probe_dpx / probe

        det = np.linalg.det(self.result.motor_to_px)
        if abs(det) < 1e-10:
            self.stop("Motor axes appear parallel in image.")
            return False

        self.result.px_to_motor = np.linalg.inv(self.result.motor_to_px)

        if np.linalg.norm(self.result.grid_pitch_col_px) > 1.0:
            self.result.grid_pitch_col_fs = (
                self.result.px_to_motor @ self.result.grid_pitch_col_px)
        if np.linalg.norm(self.result.grid_pitch_row_px) > 1.0:
            self.result.grid_pitch_row_fs = (
                self.result.px_to_motor @ self.result.grid_pitch_row_px)

        self.result.valid = True
        self._print_result("Initial Probe Result")
        return True

    # ═══════════════════════════════════════════════════════════════
    # Calibration Phase 3: Sequential Live-Refinement
    # ═══════════════════════════════════════════════════════════════

    def _start_refinement(self):
        if not self._ref_markers:
            self._finalize()
            return

        ref_m = self._ref_markers[0]
        self._anchor_col = ref_m['col_id']
        self._anchor_row = ref_m['row_id']

        d1_c = 1 if self._anchor_col + 1 <= 31 else -1
        d1_r = 1 if self._anchor_row + 1 <= 31 else -1

        self._refine_targets = [
            (self._anchor_col + d1_c, self._anchor_row),       # +1 col
            (self._anchor_col, self._anchor_row + d1_r),       # +1 row
            (self._anchor_col + d1_c, self._anchor_row + d1_r) # +1 diagonal
        ]
        self._refine_idx = 0

        self._refine_start_x = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
        self._refine_start_y = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))

        print(f"[StageCal] Refinement targets: {self._refine_targets}")
        self._move_to_refine_target()

    def _move_to_refine_target(self):
        if self._stop_requested:
            return
        if self._refine_idx >= len(self._refine_targets):
            self._phase = 'refine_return_final'
            self.progress.emit('refine_return', 'Returning to start...')
            self.move_to_ctrl.start(
                self._refine_start_x, self._refine_start_y, self._current_z())
            return

        target_col, target_row = self._refine_targets[self._refine_idx]
        self.progress.emit('refining',
                           f'Refine {self._refine_idx+1}/{len(self._refine_targets)}: '
                           f'({target_col}, {target_row})...')

        frame = self.camera_preview.flush_and_capture(
            self._camera_settings, discard=self.DISCARD_FRAMES)
        if frame is None:
            self._refine_idx += 1
            self._move_to_refine_target()
            return

        markers = detect_markers(frame, threshold=self.DETECTION_THRESHOLD)
        if not markers:
            p = save_failed_frame(frame, "cal_refine_predict")
            print(f"[StageCal] Refine predict: no markers — saved {p}")
            self._refine_idx += 1
            self._move_to_refine_target()
            return

        markers = self._filter_suspicious_markers(markers)
        cx, cy = self.result.image_center_x, self.result.image_center_y
        ref_m = min(markers, key=lambda m:
                    (m['center'][0] - cx)**2 + (m['center'][1] - cy)**2)

        dcol = target_col - ref_m['col_id']
        drow = target_row - ref_m['row_id']

        # Use observed pixel pitch
        obs_col_px, obs_row_px = measure_pixel_pitch_from_markers(markers)
        col_px = obs_col_px if obs_col_px is not None else self.result.grid_pitch_col_px
        row_px = obs_row_px if obs_row_px is not None else self.result.grid_pitch_row_px

        grid_dpx = dcol * col_px + drow * row_px
        ideal_target_px = np.array([cx, cy]) - 0.5 * col_px - 0.5 * row_px
        offset_px = ideal_target_px - np.array([ref_m['center'][0], ref_m['center'][1]])
        total_dpx = -grid_dpx + offset_px
        motor_delta = self.result.px_to_motor @ total_dpx

        cur_x = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
        cur_y = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))

        self._refine_predicted_motor = np.array([cur_x + motor_delta[0],
                                                 cur_y + motor_delta[1]])
        self._refine_current_target = (target_col, target_row)

        self._phase = 'refine_move'
        self.move_to_ctrl.start(
            self._refine_predicted_motor[0],
            self._refine_predicted_motor[1],
            self._current_z())

    def _capture_refine(self):
        if self._stop_requested:
            return

        frame = self.camera_preview.flush_and_capture(
            self._camera_settings, discard=self.DISCARD_FRAMES)
        if frame is None:
            self._refine_idx += 1
            self._move_to_refine_target()
            return

        markers = detect_markers(frame, threshold=self.DETECTION_THRESHOLD)
        if not markers:
            p = save_failed_frame(frame, "cal_refine_capture")
            print(f"[StageCal] Refine capture: no markers — saved {p}")
            self._refine_idx += 1
            self._move_to_refine_target()
            return

        markers = self._filter_suspicious_markers(markers)
        target_col, target_row = self._refine_current_target

        target_m = None
        for m in markers:
            if m['col_id'] == target_col and m['row_id'] == target_row:
                target_m = m
                break

        if target_m is None:
            closest = min(markers, key=lambda m:
                          (m['col_id'] - target_col)**2 +
                          (m['row_id'] - target_row)**2)
            target_m = closest

        ideal_target_px = self._get_ideal_target_px()
        dcol_from_target = target_m['col_id'] - target_col
        drow_from_target = target_m['row_id'] - target_row

        # Use observed pixel pitch
        obs_col_px, obs_row_px = measure_pixel_pitch_from_markers(markers)
        col_px = obs_col_px if obs_col_px is not None else self.result.grid_pitch_col_px
        row_px = obs_row_px if obs_row_px is not None else self.result.grid_pitch_row_px

        expected_px = ideal_target_px + \
                      (dcol_from_target * col_px) + \
                      (drow_from_target * row_px)

        actual_px = np.array([target_m['center'][0], target_m['center'][1]])

        # How far off: positive means marker is to the right/below expected
        # This means our grid pitch UNDER-predicted the spacing
        miss_px = actual_px - expected_px

        dcol_from_anchor = target_col - self._anchor_col
        drow_from_anchor = target_row - self._anchor_row

        print(f"[StageCal] Refine ({target_col},{target_row}): "
              f"miss=({miss_px[0]:.1f}, {miss_px[1]:.1f})px "
              f"Δanchor=({dcol_from_anchor},{drow_from_anchor})")

        # LIVE RECALIBRATION (with safety caps)
        # miss_px > 0 means marker ended up further than predicted → pitch too small → ADD.

        if dcol_from_anchor != 0 and drow_from_anchor == 0:
            raw = miss_px / dcol_from_anchor
            max_c = np.linalg.norm(self.result.grid_pitch_col_px) * 0.15
            if max_c > 0 and np.linalg.norm(raw) > max_c:
                raw = raw / np.linalg.norm(raw) * max_c
                print(f"[StageCal] Refine: capped col correction at {max_c:.0f}px")
            self.result.grid_pitch_col_px += raw
            self.result.grid_pitch_col_fs = self.result.px_to_motor @ self.result.grid_pitch_col_px
            print(f"[StageCal] REFINE col: +({raw[0]:.1f}, {raw[1]:.1f})/step "
                  f"→ ({self.result.grid_pitch_col_px[0]:.1f}, {self.result.grid_pitch_col_px[1]:.1f})")

        elif drow_from_anchor != 0 and dcol_from_anchor == 0:
            raw = miss_px / drow_from_anchor
            max_c = np.linalg.norm(self.result.grid_pitch_row_px) * 0.15
            if max_c > 0 and np.linalg.norm(raw) > max_c:
                raw = raw / np.linalg.norm(raw) * max_c
                print(f"[StageCal] Refine: capped row correction at {max_c:.0f}px")
            self.result.grid_pitch_row_px += raw
            self.result.grid_pitch_row_fs = self.result.px_to_motor @ self.result.grid_pitch_row_px
            print(f"[StageCal] REFINE row: +({raw[0]:.1f}, {raw[1]:.1f})/step "
                  f"→ ({self.result.grid_pitch_row_px[0]:.1f}, {self.result.grid_pitch_row_px[1]:.1f})")

        elif dcol_from_anchor != 0 and drow_from_anchor != 0:
            col_raw = (miss_px / 2.0) / dcol_from_anchor
            row_raw = (miss_px / 2.0) / drow_from_anchor
            max_col = np.linalg.norm(self.result.grid_pitch_col_px) * 0.10
            max_row = np.linalg.norm(self.result.grid_pitch_row_px) * 0.10
            if max_col > 0 and np.linalg.norm(col_raw) > max_col:
                col_raw = col_raw / np.linalg.norm(col_raw) * max_col
            if max_row > 0 and np.linalg.norm(row_raw) > max_row:
                row_raw = row_raw / np.linalg.norm(row_raw) * max_row
            self.result.grid_pitch_col_px += col_raw
            self.result.grid_pitch_row_px += row_raw
            self.result.grid_pitch_col_fs = self.result.px_to_motor @ self.result.grid_pitch_col_px
            self.result.grid_pitch_row_fs = self.result.px_to_motor @ self.result.grid_pitch_row_px
            print(f"[StageCal] REFINE diag: col +=({col_raw[0]:.1f}, {col_raw[1]:.1f}), "
                  f"row +=({row_raw[0]:.1f}, {row_raw[1]:.1f})")

        self._refine_idx += 1
        self._move_to_refine_target()

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Z-Tilt Survey
    # Visit 9 tiles in a 5x5 region, AF at each, fit a tilt plane.
    # ═══════════════════════════════════════════════════════════════

    def _start_z_tilt_survey(self):
        """Begin Z-tilt survey: visit 9 spread-out tiles, AF at each."""
        if self._stop_requested:
            return

        # Use the anchor position from refinement as our reference
        center_col = self._anchor_col
        center_row = self._anchor_row

        # 9 survey points in a 3x3 grid (±1 tile) — single-tile moves only
        offsets = [
            (0, 0),            # center (reference point)
            (-1, 0), (1, 0),   # left/right
            (0, -1), (0, 1),   # up/down
            (-1, -1), (1, -1), # corners
            (-1, 1), (1, 1),
        ]

        self._ztilt_tiles = []
        for dc, dr in offsets:
            c = center_col + dc
            r = center_row + dr
            if 0 <= c <= 30 and 0 <= r <= 30:
                self._ztilt_tiles.append((c, r))

        self._ztilt_idx = 0
        self._ztilt_data = []  # list of (col, row, z_fs)
        self._ztilt_start_x = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
        self._ztilt_start_y = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))

        print(f"[StageCal] Z-tilt survey: {len(self._ztilt_tiles)} tiles around ({center_col},{center_row})")
        self._advance_ztilt_survey()

    def _advance_ztilt_survey(self):
        """Move to next survey tile or finish if done."""
        if self._stop_requested:
            return

        if self._ztilt_idx >= len(self._ztilt_tiles):
            # All tiles surveyed — fit the plane and return to start
            self._fit_z_tilt_plane()
            self._phase = 'z_tilt_return'
            self.progress.emit('z_tilt', 'Returning to start...')
            self.move_to_ctrl.start(
                self._ztilt_start_x, self._ztilt_start_y, self._current_z())
            return

        target_col, target_row = self._ztilt_tiles[self._ztilt_idx]
        self.progress.emit('z_tilt',
            f'Z-tilt: moving to ({target_col},{target_row}) '
            f'[{self._ztilt_idx+1}/{len(self._ztilt_tiles)}]')

        # Detect current position and compute move
        frame = self.camera_preview.flush_and_capture(
            self._camera_settings, discard=self.DISCARD_FRAMES)
        if frame is None:
            print(f"[StageCal] Z-tilt: capture failed, skipping tile")
            self._ztilt_idx += 1
            QTimer.singleShot(100, self._advance_ztilt_survey)
            return

        markers = detect_markers(frame, threshold=self.DETECTION_THRESHOLD)
        if not markers:
            p = save_failed_frame(frame, "cal_ztilt")
            print(f"[StageCal] Z-tilt: no markers, skipping tile — saved {p}")
            self._ztilt_idx += 1
            QTimer.singleShot(100, self._advance_ztilt_survey)
            return

        markers = self._filter_suspicious_markers(markers)
        cx, cy = self.result.image_center_x, self.result.image_center_y
        ref_m = min(markers, key=lambda m:
                    (m['center'][0] - cx)**2 + (m['center'][1] - cy)**2)

        dcol = target_col - ref_m['col_id']
        drow = target_row - ref_m['row_id']

        # Use observed pixel pitch
        obs_col_px, obs_row_px = measure_pixel_pitch_from_markers(markers)
        col_px = obs_col_px if obs_col_px is not None else self.result.grid_pitch_col_px
        row_px = obs_row_px if obs_row_px is not None else self.result.grid_pitch_row_px

        grid_dpx = dcol * col_px + drow * row_px
        ideal_target_px = np.array([cx, cy]) - 0.5 * col_px - 0.5 * row_px
        offset_px = ideal_target_px - np.array([ref_m['center'][0], ref_m['center'][1]])
        total_dpx = -grid_dpx + offset_px
        motor_delta = self.result.px_to_motor @ total_dpx

        cur_x = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
        cur_y = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))

        self._phase = 'z_tilt_move'
        self.move_to_ctrl.start(
            cur_x + motor_delta[0], cur_y + motor_delta[1], self._current_z())

    def _on_ztilt_af_done(self, success):
        """Called when AF finishes at a survey tile."""
        col, row = self._ztilt_tiles[self._ztilt_idx]
        z_fs = self._current_z()
        self._ztilt_data.append((col, row, z_fs))
        print(f"[StageCal] Z-tilt: ({col},{row}) -> Z={z_fs:.3f}fs (AF {'ok' if success else 'fail'})")

        self._ztilt_idx += 1
        QTimer.singleShot(100, self._advance_ztilt_survey)

    def _fit_z_tilt_plane(self):
        """Fit a plane z = dz_per_col * dcol + dz_per_row * drow + z0
        to the survey data using least-squares."""
        if len(self._ztilt_data) < 3:
            print(f"[StageCal] Z-tilt: only {len(self._ztilt_data)} points, need >= 3. Skipping.")
            return

        # Use center tile as reference (first point)
        ref_col, ref_row, ref_z = self._ztilt_data[0]

        # Build least-squares: for each point, (col - ref_col, row - ref_row) -> (z - ref_z)
        A = []
        b = []
        for col, row, z in self._ztilt_data:
            dc = col - ref_col
            dr = row - ref_row
            dz = z - ref_z
            A.append([dc, dr])
            b.append(dz)

        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        try:
            # Least squares: b = A @ [dz_per_col, dz_per_row]
            result, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            dz_per_col = float(result[0])
            dz_per_row = float(result[1])

            # dz_per_tile: cumulative drift per move, not measurable statically.
            # Set to 0 and let the spiral scan learn it online.
            dz_per_tile = 0.0

            self.result.dz_per_col = dz_per_col
            self.result.dz_per_row = dz_per_row
            self.result.dz_per_tile = dz_per_tile
            self.result.z_tilt_valid = True

            print(f"[StageCal] Z-tilt fit: dz/col={dz_per_col:.4f} dz/row={dz_per_row:.4f} "
                  f"dz/tile={dz_per_tile:.5f} ({len(self._ztilt_data)} points)")

            # Print survey data for debugging
            for col, row, z in self._ztilt_data:
                dc, dr = col - ref_col, row - ref_row
                pred_z = ref_z + dz_per_col * dc + dz_per_row * dr
                err = z - pred_z
                print(f"  ({col},{row}) Z={z:.3f} pred={pred_z:.3f} err={err:.3f}")

        except Exception as e:
            print(f"[StageCal] Z-tilt fit failed: {e}")
            self.result.z_tilt_valid = False

    # ═══════════════════════════════════════════════════════════════
    # Finalize
    # ═══════════════════════════════════════════════════════════════

    def _finalize(self):
        if self._stop_requested:
            return

        self._print_result("Final Calibration Matrix")

        if self.config is not None:
            save_calibration(self.result, self.config)

        self._active = False
        self._phase = 'idle'

        m2p = self.result.motor_to_px
        self.finished.emit(True,
                           f"Calibrated & saved. "
                           f"+X→px({m2p[0,0]:+.2f},{m2p[1,0]:+.2f}), "
                           f"+Y→px({m2p[0,1]:+.2f},{m2p[1,1]:+.2f})",
                           self.result)

    def _print_result(self, label):
        print(f"\n[StageCal] ══ {label} ══")
        m = self.result.motor_to_px
        print(f"  motor→px: +X → ({m[0,0]:+.3f}, {m[1,0]:+.3f})")
        print(f"            +Y → ({m[0,1]:+.3f}, {m[1,1]:+.3f})")
        p = self.result.px_to_motor
        print(f"  px→motor: +px_x → ({p[0,0]:+.5f}, {p[1,0]:+.5f})")
        print(f"            +px_y → ({p[0,1]:+.5f}, {p[1,1]:+.5f})")
        print(f"  grid/col px=({self.result.grid_pitch_col_px[0]:.1f}, "
              f"{self.result.grid_pitch_col_px[1]:.1f}) "
              f"fs=({self.result.grid_pitch_col_fs[0]:.2f}, "
              f"{self.result.grid_pitch_col_fs[1]:.2f})")
        print(f"  grid/row px=({self.result.grid_pitch_row_px[0]:.1f}, "
              f"{self.result.grid_pitch_row_px[1]:.1f}) "
              f"fs=({self.result.grid_pitch_row_fs[0]:.2f}, "
              f"{self.result.grid_pitch_row_fs[1]:.2f})")
        if self.result.z_tilt_valid:
            print(f"  Z-tilt: dz/col={self.result.dz_per_col:.4f} "
                  f"dz/row={self.result.dz_per_row:.4f} "
                  f"dz/tile={self.result.dz_per_tile:.5f}")
        print(f"[StageCal] ══════════════════════\n")

    # ── Move Finished Router ──

    def _on_move_finished(self, success, message):
        if not self._active or self._phase == 'idle':
            return
        if self._stop_requested:
            return
        if not success:
            self.stop(f"Move failed ({self._phase}): {message}")
            return

        if self._phase == 'x_probe_move':
            self._request_capture_with_af('x_probe_capture', '_capture_x_probe',
                                          'Measuring X probe...')
        elif self._phase == 'x_return':
            self._phase = 'y_probe_move'
            self.progress.emit('y_probe', f'Probing motor Y (+{self.PROBE_DISTANCE_FS:.0f}fs)...')
            self.move_to_ctrl.start(
                self._ref_x_fs,
                self._ref_y_fs + self.PROBE_DISTANCE_FS,
                self._current_z())

        elif self._phase == 'y_probe_move':
            self._request_capture_with_af('y_probe_capture', '_capture_y_probe',
                                          'Measuring Y probe...')
        elif self._phase == 'y_return':
            if self._build_initial_matrix():
                self._request_capture_with_af('refining', '_start_refinement',
                                              'Starting refinement...')
        elif self._phase == 'refine_move':
            self._request_capture_with_af('refine_capture', '_capture_refine',
                                          'Measuring landing...')
        elif self._phase == 'refine_return_final':
            self._phase = 'z_tilt_survey'
            self.progress.emit('z_tilt', 'Starting Z-tilt survey...')
            QTimer.singleShot(100, self._start_z_tilt_survey)

        elif self._phase == 'z_tilt_move':
            # Arrived at survey tile — settle then AF to measure Z
            self._phase = 'z_tilt_af'
            self.progress.emit('z_tilt',
                f'AF at survey tile {self._ztilt_idx+1}/{len(self._ztilt_tiles)}...')
            QTimer.singleShot(self.SETTLE_MS,
                lambda: self.af_controller.start_quick(self._camera_settings, mode='tight'))

        elif self._phase == 'z_tilt_return':
            # Returned to start after survey — finalize
            self._phase = 'finalizing'
            self.progress.emit('finalizing', 'Finalizing...')
            QTimer.singleShot(100, self._finalize)

        elif self._phase == 'move_to_feature_x':
            # X done, now move Y
            cur_x = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
            cur_y = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))
            need_y = abs(self._final_target_y - cur_y) > 0.5
            if need_y:
                self._phase = 'move_to_feature_y'
                self.progress.emit('moving', 'Moving Y...')
                self.move_to_ctrl.start(cur_x, self._final_target_y, self._current_z())
            else:
                # No Y needed, go straight to AF + fine correction
                self._request_capture_with_af('fine_correction', '_fine_correct_capture',
                                              'Fine-correcting...')

        elif self._phase == 'move_to_feature_y':
            # Y done, now AF + fine correction
            self._request_capture_with_af('fine_correction', '_fine_correct_capture',
                                          'Fine-correcting...')

    # ── Fine Correction (with Continuous Learning) ──

    def _fine_correct_capture(self):
        if self._stop_requested:
            return
        frame = self.camera_preview.flush_and_capture(
            self._camera_settings, discard=self.DISCARD_FRAMES)
        if frame is None:
            self._active = False
            self.finished.emit(False, "Fine capture failed.", self.result)
            return

        markers = detect_markers(frame, threshold=self.DETECTION_THRESHOLD)
        if not markers:
            p = save_failed_frame(frame, "cal_fine_start")
            print(f"[StageCal] Fine start: no markers — saved {p}")
            self._active = False
            self._phase = 'idle'
            self.finished.emit(True, "Coarse done (no markers for fine).", self.result)
            return

        markers = self._filter_suspicious_markers(markers)
        cx, cy = self.result.image_center_x, self.result.image_center_y

        # Find closest marker to center
        target_m = min(markers, key=lambda m:
                       (m['center'][0] - cx)**2 + (m['center'][1] - cy)**2)

        # Where SHOULD this marker be if calibration was perfect?
        ideal_target_px = self._get_ideal_target_px()
        dcol_from_target = target_m['col_id'] - self._fine_target_col
        drow_from_target = target_m['row_id'] - self._fine_target_row

        expected_px = ideal_target_px + \
                      (dcol_from_target * self.result.grid_pitch_col_px) + \
                      (drow_from_target * self.result.grid_pitch_row_px)

        actual_px = np.array([target_m['center'][0], target_m['center'][1]])

        # error_px points FROM actual TO expected (where we need to push it)
        error_px = expected_px - actual_px
        correction_fs = self.result.px_to_motor @ error_px

        dist_from_expected = float(np.linalg.norm(error_px))

        print(f"[Move] Fine using ({target_m['col_id']},{target_m['row_id']}): "
              f"err=({error_px[0]:.1f}, {error_px[1]:.1f})px "
              f"→ corr=({correction_fs[0]:.2f}, {correction_fs[1]:.2f})fs "
              f"dist={dist_from_expected:.0f}")

        # ── CONTINUOUS LEARNING ──
        # Only on the first fine correction after a coarse jump.
        # The miss tells us how our grid pitch prediction was wrong.
        if self._is_first_fine_correction:
            self._is_first_fine_correction = False

            dcol_move = self._fine_target_col - self._move_start_col
            drow_move = self._fine_target_row - self._move_start_row

            # miss_px = actual - expected
            # Positive means marker ended up further than expected → pitch was too small
            # So we need to ADD (miss / n_steps) to make pitch larger
            miss_px = actual_px - expected_px

            learning_rate = 0.20

            # Only update the axis that was primarily responsible.
            # For diagonal moves, we can't cleanly separate, so we skip
            # or use very conservative split.
            abs_dcol = abs(dcol_move)
            abs_drow = abs(drow_move)

            if abs_dcol > 0 and abs_drow == 0:
                # Pure col move: safe to attribute all error to col pitch
                raw_correction = miss_px / dcol_move
                correction = raw_correction * learning_rate
                # Safety cap: max 5% change per iteration
                max_change = np.linalg.norm(self.result.grid_pitch_col_px) * 0.05
                if np.linalg.norm(correction) > max_change:
                    correction = correction / np.linalg.norm(correction) * max_change
                self.result.grid_pitch_col_px += correction
                print(f"[StageCal] Col pitch += ({correction[0]:.2f}, {correction[1]:.2f}) "
                      f"→ ({self.result.grid_pitch_col_px[0]:.1f}, {self.result.grid_pitch_col_px[1]:.1f})")

            elif abs_drow > 0 and abs_dcol == 0:
                # Pure row move: safe to attribute all error to row pitch
                raw_correction = miss_px / drow_move
                correction = raw_correction * learning_rate
                max_change = np.linalg.norm(self.result.grid_pitch_row_px) * 0.05
                if np.linalg.norm(correction) > max_change:
                    correction = correction / np.linalg.norm(correction) * max_change
                self.result.grid_pitch_row_px += correction
                print(f"[StageCal] Row pitch += ({correction[0]:.2f}, {correction[1]:.2f}) "
                      f"→ ({self.result.grid_pitch_row_px[0]:.1f}, {self.result.grid_pitch_row_px[1]:.1f})")

            elif abs_dcol > 0 and abs_drow > 0:
                # Diagonal: use least-squares to separate col vs row error
                # miss_px ≈ dcol * col_err + drow * row_err
                # We can't solve 2 unknowns from 1 equation, so use a
                # proportional split weighted by how much each axis moved
                total_steps = abs_dcol + abs_drow
                col_weight = abs_dcol / total_steps
                row_weight = abs_drow / total_steps

                col_correction = (miss_px * col_weight / dcol_move) * learning_rate
                row_correction = (miss_px * row_weight / drow_move) * learning_rate

                # Tighter safety cap for diagonal (less confident)
                max_col = np.linalg.norm(self.result.grid_pitch_col_px) * 0.03
                max_row = np.linalg.norm(self.result.grid_pitch_row_px) * 0.03
                if np.linalg.norm(col_correction) > max_col:
                    col_correction = col_correction / np.linalg.norm(col_correction) * max_col
                if np.linalg.norm(row_correction) > max_row:
                    row_correction = row_correction / np.linalg.norm(row_correction) * max_row

                self.result.grid_pitch_col_px += col_correction
                self.result.grid_pitch_row_px += row_correction
                print(f"[StageCal] Diagonal: col += ({col_correction[0]:.2f}, {col_correction[1]:.2f}), "
                      f"row += ({row_correction[0]:.2f}, {row_correction[1]:.2f})")

            else:
                print(f"[StageCal] No movement detected, skipping update")

            # Recompute fs equivalents and save
            self.result.grid_pitch_col_fs = self.result.px_to_motor @ self.result.grid_pitch_col_px
            self.result.grid_pitch_row_fs = self.result.px_to_motor @ self.result.grid_pitch_row_px
            if self.config is not None:
                save_calibration(self.result, self.config)
        # ── END CONTINUOUS LEARNING ──

        # Always finish after first landing + learning. No second correction move.
        self._active = False
        self._phase = 'idle'
        msg = f"Arrived near ({self._fine_target_col}, {self._fine_target_row})."
        if target_m['col_id'] != self._fine_target_col or \
           target_m['row_id'] != self._fine_target_row:
            msg += f" (Visible: {target_m['col_id']},{target_m['row_id']})"
        msg += f" (err: {dist_from_expected:.0f}px)"
        self.finished.emit(True, msg, self.result)

    # ── Helpers ──

    def _match_markers(self, ref_markers, probe_markers):
        matched = []
        probe_by_id = {(m['col_id'], m['row_id']): m for m in probe_markers}
        for ref_m in ref_markers:
            key = (ref_m['col_id'], ref_m['row_id'])
            if key in probe_by_id:
                matched.append((ref_m, probe_by_id[key]))
        return matched

    def _filter_suspicious_markers(self, markers, expected_pos=None):
        if len(markers) < 2:
            return markers

        pitch_col = self.result.grid_pitch_col_px
        pitch_row = self.result.grid_pitch_row_px
        has_pitch = (np.linalg.norm(pitch_col) > 1.0 and
                     np.linalg.norm(pitch_row) > 1.0)

        if not has_pitch:
            return markers

        # Find best anchor
        if expected_pos is not None:
            # Use expected position to pick anchor: which marker's pixel position
            # best matches where its claimed ID should be on screen?
            cx, cy = self.result.image_center_x, self.result.image_center_y
            center_px = np.array([cx, cy])
            best_anchor = None
            best_score = float('inf')
            for m in markers:
                m_px = np.array([m['center'][0], m['center'][1]])
                # Where should this marker's ID appear relative to expected position?
                dcol = m['col_id'] - expected_pos[0]
                drow = m['row_id'] - expected_pos[1]
                predicted_px = center_px + (dcol - 0.5) * pitch_col + (drow - 0.5) * pitch_row
                score = np.linalg.norm(m_px - predicted_px)
                if score < best_score:
                    best_score = score
                    best_anchor = m
            # If best match is way off, fall back to consensus
            if best_score > 500:
                best_anchor = None
        else:
            best_anchor = None

        if best_anchor is None:
            # Consensus fallback (original logic)
            max_inliers = -1
            for m1 in markers:
                m1_px = np.array([m1['center'][0], m1['center'][1]])
                inlier_count = 0
                for m2 in markers:
                    m2_px = np.array([m2['center'][0], m2['center'][1]])
                    expected = m1_px + (m2['col_id'] - m1['col_id']) * pitch_col + \
                                       (m2['row_id'] - m1['row_id']) * pitch_row
                    if np.linalg.norm(expected - m2_px) < 100:
                        inlier_count += 1
                if inlier_count > max_inliers:
                    max_inliers = inlier_count
                    best_anchor = m1
            if max_inliers < 2 and len(markers) > 2:
                best_anchor = markers[0]


        filtered = []
        anchor_px = np.array([best_anchor['center'][0], best_anchor['center'][1]])
        A = np.column_stack([pitch_col, pitch_row])
        det = np.linalg.det(A)

        for m in markers:
            m_px = np.array([m['center'][0], m['center'][1]])
            expected = anchor_px + (m['col_id'] - best_anchor['col_id']) * pitch_col + \
                                   (m['row_id'] - best_anchor['row_id']) * pitch_row
            if np.linalg.norm(expected - m_px) < 100:
                filtered.append(m)
            else:
                if abs(det) > 1e-6:
                    delta_px = m_px - anchor_px
                    d = np.linalg.solve(A, delta_px)
                    inferred_col = int(best_anchor['col_id'] + round(d[0]))
                    inferred_row = int(best_anchor['row_id'] + round(d[1]))
                    residual = delta_px - (round(d[0]) * pitch_col + round(d[1]) * pitch_row)
                    if np.linalg.norm(residual) < 100:
                        # Check: does this rewrite move the ID closer to expected position?
                        if expected_pos is not None:
                            old_dist = abs(m['col_id'] - expected_pos[0]) + abs(m['row_id'] - expected_pos[1])
                            new_dist = abs(inferred_col - expected_pos[0]) + abs(inferred_row - expected_pos[1])
                            if new_dist > old_dist:
                                # Rewrite moves AWAY from expected — reject, keep original
                                print(f"[Filter] ({m['col_id']},{m['row_id']}) → "
                                      f"({inferred_col},{inferred_row}) REJECTED "
                                      f"(away from expected ({expected_pos[0]},{expected_pos[1]}))")
                                filtered.append(m)
                                continue
                        corrected = dict(m)
                        corrected['col_id'] = inferred_col
                        corrected['row_id'] = inferred_row
                        corrected['_inferred'] = True
                        print(f"[Filter] ({m['col_id']},{m['row_id']}) → "
                              f"({inferred_col},{inferred_row})")
                        filtered.append(corrected)
                    else:
                        print(f"[Filter] Discarded at px ({m_px[0]:.0f}, {m_px[1]:.0f})")
                else:
                    print(f"[Filter] Discarded (singular)")

        return filtered if filtered else markers

    def _measure_grid_pitch_px(self, markers):
        """
        Measure grid pitch from detected markers, robust to misidentified markers.

        Uses a consistency-based approach: for each pair of markers that share
        a row (for col pitch) or share a col (for row pitch), compute the pitch.
        Then check which measurements are mutually consistent and use only those.
        """
        if len(markers) < 2:
            print("[StageCal] Only 1 marker — pitch from probes only")
            return

        col_meas, row_meas = [], []
        for i in range(len(markers)):
            for j in range(i + 1, len(markers)):
                mi, mj = markers[i], markers[j]
                dcol = mj['col_id'] - mi['col_id']
                drow = mj['row_id'] - mi['row_id']
                dpx = np.array([mj['center'][0] - mi['center'][0],
                                mj['center'][1] - mi['center'][1]])
                if dcol != 0 and drow == 0:
                    col_meas.append(dpx / dcol)
                if drow != 0 and dcol == 0:
                    row_meas.append(dpx / drow)

        # Diagonal pairs are intentionally NOT used as fallback:
        # dpx/dcol from a diagonal conflates both axes, producing
        # systematically wrong cross-terms that poison the pitch estimate.
        # If no axis-aligned pairs exist, we rely on the probe-based pitch.

        # Robust selection: if we have multiple measurements, find the
        # largest consistent cluster and reject outliers
        if col_meas:
            self.result.grid_pitch_col_px = self._robust_pitch(col_meas, "col")
            print(f"[StageCal] Pitch/col: ({self.result.grid_pitch_col_px[0]:.1f}, "
                  f"{self.result.grid_pitch_col_px[1]:.1f}) px "
                  f"[{len(col_meas)} raw measurements]")
        if row_meas:
            self.result.grid_pitch_row_px = self._robust_pitch(row_meas, "row")
            print(f"[StageCal] Pitch/row: ({self.result.grid_pitch_row_px[0]:.1f}, "
                  f"{self.result.grid_pitch_row_px[1]:.1f}) px "
                  f"[{len(row_meas)} raw measurements]")

    def _robust_pitch(self, measurements, label):
        """
        Given a list of 2D pitch measurements, return the best one,
        rejecting outliers.

        Strategy: pick the measurement that has the most other measurements
        agreeing with it (within 15% of its magnitude). If only 1 measurement,
        use it directly.
        """
        if len(measurements) == 1:
            return np.array(measurements[0])

        arr = np.array(measurements)

        if len(measurements) == 2:
            # Only 2 measurements: check if they agree
            diff = np.linalg.norm(arr[0] - arr[1])
            avg_mag = (np.linalg.norm(arr[0]) + np.linalg.norm(arr[1])) / 2.0
            if avg_mag > 0 and diff / avg_mag < 0.15:
                # They agree, use mean
                return np.mean(arr, axis=0)
            else:
                # They disagree — we can't tell which is right with only 2
                # Use the one with larger magnitude (less likely to be a
                # misidentified marker with wrong dcol/drow)
                print(f"[StageCal] WARNING: 2 {label} pitch measurements disagree "
                      f"({arr[0]} vs {arr[1]}), using larger magnitude")
                if np.linalg.norm(arr[0]) >= np.linalg.norm(arr[1]):
                    return arr[0].copy()
                else:
                    return arr[1].copy()

        # 3+ measurements: consensus voting
        best_idx = 0
        best_inliers = 0
        for i in range(len(arr)):
            inliers = 0
            mag_i = np.linalg.norm(arr[i])
            if mag_i < 1.0:
                continue
            for j in range(len(arr)):
                diff = np.linalg.norm(arr[i] - arr[j])
                if diff / mag_i < 0.15:
                    inliers += 1
            if inliers > best_inliers:
                best_inliers = inliers
                best_idx = i

        # Average all inliers of the winning cluster
        winner = arr[best_idx]
        mag_w = np.linalg.norm(winner)
        inlier_list = []
        for m in arr:
            if np.linalg.norm(m - winner) / mag_w < 0.15:
                inlier_list.append(m)

        if len(inlier_list) < len(arr):
            rejected = len(arr) - len(inlier_list)
            print(f"[StageCal] {label} pitch: rejected {rejected}/{len(arr)} outlier(s)")

        return np.mean(inlier_list, axis=0)


def detect_and_report(frame, threshold=50):
    markers = detect_markers(frame, threshold=threshold, debug=False)
    if not markers:
        print("[detect] No markers.")
        return []
    print(f"[detect] {len(markers)} marker(s):")
    for m in markers:
        print(f"  ({m['col_id']:2d}, {m['row_id']:2d}) "
              f"px=({m['center'][0]:.0f}, {m['center'][1]:.0f}) "
              f"conf={m['confidence']:.3f}")
    return markers