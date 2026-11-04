"""
Plate Orientation Detection and Calibration.

Orientation detection: move in known motor directions, observe how raw
marker IDs trend. Resolves to one of 4 orientations: normal, h_flip,
v_flip, 180.

Calibration from scan data:
  - Motor-to-pixel: match same marker (by corrected ID) across
    consecutive scan frames. pixel_shift / motor_delta averaged over
    10+ data points.
  - Grid pitch: when corrected ID changes between consecutive frames,
    the pixel shift vs predicted gives grid_pitch_px directly.
  - px_to_motor = inverse of motor_to_px.
  - grid_pitch_fs = px_to_motor @ grid_pitch_px.

After calibration: refinement moves + Z-tilt survey.
All moves use slow speed, AF at every stop.

Initial AF uses AutoFocusController.start_calibration_af(); the best Z
is persisted to calibration.json as 'known_focus_fs' for regular AF.
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from vision.feature_detection import detect_markers, save_failed_frame
from calibration.stage_calibration import (
    CalibrationResult, measure_pixel_pitch_from_markers,
    save_calibration, load_calibration, _cal_path
)


# ═══════════════════════════════════════════════════════════════
# Bit reversal for 5-bit IDs
# ═══════════════════════════════════════════════════════════════

def bitrev5(n: int) -> int:
    result = 0
    for i in range(5):
        result |= ((n >> i) & 1) << (4 - i)
    return result

_BITREV5 = [bitrev5(i) for i in range(32)]


# ═══════════════════════════════════════════════════════════════
# Orientation correction
# ═══════════════════════════════════════════════════════════════

ORIENTATIONS = ("normal", "h_flip", "v_flip", "180")


def correct_marker_ids(markers: List[Dict], orientation: str) -> List[Dict]:
    if orientation == "normal":
        return markers
    for m in markers:
        c, r = m['col_id'], m['row_id']
        if orientation == "h_flip":
            m['col_id'] = _BITREV5[c]
            m['row_id'] = _BITREV5[r]
        elif orientation == "v_flip":
            m['col_id'] = r
            m['row_id'] = c
        elif orientation == "180":
            m['col_id'] = _BITREV5[r]
            m['row_id'] = _BITREV5[c]
    return markers


def detect_markers_corrected(rgb, orientation: str, threshold: int = 50,
                              debug: bool = False) -> List[Dict]:
    markers = detect_markers(rgb, threshold=threshold, debug=debug)
    return correct_marker_ids(markers, orientation)


# ═══════════════════════════════════════════════════════════════
# Orientation persistence
# ═══════════════════════════════════════════════════════════════

def save_orientation(orientation: str, config):
    path = _cal_path(config)
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            pass
    data['orientation'] = orientation
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[Orientation] Saved '{orientation}' to {path}")


def load_orientation(config) -> str:
    path = _cal_path(config)
    if not os.path.exists(path):
        return "normal"
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get('orientation', 'normal')
    except Exception:
        return "normal"


# ═══════════════════════════════════════════════════════════════
# Orientation analysis
# ═══════════════════════════════════════════════════════════════

def _scan_metrics(ids: List[int]) -> Tuple[float, float]:
    """Return (sign_agreement, change_rate) for a sequence of integer IDs.

      sign_agreement — among NON-ZERO consecutive deltas, the fraction that
                       share the dominant sign. 1.0 = monotonic, 0.5 = pure
                       noise. Robust against occasional jumps caused by
                       marker hops or skipped frames.
      change_rate    — fraction of consecutive deltas that are non-zero.
                       High = axis is "marching"; low = axis is "flat".

    Both metrics live in [0, 1], are bias-free across orientations (BITREV5
    doesn't inflate either of them artificially), and tolerate sparse data.
    """
    arr = [v for v in ids if v is not None]
    if len(arr) < 2:
        return 0.0, 0.0
    arr = np.asarray(arr, dtype=float)
    deltas = np.diff(arr)
    nonzero = deltas[deltas != 0]
    change_rate = float(len(nonzero)) / float(len(deltas))
    if len(nonzero) == 0:
        return 0.0, 0.0
    pos = int((nonzero > 0).sum())
    neg = int((nonzero < 0).sum())
    sign_agreement = float(max(pos, neg)) / float(len(nonzero))
    return sign_agreement, change_rate


def _apply_orientation(col_id: int, row_id: int, orientation: str) -> Tuple[int, int]:
    """Mirror of correct_marker_ids() but for a single (col,row) tuple. Kept
    local so the scoring code stays self-contained."""
    if orientation == "normal":
        return col_id, row_id
    if orientation == "h_flip":
        return _BITREV5[col_id], _BITREV5[row_id]
    if orientation == "v_flip":
        return row_id, col_id
    if orientation == "180":
        return _BITREV5[row_id], _BITREV5[col_id]
    return col_id, row_id


def _orientation_score(row_scan_data: List[Dict], col_scan_data: List[Dict],
                       orientation: str) -> Tuple[float, Dict[str, float]]:
    """Score how well a candidate orientation explains the row+col scan data.

    For the right orientation:
      - During the row scan (motor moves in Y), the CORRECTED row_id should
        march in one direction (high sign_agreement, high change_rate) and
        the col_id should stay mostly flat (low change_rate).
      - During the col scan (motor moves in X), col_id marches, row_id flat.

    Score is the sum of four [0, 1] components — march quality for the two
    expected-marching axes, plus "flatness" (1 - change_rate) for the two
    expected-flat axes. Max possible = 4.0.
    """
    row_cols, row_rows = [], []
    for d in row_scan_data:
        if d is None:
            continue
        c, r = _apply_orientation(d['col_id'], d['row_id'], orientation)
        row_cols.append(c)
        row_rows.append(r)
    col_cols, col_rows = [], []
    for d in col_scan_data:
        if d is None:
            continue
        c, r = _apply_orientation(d['col_id'], d['row_id'], orientation)
        col_cols.append(c)
        col_rows.append(r)

    rr_agree, rr_rate = _scan_metrics(row_rows)   # expect: marching
    rc_agree, rc_rate = _scan_metrics(row_cols)   # expect: flat
    cc_agree, cc_rate = _scan_metrics(col_cols)   # expect: marching
    cr_agree, cr_rate = _scan_metrics(col_rows)   # expect: flat

    # March quality = agreement × rate. Sign-agreement keeps BITREV5 from
    # winning on raw magnitude (its deltas alternate sign, dragging agreement
    # below the monotonic case).
    row_march = rr_agree * rr_rate
    col_march = cc_agree * cc_rate
    row_flat  = 1.0 - rc_rate
    col_flat  = 1.0 - cr_rate

    parts = {
        'row_march': row_march, 'col_march': col_march,
        'row_flat':  row_flat,  'col_flat':  col_flat,
        'rr_agree':  rr_agree,  'rr_rate':   rr_rate,
        'cc_agree':  cc_agree,  'cc_rate':   cc_rate,
        'rc_rate':   rc_rate,   'cr_rate':   cr_rate,
    }
    return row_march + col_march + row_flat + col_flat, parts


def analyze_orientation(row_scan_data: List[Dict], col_scan_data: List[Dict]) -> str:
    """Choose the orientation (normal / h_flip / v_flip / 180) that best
    explains the row+col scan data. Replaces the old R²-gated decision tree,
    which silently fell into the '180' bucket whenever R² < 0.7 — and R²
    drops fast under sparse markers / marker hops / detection noise.

    The new scoring is bias-free across orientations (no BITREV5 magnitude
    inflation), tolerates skipped frames, and degrades gracefully when only
    a few clean frames are available."""
    scores = {}
    for ori in ORIENTATIONS:
        s, parts = _orientation_score(row_scan_data, col_scan_data, ori)
        scores[ori] = (s, parts)
        print(f"[Orient] {ori:>7s}: score={s:.3f} "
              f"(row_march={parts['row_march']:.2f} col_march={parts['col_march']:.2f} "
              f"row_flat={parts['row_flat']:.2f} col_flat={parts['col_flat']:.2f})")

    best = max(scores, key=lambda o: scores[o][0])
    best_score = scores[best][0]
    # Margin = how decisively the winner beat the runner-up. Helps the user
    # diagnose ambiguous calibrations from the log without aborting.
    runner_up = max((s for o, (s, _) in scores.items() if o != best), default=0.0)
    margin = best_score - runner_up
    print(f"[Orient] → {best} (margin over runner-up: {margin:+.3f})")
    return best


# ═══════════════════════════════════════════════════════════════
# Calibration from scan data
# ═══════════════════════════════════════════════════════════════

def _compute_grid_pitch(scan_frames: List[Dict], direction: str) -> Optional[np.ndarray]:
    """Compute grid pitch in pixels from tile boundary crossings.

    When the corrected marker ID changes between consecutive frames,
    we've crossed a tile boundary. The pixel position difference of
    markers at the new tile position gives grid_pitch_px directly.

    direction: 'row' or 'col' — which pitch to compute.
    """
    pitch_samples = []

    for i in range(len(scan_frames) - 1):
        markers_a = scan_frames[i]['markers']
        markers_b = scan_frames[i + 1]['markers']

        if not markers_a or not markers_b:
            continue

        # Get best marker from each frame
        best_a = markers_a[0]  # already sorted by centrality
        best_b = markers_b[0]

        dcol = best_b['col_id'] - best_a['col_id']
        drow = best_b['row_id'] - best_a['row_id']

        if direction == 'col' and dcol != 0 and drow == 0:
            # Pure col boundary crossing
            dpx = np.array([best_b['center'][0] - best_a['center'][0],
                            best_b['center'][1] - best_a['center'][1]])
            pitch_samples.append(dpx / dcol)
        elif direction == 'row' and drow != 0 and dcol == 0:
            # Pure row boundary crossing
            dpx = np.array([best_b['center'][0] - best_a['center'][0],
                            best_b['center'][1] - best_a['center'][1]])
            pitch_samples.append(dpx / drow)

    if not pitch_samples:
        return None

    return np.mean(pitch_samples, axis=0)


# ═══════════════════════════════════════════════════════════════
# Controller
# ═══════════════════════════════════════════════════════════════

SAFE_CENTER_X_FS = 4265.0
SAFE_CENTER_Y_FS = 470.0
SCAN_STEPS = 10
DISCARD_FRAMES = 3
DETECTION_THRESHOLD = 20
SETTLE_MS = 400
CAL_SPEED_MULTIPLIER = 0.5


class OrientCalibrationController(QObject):
    """
    Combined orientation detection + stage calibration.

    All tile moves use slow speed, AF at every stop.
    Calibration is computed from scan data:
      - motor_to_px from matched markers across consecutive frames
      - grid_pitch from tile boundary crossings
    No requirement for 2+ markers in any single frame.

    The initial AF after move-to-center uses the calibration-AF entry point
    on AutoFocusController, which sweeps Z=100..200 at 1.5fs and persists
    the result as the new known_focus_fs in calibration.json.
    """

    progress = pyqtSignal(str, str)
    finished = pyqtSignal(bool, str, object)

    DETECTION_THRESHOLD = DETECTION_THRESHOLD

    # Cap-bounded reference detection
    MAX_REFERENCE_RETRIES = 5
    REFERENCE_RETRY_DELAY_MS = 300

    def __init__(self, move_to_ctrl, camera_preview, mc, motors_model,
                 config=None, af_controller=None, homing=None, parent=None):
        super().__init__(parent)
        self.move_to_ctrl = move_to_ctrl
        self.camera_preview = camera_preview
        self.mc = mc
        self.MOTORS = motors_model
        self.config = config
        self.af_controller = af_controller
        self.homing = homing

        self._active = False
        self._stop_requested = False
        self._phase = 'idle'
        self._camera_settings = {}
        self._saved_speed_mult = 1.0
        self._af_kickoff_pending = False

        # Orientation
        self._orientation = load_orientation(config) if config else "normal"

        # Scan data for orientation trend analysis
        self._row_scan_data: List[Optional[Dict]] = []
        self._col_scan_data: List[Optional[Dict]] = []
        self._scan_step = 0
        self._scan_center_x = 0.0
        self._scan_center_y = 0.0

        # Scan frames: each entry has {'markers': [...], 'motor_pos': np.array}
        # Stored per direction for calibration computation
        self._row_scan_frames: List[Dict] = []
        self._col_scan_frames: List[Dict] = []
        self._row_motor_deltas: List[np.ndarray] = []
        self._col_motor_deltas: List[np.ndarray] = []
        self._last_scan_motor_pos = np.zeros(2)

        # Reverse walk
        self._reverse_steps_remaining = 0
        self._scan_reverse_direction = 'row'

        # Calibration result — load existing
        self.result = CalibrationResult()
        if config is not None:
            saved = load_calibration(config)
            if saved and saved.valid:
                self.result = saved

        # Refinement
        self._refine_targets = []
        self._refine_idx = 0
        self._refine_start_x = 0.0
        self._refine_start_y = 0.0
        self._refine_current_target = (0, 0)
        self._anchor_col = 0
        self._anchor_row = 0

        # Z-tilt calibration scan
        self._z_scan = None  # SpiralScanController for Z calibration

        # Phase 6: refinement cycles
        self._refine_cycles_target = 0
        self._refine_cycles_done = 0

        # Connect signals
        self.move_to_ctrl.finished.connect(self._on_move_finished)
        if self.af_controller:
            self.af_controller.finished.connect(self._on_af_finished)
        print(f"[OrientCal id={id(self)}] __init__ done, connected to move_to_ctrl.finished (move_to_ctrl id={id(self.move_to_ctrl)})")

    # ── Public API ──

    def get_orientation(self) -> str:
        return self._orientation

    def get_result(self) -> CalibrationResult:
        return self.result

    def is_active(self) -> bool:
        return self._active

    def start(self, camera_settings, skip_move_to_center=False):
        """Start orientation calibration.

        If skip_move_to_center=True, the caller guarantees the stage is
        already at (SAFE_CENTER_X_FS, SAFE_CENTER_Y_FS, 0). We skip the
        internal move-to-center entirely and go straight to AF — no
        move_to_ctrl.finished coupling for the setup phase, so no race
        can prematurely trigger AF.
        """
        if self._active:
            return
        if not self.camera_preview.is_running():
            self.finished.emit(False, "Start live preview first.", None)
            return

        self._active = True
        self._stop_requested = False
        self._af_kickoff_pending = False
        self._center_arrival_polls = 0
        self._camera_settings = camera_settings
        self._row_scan_data = []
        self._col_scan_data = []
        self._row_scan_frames = []
        self._col_scan_frames = []
        self._row_motor_deltas = []
        self._col_motor_deltas = []
        self._scan_step = 0
        self._reverse_steps_remaining = 0

        try:
            self._saved_speed_mult = self.mc.get_speed_multiplier()
            self.mc.set_speed_multiplier(CAL_SPEED_MULTIPLIER)
        except Exception:
            self._saved_speed_mult = 1.0

        # phase='move_to_center' is required either way — _start_autofocus
        # checks it as an idempotency guard.
        self._phase = 'move_to_center'
        if skip_move_to_center:
            self.progress.emit('setup', 'Starting calibration AF (already at center)...')
            QTimer.singleShot(SETTLE_MS, self._start_autofocus)
        else:
            self.progress.emit('setup', 'Moving to safe region center...')
            self.move_to_ctrl.start(SAFE_CENTER_X_FS, SAFE_CENTER_Y_FS, 0.0)

    def stop(self, reason="Cancelled"):
        if not self._active:
            return
        self._stop_requested = True
        self._active = False
        self._phase = 'idle'
        # Stop Z calibration spiral scan if running
        if self._z_scan is not None and self._z_scan.is_active():
            self._z_scan.stop("Calibration cancelled")
        self._restore_speed()
        print(f"[OrientCal] Stopped: {reason}")
        self.finished.emit(False, reason, None)

    # ── Helpers ──

    def _restore_speed(self):
        try:
            self.mc.set_speed_multiplier(self._saved_speed_mult)
        except Exception:
            pass

    def _cur_x(self) -> float:
        return float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))

    def _cur_y(self) -> float:
        return float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))

    def _cur_z(self) -> float:
        return float(getattr(self.MOTORS, "z_current_position_full_step", 0.0))

    def _cur_motor_pos(self) -> np.ndarray:
        return np.array([self._cur_x(), self._cur_y()])

    def _flush_and_capture(self):
        return self.camera_preview.flush_and_capture(
            self._camera_settings, discard=DISCARD_FRAMES)

    def _detect_raw(self, frame) -> List[Dict]:
        return detect_markers(frame, threshold=DETECTION_THRESHOLD)

    def _detect_corrected(self, frame) -> List[Dict]:
        return detect_markers_corrected(frame, self._orientation,
                                         threshold=DETECTION_THRESHOLD)

    def _get_best_marker(self, markers) -> Optional[Dict]:
        if not markers:
            return None
        cx = self.result.image_center_x or 2028.0
        cy = self.result.image_center_y or 1520.0
        best = min(markers, key=lambda m:
                   (m['center'][0] - cx)**2 + (m['center'][1] - cy)**2)
        return {'col_id': best['col_id'], 'row_id': best['row_id'],
                'center': best['center']}

    def _half_tile_row_fs(self) -> np.ndarray:
        """Half-tile step for row scan: pure motor X direction."""
        return np.array([-5.0, 0.0])

    def _half_tile_col_fs(self) -> np.ndarray:
        """Half-tile step for col scan: pure motor Y direction."""
        return np.array([0.0, 7.9])

    def _do_quick_af(self, next_phase):
        self._phase = next_phase
        if self.af_controller:
            QTimer.singleShot(SETTLE_MS,
                lambda: self.af_controller.start_quick(
                    self._camera_settings, mode='minimal'))
        else:
            QTimer.singleShot(SETTLE_MS, lambda: self._on_af_finished(True, "no AF"))

    def _compute_tile_move(self, markers, target_col, target_row):
        cx, cy = self.result.image_center_x, self.result.image_center_y
        ref_m = min(markers, key=lambda m:
                    (m['center'][0] - cx)**2 + (m['center'][1] - cy)**2)
        dcol = target_col - ref_m['col_id']
        drow = target_row - ref_m['row_id']

        obs_col_px, obs_row_px = measure_pixel_pitch_from_markers(markers)
        col_px = obs_col_px if obs_col_px is not None else self.result.grid_pitch_col_px
        row_px = obs_row_px if obs_row_px is not None else self.result.grid_pitch_row_px

        grid_dpx = dcol * col_px + drow * row_px
        ideal_px = np.array([cx, cy]) - 0.5 * col_px - 0.5 * row_px
        offset_px = ideal_px - np.array([ref_m['center'][0], ref_m['center'][1]])
        total_dpx = -grid_dpx + offset_px
        motor_delta = self.result.px_to_motor @ total_dpx

        return (self._cur_x() + motor_delta[0],
                self._cur_y() + motor_delta[1],
                ref_m)

    # ═══════════════════════════════════════════════════════════════
    # Phase 0: Move to center + AF
    # ═══════════════════════════════════════════════════════════════

    def _start_autofocus(self):
        if self._stop_requested:
            return
        # Idempotency guard: a stale move_to_ctrl.finished must not start
        # AF on top of an in-progress AF.
        if self._phase != 'move_to_center':
            print(f"[OrientCal id={id(self)}] _start_autofocus ignored "
                  f"(phase={self._phase}, not move_to_center)")
            return

        # Only start AF if the stage has actually arrived at SAFE_CENTER;
        # otherwise poll every 200 ms (bounded) to avoid racing z_homing.
        POS_TOL_FS = 2.0
        cur_x, cur_y, cur_z = self._cur_x(), self._cur_y(), self._cur_z()
        at_target = (abs(cur_x - SAFE_CENTER_X_FS) <= POS_TOL_FS
                     and abs(cur_y - SAFE_CENTER_Y_FS) <= POS_TOL_FS
                     and abs(cur_z - 0.0) <= POS_TOL_FS)
        if not at_target:
            polls = getattr(self, '_center_arrival_polls', 0) + 1
            self._center_arrival_polls = polls
            if polls <= 300:  # 300 * 200ms = 60s
                print(f"[OrientCal id={id(self)}] _start_autofocus deferred: "
                      f"not at SAFE_CENTER yet "
                      f"(cur=({cur_x:.1f}, {cur_y:.1f}, {cur_z:.1f}) "
                      f"target=({SAFE_CENTER_X_FS}, {SAFE_CENTER_Y_FS}, 0.0)) "
                      f"poll {polls}/300")
                QTimer.singleShot(200, self._start_autofocus)
                return
            print(f"[OrientCal id={id(self)}] _start_autofocus: arrival "
                  f"timeout after {polls} polls — proceeding anyway")
        self._center_arrival_polls = 0

        self._af_kickoff_pending = False
        print(f"[OrientCal id={id(self)}] _start_autofocus fired, phase was {self._phase}")
        self._phase = 'initial_af'
        self.progress.emit('af', 'Calibration AF (Z=130..200)...')
        if self.af_controller:
            # Wide-sweep calibration AF: stage 1 sweeps Z=100..200 at 1.5fs,
            # subsequent stages refine, and the final best Z is persisted to
            # calibration.json under 'known_focus_fs' for future regular AF
            # runs.
            self.af_controller.start_calibration_af(self._camera_settings)
        else:
            QTimer.singleShot(SETTLE_MS, self._capture_reference)

    def _capture_reference(self):
        if self._stop_requested:
            return
        frame = self._flush_and_capture()
        if frame is None:
            self.stop("Reference capture failed")
            return

        h, w = frame.shape[:2]
        self.result.image_center_x = w / 2.0
        self.result.image_center_y = h / 2.0

        markers = self._detect_raw(frame)
        if not markers:
            p = save_failed_frame(frame, "orient_reference")
            self._ref_retries = getattr(self, '_ref_retries', 0) + 1
            if self._ref_retries < self.MAX_REFERENCE_RETRIES:
                print(f"[OrientCal] No markers at safe center — retry "
                      f"{self._ref_retries}/{self.MAX_REFERENCE_RETRIES} "
                      f"(saved {p})")
                QTimer.singleShot(self.REFERENCE_RETRY_DELAY_MS, self._capture_reference)
                return
            print(f"[OrientCal] No markers after {self._ref_retries} retries — saved {p}")
            self._ref_retries = 0
            self.stop("No markers visible at safe center.")
            return
        self._ref_retries = 0

        print(f"[OrientCal] Reference: {len(markers)} markers at center")
        for m in markers:
            print(f"  ({m['col_id']}, {m['row_id']}) "
                  f"px=({m['center'][0]:.0f}, {m['center'][1]:.0f})")

        self._scan_center_x = self._cur_x()
        self._scan_center_y = self._cur_y()
        self._last_scan_motor_pos = self._cur_motor_pos()

        # Start row-direction scan
        self._scan_step = 0
        self._row_scan_data = []
        self._row_scan_frames = []
        self._row_motor_deltas = []
        self.progress.emit('orient', 'Orientation scan: row direction...')
        self._scan_capture_and_advance('row')

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Orientation Scan
    # ═══════════════════════════════════════════════════════════════

    def _scan_capture_and_advance(self, direction: str):
        if self._stop_requested:
            return

        cur_motor = self._cur_motor_pos()
        frame = self._flush_and_capture()
        markers = self._detect_raw(frame) if frame is not None else []
        best = self._get_best_marker(markers) if markers else None

        # Store for orientation trend analysis
        if direction == 'row':
            self._row_scan_data.append(best)
            # Store full frame data for calibration
            if self._scan_step > 0:
                motor_delta = cur_motor - self._last_scan_motor_pos
                self._row_motor_deltas.append(motor_delta)
            self._row_scan_frames.append({
                'markers': list(markers),
                'motor_pos': cur_motor.copy()
            })
        else:
            self._col_scan_data.append(best)
            if self._scan_step > 0:
                motor_delta = cur_motor - self._last_scan_motor_pos
                self._col_motor_deltas.append(motor_delta)
            self._col_scan_frames.append({
                'markers': list(markers),
                'motor_pos': cur_motor.copy()
            })

        self._last_scan_motor_pos = cur_motor.copy()
        self._scan_step += 1

        if best:
            print(f"[OrientCal] {direction}-scan step {self._scan_step}: "
                  f"col_id={best['col_id']}, row_id={best['row_id']}")
        else:
            print(f"[OrientCal] {direction}-scan step {self._scan_step}: no detection")

        if self._scan_step > SCAN_STEPS:
            self._reverse_steps_remaining = SCAN_STEPS
            self._scan_reverse_direction = direction
            self.progress.emit('orient', f'Reversing {direction} scan...')
            self._advance_reverse_step()
            return

        delta_fs = (self._half_tile_row_fs() if direction == 'row'
                    else self._half_tile_col_fs())
        self._phase = f'{direction}_scan_move'
        self.move_to_ctrl.start(self._cur_x() + delta_fs[0],
                                 self._cur_y() + delta_fs[1],
                                 self._cur_z())

    def _advance_reverse_step(self):
        if self._stop_requested:
            return
        if self._reverse_steps_remaining <= 0:
            direction = self._scan_reverse_direction
            self._phase = f'{direction}_scan_snap'
            self.progress.emit('orient', 'Snapping to center...')
            self.move_to_ctrl.start(self._scan_center_x, self._scan_center_y,
                                     self._cur_z())
            return

        direction = self._scan_reverse_direction
        delta_fs = (self._half_tile_row_fs() if direction == 'row'
                    else self._half_tile_col_fs())
        self._reverse_steps_remaining -= 1
        self._phase = f'{direction}_reverse_move'
        self.progress.emit('orient',
            f'Reversing {direction}: {SCAN_STEPS - self._reverse_steps_remaining}/{SCAN_STEPS}...')
        self.move_to_ctrl.start(self._cur_x() - delta_fs[0],
                                 self._cur_y() - delta_fs[1],
                                 self._cur_z())

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Analyze orientation
    # ═══════════════════════════════════════════════════════════════

    def _analyze_and_store_orientation(self):
        if self._stop_requested:
            return

        row_valid = [d for d in self._row_scan_data if d is not None]
        col_valid = [d for d in self._col_scan_data if d is not None]

        if len(row_valid) < 3 or len(col_valid) < 3:
            print(f"[OrientCal] Not enough data (row={len(row_valid)}, "
                  f"col={len(col_valid)}). Assuming normal.")
            self._orientation = "normal"
        else:
            self._orientation = analyze_orientation(row_valid, col_valid)

        print(f"[OrientCal] Detected orientation: {self._orientation}")
        self.progress.emit('orient', f'Orientation: {self._orientation}')

        if self.config:
            save_orientation(self._orientation, self.config)

        # Now compute calibration from scan data
        QTimer.singleShot(200, self._compute_calibration_from_scans)

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Compute calibration from scan data
    # ═══════════════════════════════════════════════════════════════

    def _compute_calibration_from_scans(self):
        """Use scan data to compute motor_to_px and grid pitch.

        1. Reset calibration (no stale fallbacks)
        2. Correct all stored marker IDs for orientation
        3. Match same markers across consecutive frames → motor_to_px
        4. Track boundary crossings across frames → grid_pitch_fs
        5. Derive px_to_motor and grid_pitch_px
        """
        if self._stop_requested:
            return

        self.progress.emit('calibrating', 'Computing calibration from scan data...')

        # Step 1: Reset result — keep only image center, no stale fallbacks
        saved_cx = self.result.image_center_x
        saved_cy = self.result.image_center_y
        self.result = CalibrationResult()
        self.result.image_center_x = saved_cx
        self.result.image_center_y = saved_cy

        # Step 2: Correct all stored markers for determined orientation
        for frame_data in self._row_scan_frames:
            correct_marker_ids(frame_data['markers'], self._orientation)
        for frame_data in self._col_scan_frames:
            correct_marker_ids(frame_data['markers'], self._orientation)

        # Sort markers in each frame by centrality (closest to image center first)
        cx = self.result.image_center_x or 2028.0
        cy = self.result.image_center_y or 1520.0
        for frame_data in self._row_scan_frames + self._col_scan_frames:
            frame_data['markers'].sort(
                key=lambda m: (m['center'][0] - cx)**2 + (m['center'][1] - cy)**2)

        # Step 3: Compute motor_to_px from matched markers across consecutive frames
        px_shifts = []
        mt_shifts = []

        for frames, deltas in [(self._row_scan_frames, self._row_motor_deltas),
                                (self._col_scan_frames, self._col_motor_deltas)]:
            for i in range(len(deltas)):
                markers_a = frames[i]['markers']
                markers_b = frames[i + 1]['markers']
                motor_delta = deltas[i]

                if not markers_a or not markers_b:
                    continue
                if np.linalg.norm(motor_delta) < 0.01:
                    continue

                # Match markers by corrected ID
                b_by_id = {}
                for m in markers_b:
                    b_by_id[(m['col_id'], m['row_id'])] = m

                frame_shifts = []
                for m_a in markers_a:
                    key = (m_a['col_id'], m_a['row_id'])
                    if key in b_by_id:
                        m_b = b_by_id[key]
                        dpx = np.array([m_b['center'][0] - m_a['center'][0],
                                        m_b['center'][1] - m_a['center'][1]])
                        frame_shifts.append(dpx)

                if frame_shifts:
                    px_shifts.append(np.mean(frame_shifts, axis=0))
                    mt_shifts.append(motor_delta)

        motor_to_px_ok = False
        if len(px_shifts) >= 2:
            P = np.array(px_shifts)
            M = np.array(mt_shifts)
            try:
                MtM = M.T @ M
                if abs(np.linalg.det(MtM)) > 1e-10:
                    M2P_T = np.linalg.solve(MtM, M.T @ P)
                    motor_to_px = M2P_T.T
                    if abs(np.linalg.det(motor_to_px)) > 1e-10:
                        self.result.motor_to_px = motor_to_px
                        self.result.px_to_motor = np.linalg.inv(motor_to_px)
                        motor_to_px_ok = True
                        m = motor_to_px
                        print(f"[OrientCal] Motor-to-pixel from {len(px_shifts)} pairs:")
                        print(f"  +motor_X → px ({m[0,0]:+.3f}, {m[1,0]:+.3f})")
                        print(f"  +motor_Y → px ({m[0,1]:+.3f}, {m[1,1]:+.3f})")
                    else:
                        print("[OrientCal] motor_to_px singular!")
                else:
                    print("[OrientCal] Motor matrix singular!")
            except Exception as e:
                print(f"[OrientCal] motor_to_px failed: {e}")
        else:
            print(f"[OrientCal] Only {len(px_shifts)} matched pairs — not enough for motor_to_px.")

        if not motor_to_px_ok:
            self.stop("Calibration failed: could not compute motor-to-pixel matrix")
            return

        # ── Step 4: Grid pitch ────────────────────────────────────────
        # Within-frame is the primary source: axis-aligned marker pairs in
        # one frame give direct geometric observations. Crossings is only
        # used as a fallback for scan regions where every frame has <2
        # markers.
        col_px_samples, row_px_samples = [], []
        for frame_data in self._row_scan_frames + self._col_scan_frames:
            if len(frame_data['markers']) >= 2:
                cp, rp = measure_pixel_pitch_from_markers(frame_data['markers'])
                if cp is not None:
                    col_px_samples.append(cp)
                if rp is not None:
                    row_px_samples.append(rp)

        col_pitch_fs = row_pitch_fs = None
        if col_px_samples:
            col_pitch_px_mean = np.mean(np.stack(col_px_samples), axis=0)
            col_pitch_fs = self.result.px_to_motor @ col_pitch_px_mean
            print(f"[OrientCal] Col pitch from {len(col_px_samples)} within-frame "
                  f"samples: ({col_pitch_fs[0]:.2f}, {col_pitch_fs[1]:.2f})fs "
                  f"px=({col_pitch_px_mean[0]:.1f}, {col_pitch_px_mean[1]:.1f})")
        if row_px_samples:
            row_pitch_px_mean = np.mean(np.stack(row_px_samples), axis=0)
            row_pitch_fs = self.result.px_to_motor @ row_pitch_px_mean
            print(f"[OrientCal] Row pitch from {len(row_px_samples)} within-frame "
                  f"samples: ({row_pitch_fs[0]:.2f}, {row_pitch_fs[1]:.2f})fs "
                  f"px=({row_pitch_px_mean[0]:.1f}, {row_pitch_px_mean[1]:.1f})")

        # Crossings fallback for axes that within-frame couldn't measure
        if col_pitch_fs is None:
            col_pitch_fs = self._measure_pitch_from_crossings(
                self._col_scan_frames, self._col_motor_deltas, 'col')
        if row_pitch_fs is None:
            row_pitch_fs = self._measure_pitch_from_crossings(
                self._row_scan_frames, self._row_motor_deltas, 'row')

        # Sanity check whichever source produced each pitch. A pitch whose
        # pixel magnitude exceeds the image diagonal can't correspond to
        # adjacent tiles in this FOV — that's a math failure, not a real
        # measurement. Drop it back to None so the next-tier source can try.
        h_px = (self.result.image_center_y or 1520.0) * 2.0
        w_px = (self.result.image_center_x or 2028.0) * 2.0
        img_diag_px = float(np.hypot(w_px, h_px))

        def _is_pitch_sane(pitch_fs: Optional[np.ndarray], label: str) -> bool:
            if pitch_fs is None:
                return False
            pitch_px = self.result.motor_to_px @ pitch_fs
            mag = float(np.linalg.norm(pitch_px))
            if mag > img_diag_px:
                print(f"[OrientCal] REJECT {label} pitch — px-magnitude "
                      f"{mag:.0f} > image diagonal {img_diag_px:.0f} "
                      f"(physically impossible for adjacent tiles)")
                return False
            return True

        if not _is_pitch_sane(col_pitch_fs, 'col'):
            col_pitch_fs = None
        if not _is_pitch_sane(row_pitch_fs, 'row'):
            row_pitch_fs = None

        if col_pitch_fs is None or row_pitch_fs is None:
            self.stop("Calibration failed: could not measure grid pitch")
            return

        # grid_pitch_*_fs is the raw within-frame pitch in motor units
        # ("px shift per +1 col_id step", run through px_to_motor).
        # spiral_scan applies its own −grid_dpx in the final motor_delta
        # so the sign stored here is positive.
        self.result.grid_pitch_col_fs = col_pitch_fs
        self.result.grid_pitch_row_fs = row_pitch_fs
        self.result.grid_pitch_col_px = self.result.motor_to_px @ self.result.grid_pitch_col_fs
        self.result.grid_pitch_row_px = self.result.motor_to_px @ self.result.grid_pitch_row_fs

        print(f"[OrientCal] Grid pitch col: fs=({self.result.grid_pitch_col_fs[0]:.2f}, "
              f"{self.result.grid_pitch_col_fs[1]:.2f}) "
              f"px=({self.result.grid_pitch_col_px[0]:.1f}, {self.result.grid_pitch_col_px[1]:.1f})")
        print(f"[OrientCal] Grid pitch row: fs=({self.result.grid_pitch_row_fs[0]:.2f}, "
              f"{self.result.grid_pitch_row_fs[1]:.2f}) "
              f"px=({self.result.grid_pitch_row_px[0]:.1f}, {self.result.grid_pitch_row_px[1]:.1f})")

        self.result.valid = True
        self._print_result("Calibration (from scan data)")
        self._start_refinement()

    def _measure_pitch_from_crossings(self, frames, motor_deltas, scan_name=''):
        if len(frames) < 2:
            return None

        sequence = []
        cum_motor = np.zeros(2)
        for i, frame_data in enumerate(frames):
            if i > 0 and i - 1 < len(motor_deltas):
                cum_motor = cum_motor + motor_deltas[i - 1]
            if not frame_data['markers']:
                continue
            best = frame_data['markers'][0]  # already sorted by centrality
            sequence.append({
                'col_id': best['col_id'],
                'row_id': best['row_id'],
                'motor_pos': cum_motor.copy(),
            })

        if len(sequence) < 3:
            return None

        first = sequence[0]
        last = sequence[-1]
        total_dcol = last['col_id'] - first['col_id']
        total_drow = last['row_id'] - first['row_id']
        total_motor = last['motor_pos'] - first['motor_pos']

        if abs(total_dcol) >= abs(total_drow) and abs(total_dcol) >= 2:
            pitch = total_motor / total_dcol
            print(f"[OrientCal] {scan_name} pitch: total dcol={total_dcol} over "
                  f"{len(sequence)} frames → ({pitch[0]:.2f}, {pitch[1]:.2f})fs")
            return pitch
        elif abs(total_drow) >= 2:
            pitch = total_motor / total_drow
            print(f"[OrientCal] {scan_name} pitch: total drow={total_drow} over "
                  f"{len(sequence)} frames → ({pitch[0]:.2f}, {pitch[1]:.2f})fs")
            return pitch

        print(f"[OrientCal] {scan_name} scan: insufficient ID change "
              f"(dcol={total_dcol}, drow={total_drow})")
        return None

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Refinement — one tile at a time, AF at each stop
    # ═══════════════════════════════════════════════════════════════

    def _start_refinement(self):
        if self._stop_requested:
            return

        # AF at current position before starting refinement
        self._do_quick_af('pre_refine_af')

    def _begin_refinement_after_af(self):
        if self._stop_requested:
            return

        frame = self._flush_and_capture()
        if frame is None:
            self._start_z_tilt_survey()
            return
        markers = self._detect_corrected(frame)
        if not markers:
            p = save_failed_frame(frame, "orient_refine_begin")
            print(f"[OrientCal] Refine begin: no markers — saved {p}")
            self._start_z_tilt_survey()
            return

        cx, cy = self.result.image_center_x, self.result.image_center_y
        ref_m = min(markers, key=lambda m:
                    (m['center'][0] - cx)**2 + (m['center'][1] - cy)**2)

        self._anchor_col = ref_m['col_id']
        self._anchor_row = ref_m['row_id']

        d1_c = 1 if self._anchor_col + 1 <= 31 else -1
        d1_r = 1 if self._anchor_row + 1 <= 31 else -1

        self._refine_targets = [
            (self._anchor_col + d1_c, self._anchor_row),
            (self._anchor_col, self._anchor_row + d1_r),
            (self._anchor_col + d1_c, self._anchor_row + d1_r)
        ]
        self._refine_idx = 0
        self._refine_start_x = self._cur_x()
        self._refine_start_y = self._cur_y()

        print(f"[OrientCal] Refinement: anchor ({self._anchor_col},{self._anchor_row}), "
              f"targets: {self._refine_targets}")
        self._refine_detect_and_move()

    def _refine_detect_and_move(self):
        if self._stop_requested:
            return
        if self._refine_idx >= len(self._refine_targets):
            self._phase = 'refine_return'
            self.progress.emit('refining', 'Returning to start...')
            self.move_to_ctrl.start(self._refine_start_x, self._refine_start_y,
                                     self._cur_z())
            return

        target_col, target_row = self._refine_targets[self._refine_idx]
        self.progress.emit('refining',
            f'Refine {self._refine_idx+1}/{len(self._refine_targets)}: '
            f'({target_col},{target_row})...')

        frame = self._flush_and_capture()
        if frame is None:
            self._refine_idx += 1
            self._refine_detect_and_move()
            return
        markers = self._detect_corrected(frame)
        if not markers:
            p = save_failed_frame(frame, "orient_refine_predict")
            print(f"[OrientCal] Refine predict: no markers — saved {p}")
            self._refine_idx += 1
            self._refine_detect_and_move()
            return

        result = self._compute_tile_move(markers, target_col, target_row)
        if result is None:
            self._refine_idx += 1
            self._refine_detect_and_move()
            return

        target_x, target_y, _ = result
        self._refine_current_target = (target_col, target_row)
        self._phase = 'refine_move'
        self.move_to_ctrl.start(target_x, target_y, self._cur_z())

    def _refine_capture(self):
        if self._stop_requested:
            return
        frame = self._flush_and_capture()
        if frame is None:
            self._refine_idx += 1
            self._refine_return_to_start()
            return
        markers = self._detect_corrected(frame)
        if not markers:
            p = save_failed_frame(frame, "orient_refine_capture")
            print(f"[OrientCal] Refine capture: no markers — saved {p}")
            self._refine_idx += 1
            self._refine_return_to_start()
            return

        target_col, target_row = self._refine_current_target
        target_m = None
        for m in markers:
            if m['col_id'] == target_col and m['row_id'] == target_row:
                target_m = m
                break
        if target_m is None:
            target_m = min(markers, key=lambda m:
                           (m['col_id'] - target_col)**2 +
                           (m['row_id'] - target_row)**2)

        ideal_px = np.array([self.result.image_center_x, self.result.image_center_y]) \
                   - 0.5 * self.result.grid_pitch_col_px - 0.5 * self.result.grid_pitch_row_px

        dcol_off = target_m['col_id'] - target_col
        drow_off = target_m['row_id'] - target_row

        obs_col_px, obs_row_px = measure_pixel_pitch_from_markers(markers)
        col_px = obs_col_px if obs_col_px is not None else self.result.grid_pitch_col_px
        row_px = obs_row_px if obs_row_px is not None else self.result.grid_pitch_row_px

        expected_px = ideal_px + dcol_off * col_px + drow_off * row_px
        actual_px = np.array([target_m['center'][0], target_m['center'][1]])
        miss_px = actual_px - expected_px

        print(f"[OrientCal] Refine {self._refine_idx+1}: "
              f"target ({target_col},{target_row}), "
              f"miss px=({miss_px[0]:.1f},{miss_px[1]:.1f})")

        dcol_total = target_col - self._anchor_col
        drow_total = target_row - self._anchor_row
        learn_rate = 0.5

        if dcol_total != 0 and drow_total == 0:
            correction = miss_px / dcol_total * learn_rate
            cap = np.linalg.norm(self.result.grid_pitch_col_px) * 0.25
            if np.linalg.norm(correction) > cap:
                correction = correction / np.linalg.norm(correction) * cap
            self.result.grid_pitch_col_px += correction
        elif drow_total != 0 and dcol_total == 0:
            correction = miss_px / drow_total * learn_rate
            cap = np.linalg.norm(self.result.grid_pitch_row_px) * 0.25
            if np.linalg.norm(correction) > cap:
                correction = correction / np.linalg.norm(correction) * cap
            self.result.grid_pitch_row_px += correction
        elif dcol_total != 0 and drow_total != 0:
            frac = abs(dcol_total) / (abs(dcol_total) + abs(drow_total))
            cc = miss_px * frac / dcol_total * learn_rate
            rc = miss_px * (1 - frac) / drow_total * learn_rate
            cap_c = np.linalg.norm(self.result.grid_pitch_col_px) * 0.15
            cap_r = np.linalg.norm(self.result.grid_pitch_row_px) * 0.15
            if np.linalg.norm(cc) > cap_c:
                cc = cc / np.linalg.norm(cc) * cap_c
            if np.linalg.norm(rc) > cap_r:
                rc = rc / np.linalg.norm(rc) * cap_r
            self.result.grid_pitch_col_px += cc
            self.result.grid_pitch_row_px += rc

        self.result.grid_pitch_col_fs = self.result.px_to_motor @ self.result.grid_pitch_col_px
        self.result.grid_pitch_row_fs = self.result.px_to_motor @ self.result.grid_pitch_row_px

        self._refine_idx += 1
        self._refine_return_to_start()

    def _refine_return_to_start(self):
        self._phase = 'refine_return_for_next'
        self.move_to_ctrl.start(self._refine_start_x, self._refine_start_y,
                                 self._cur_z())

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Z-tilt calibration via full spiral scan
    # ═══════════════════════════════════════════════════════════════

    def _start_z_tilt_survey(self):
        if self._stop_requested:
            return

        from calibration.spiral_scan import SpiralScanController

        # Save calibration before Z-scan
        self.result.valid = True
        self._print_result("Pre-Z-scan Calibration")
        if self.config:
            save_calibration(self.result, self.config)
            print(f"[OrientCal] Calibration saved to disk before Z-scan")

        self._z_scan = SpiralScanController(
            calibration=self,
            af_controller=self.af_controller,
            move_to_ctrl=self.move_to_ctrl,
            camera_preview=self.camera_preview,
            mc=self.mc,
            motors_config=self.MOTORS,
            config=self.config,
            parent=self,
        )

        # Override settings for calibration: AF every tile, wider search
        self._z_scan.AF_EVERY_N_TILES = 1
        self._z_scan.SCAN_SIZE = 10
        self._z_scan._scan_size = 10
        self._z_scan._center_col = 15
        self._z_scan._center_row = 15
        self._z_scan.LEARN_RATE = 0.08
        self._z_scan.BASE_AF_RANGE = 0.6

        # Connect signals
        self._z_scan.progress.connect(self._on_z_scan_progress)
        self._z_scan.finished.connect(self._on_z_scan_finished)

        # Set phase so orient_cal's own move/AF handlers ignore signals
        self._phase = 'z_spiral_scan'

        print(f"[OrientCal] Starting Z-tilt calibration scan (10×10, AF every tile, ±0.6fs)...")
        self.progress.emit('z_tilt', 'Z-tilt: starting calibration scan...')
        self._z_scan.start(self._camera_settings)

    def _on_z_scan_progress(self, phase, message):
        """Forward spiral scan progress to calibration progress."""
        self.progress.emit('z_tilt', f'Z-scan: {message}')
        # Detect when one cycle completes (scan enters waiting state)
        if phase == 'waiting':
            print(f"[OrientCal] Z-scan cycle complete, stopping scan...")
            QTimer.singleShot(100, self._finish_z_scan)

    def _finish_z_scan(self):
        """Stop the Z scan and compute Z-tilt from collected samples."""
        if self._z_scan is None:
            self._finalize()
            return

        # Grab Z samples before stopping
        z_samples = list(self._z_scan._z_samples)
        print(f"[OrientCal] Z-scan collected {len(z_samples)} Z samples")

        # Grab the learned pitch values from the spiral scan
        z_scan_cal = self._z_scan.calibration.get_result()
        if z_scan_cal.valid:
            self.result.grid_pitch_col_px = z_scan_cal.grid_pitch_col_px.copy()
            self.result.grid_pitch_row_px = z_scan_cal.grid_pitch_row_px.copy()
            self.result.grid_pitch_col_fs = z_scan_cal.grid_pitch_col_fs.copy()
            self.result.grid_pitch_row_fs = z_scan_cal.grid_pitch_row_fs.copy()
            print(f"[OrientCal] Pitch updated from Z-scan learning:")
            print(f"  col_fs=({self.result.grid_pitch_col_fs[0]:.2f}, "
                  f"{self.result.grid_pitch_col_fs[1]:.2f})")
            print(f"  row_fs=({self.result.grid_pitch_row_fs[0]:.2f}, "
                  f"{self.result.grid_pitch_row_fs[1]:.2f})")

        # Stop the scan
        self._z_scan.stop("Z calibration complete")

        # Disconnect signals
        try:
            self._z_scan.progress.disconnect(self._on_z_scan_progress)
            self._z_scan.finished.disconnect(self._on_z_scan_finished)
        except Exception:
            pass
        self._z_scan = None

        # Fit Z-tilt from samples
        self._fit_z_tilt_from_spiral(z_samples)

        # Save intermediate results, then start refinement cycles
        self.result.valid = True
        if self.config:
            save_calibration(self.result, self.config)
            print(f"[OrientCal] Z-tilt calibration saved, starting refinement cycles...")

        self._start_refinement_cycles()

    def _on_z_scan_finished(self, success, message):
        """Handle spiral scan finishing unexpectedly (e.g. error or user stop)."""
        if self._phase != 'z_spiral_scan':
            return
        print(f"[OrientCal] Z-scan finished: ok={success} {message}")
        if not success and 'Z calibration complete' not in message:
            # Unexpected stop — try to use whatever samples we have
            QTimer.singleShot(100, self._finish_z_scan)

    def _fit_z_tilt_from_spiral(self, z_samples):
        """Fit z = dz_per_col*col + dz_per_row*row + dz_per_tile*tiles + z0
        from spiral scan Z samples. Uses direct least-squares (no blending)."""
        n = len(z_samples)
        if n < 5:
            print(f"[OrientCal] Z-tilt: only {n} samples, need >= 5. Using defaults.")
            self.result.dz_per_col = -0.220
            self.result.dz_per_row = 0.024
            self.result.dz_per_tile = -0.03
            self.result.z_tilt_valid = False
            return

        A = np.zeros((n, 4))
        b = np.zeros(n)
        for i, s in enumerate(z_samples):
            A[i, 0] = s['col']
            A[i, 1] = s['row']
            A[i, 2] = s['tiles']
            A[i, 3] = 1.0
            b[i] = s['z']

        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            pred = A @ result
            rms = float(np.sqrt(np.mean((b - pred)**2)))

            self.result.dz_per_col = float(result[0])
            self.result.dz_per_row = float(result[1])
            self.result.dz_per_tile = float(result[2])
            self.result.z_tilt_valid = True

            print(f"[OrientCal] Z-tilt fit ({n} samples, rms={rms:.4f}):")
            print(f"  dz/col = {self.result.dz_per_col:.4f}")
            print(f"  dz/row = {self.result.dz_per_row:.4f}")
            print(f"  dz/tile = {self.result.dz_per_tile:.5f}")

        except Exception as e:
            print(f"[OrientCal] Z-tilt fit failed: {e}")
            self.result.z_tilt_valid = False

    # ═══════════════════════════════════════════════════════════════
    # Phase 6: Refinement cycles — normal spiral scan to tune dz
    # ═══════════════════════════════════════════════════════════════

    def _start_refinement_cycles(self):
        """Run 2 normal spiral scan cycles (AF every 2, ±0.4fs) to refine dz."""
        if self._stop_requested:
            self._finalize()
            return

        from calibration.spiral_scan import SpiralScanController

        self._refine_cycles_target = 1
        self._refine_cycles_done = 0

        self._z_scan = SpiralScanController(
            calibration=self,
            af_controller=self.af_controller,
            move_to_ctrl=self.move_to_ctrl,
            camera_preview=self.camera_preview,
            mc=self.mc,
            motors_config=self.MOTORS,
            config=self.config,
            parent=self,
        )

        # Normal scan settings — match what the real scan will use
        self._z_scan.SCAN_SIZE = 10
        self._z_scan._scan_size = 10
        self._z_scan._center_col = 15
        self._z_scan._center_row = 15
        # AF_EVERY_N_TILES = 2 (default)
        # LEARN_RATE = 0.02 (default)
        # BASE_AF_RANGE = 0.4 (default)
        self._z_scan._scan_interval_min = 0  # no wait between cycles

        self._z_scan.progress.connect(self._on_refine_scan_progress)
        self._z_scan.finished.connect(self._on_refine_scan_finished)

        self._phase = 'refine_spiral_scan'

        print(f"[OrientCal] Starting refinement scan (1 cycle, AF every 2, normal settings)...")
        self.progress.emit('refining', 'Refinement scan: cycle 1/1...')
        self._z_scan.start(self._camera_settings)

    def _on_refine_scan_progress(self, phase, message):
        """Track refinement cycle completion."""
        self.progress.emit('refining', f'Refine: {message}')
        if phase == 'waiting':
            self._refine_cycles_done += 1
            print(f"[OrientCal] Refinement cycle {self._refine_cycles_done}/{self._refine_cycles_target} complete")
            if self._refine_cycles_done >= self._refine_cycles_target:
                print(f"[OrientCal] All refinement cycles done, finalizing...")
                QTimer.singleShot(100, self._finish_refinement_scan)
            else:
                self.progress.emit('refining',
                    f'Refinement scan: cycle {self._refine_cycles_done + 1}/{self._refine_cycles_target}...')

    def _finish_refinement_scan(self):
        """Stop refinement scan and grab final learned values."""
        if self._z_scan is None:
            self._finalize()
            return

        # Grab learned dz values
        self.result.dz_per_col = self._z_scan._dz_per_col
        self.result.dz_per_row = self._z_scan._dz_per_row
        self.result.dz_per_tile = self._z_scan._dz_per_tile
        self.result.z_tilt_valid = True

        # Grab learned pitch values
        z_scan_cal = self._z_scan.calibration.get_result()
        if z_scan_cal.valid:
            self.result.grid_pitch_col_px = z_scan_cal.grid_pitch_col_px.copy()
            self.result.grid_pitch_row_px = z_scan_cal.grid_pitch_row_px.copy()
            self.result.grid_pitch_col_fs = z_scan_cal.grid_pitch_col_fs.copy()
            self.result.grid_pitch_row_fs = z_scan_cal.grid_pitch_row_fs.copy()

        print(f"[OrientCal] Refinement results:")
        print(f"  dz/col={self.result.dz_per_col:.4f} dz/row={self.result.dz_per_row:.4f} "
              f"dz/tile={self.result.dz_per_tile:.5f}")
        print(f"  col_fs=({self.result.grid_pitch_col_fs[0]:.2f}, {self.result.grid_pitch_col_fs[1]:.2f})")
        print(f"  row_fs=({self.result.grid_pitch_row_fs[0]:.2f}, {self.result.grid_pitch_row_fs[1]:.2f})")

        # Stop and cleanup
        self._z_scan.stop("Refinement complete")
        try:
            self._z_scan.progress.disconnect(self._on_refine_scan_progress)
            self._z_scan.finished.disconnect(self._on_refine_scan_finished)
        except Exception:
            pass
        self._z_scan = None

        self._phase = 'idle'
        self._finalize()

    def _on_refine_scan_finished(self, success, message):
        """Handle refinement scan finishing unexpectedly."""
        if self._phase != 'refine_spiral_scan':
            return
        print(f"[OrientCal] Refine scan finished: ok={success} {message}")
        if not success and 'Refinement complete' not in message:
            QTimer.singleShot(100, self._finish_refinement_scan)

    # ═══════════════════════════════════════════════════════════════
    # Marker filtering (needed by spiral_scan)
    # ═══════════════════════════════════════════════════════════════

    def _filter_suspicious_markers(self, markers, expected_pos=None):
        if len(markers) < 2:
            return markers
        pitch_col = self.result.grid_pitch_col_px
        pitch_row = self.result.grid_pitch_row_px
        if np.linalg.norm(pitch_col) < 1.0 or np.linalg.norm(pitch_row) < 1.0:
            return markers

        if expected_pos is not None:
            cx, cy = self.result.image_center_x, self.result.image_center_y
            center_px = np.array([cx, cy])
            best_anchor, best_score = None, float('inf')
            for m in markers:
                m_px = np.array([m['center'][0], m['center'][1]])
                dcol = m['col_id'] - expected_pos[0]
                drow = m['row_id'] - expected_pos[1]
                pred = center_px + (dcol - 0.5) * pitch_col + (drow - 0.5) * pitch_row
                score = np.linalg.norm(m_px - pred)
                if score < best_score:
                    best_score = score
                    best_anchor = m
            if best_score > 500:
                best_anchor = None
        else:
            best_anchor = None

        if best_anchor is None:
            max_inliers = -1
            for m1 in markers:
                m1_px = np.array([m1['center'][0], m1['center'][1]])
                count = 0
                for m2 in markers:
                    m2_px = np.array([m2['center'][0], m2['center'][1]])
                    exp = m1_px + (m2['col_id'] - m1['col_id']) * pitch_col + \
                                  (m2['row_id'] - m1['row_id']) * pitch_row
                    if np.linalg.norm(exp - m2_px) < 100:
                        count += 1
                if count > max_inliers:
                    max_inliers = count
                    best_anchor = m1
            if max_inliers < 2 and len(markers) > 2:
                best_anchor = markers[0]

        filtered = []
        anchor_px = np.array([best_anchor['center'][0], best_anchor['center'][1]])
        A = np.column_stack([pitch_col, pitch_row])
        det = np.linalg.det(A)

        for m in markers:
            m_px = np.array([m['center'][0], m['center'][1]])
            exp = anchor_px + (m['col_id'] - best_anchor['col_id']) * pitch_col + \
                               (m['row_id'] - best_anchor['row_id']) * pitch_row
            if np.linalg.norm(exp - m_px) < 100:
                filtered.append(m)
            elif abs(det) > 1e-6:
                delta = m_px - anchor_px
                d = np.linalg.solve(A, delta)
                ic = int(best_anchor['col_id'] + round(d[0]))
                ir = int(best_anchor['row_id'] + round(d[1]))
                res = delta - (round(d[0]) * pitch_col + round(d[1]) * pitch_row)
                if np.linalg.norm(res) < 100:
                    if expected_pos is not None:
                        if (abs(ic - expected_pos[0]) + abs(ir - expected_pos[1])) > \
                           (abs(m['col_id'] - expected_pos[0]) + abs(m['row_id'] - expected_pos[1])):
                            filtered.append(m)
                            continue
                    corrected = dict(m)
                    corrected['col_id'] = ic
                    corrected['row_id'] = ir
                    corrected['_inferred'] = True
                    filtered.append(corrected)
                else:
                    print(f"[Filter] Discarded at px ({m_px[0]:.0f}, {m_px[1]:.0f})")

        return filtered if filtered else markers

    # ═══════════════════════════════════════════════════════════════
    # Finalize
    # ═══════════════════════════════════════════════════════════════

    def _finalize(self):
        if self._stop_requested:
            return
        self._print_result("Final Calibration")
        if self.config:
            save_calibration(self.result, self.config)
            save_orientation(self._orientation, self.config)
        self._active = False
        self._phase = 'idle'
        self._restore_speed()
        self.finished.emit(True,
                           f"Calibrated. Orientation: {self._orientation}",
                           self.result)

    def _print_result(self, label):
        print(f"\n[OrientCal] ══ {label} ══")
        print(f"  Orientation: {self._orientation}")
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
        print(f"[OrientCal] ══════════════════\n")

    # ═══════════════════════════════════════════════════════════════
    # Move/AF dispatcher
    # ═══════════════════════════════════════════════════════════════

    def _on_move_finished(self, success, message):
        if not self._active or self._phase == 'idle':
            return
        if self._phase in ('z_spiral_scan', 'refine_spiral_scan'):
            return  # spiral scan handles its own moves
        if self._stop_requested:
            return
        if not success:
            self.stop(f"Move failed ({self._phase}): {message}")
            return

        if self._phase == 'move_to_center':
            if self._af_kickoff_pending:
                print(f"[OrientCal id={id(self)}] move_to_center finished "
                      f"ignored — autofocus kickoff already pending")
                return
            self._af_kickoff_pending = True
            QTimer.singleShot(SETTLE_MS, self._start_autofocus)

        elif self._phase == 'row_scan_move':
            self._do_quick_af('row_scan_af')

        elif self._phase == 'row_reverse_move':
            self._do_quick_af('row_reverse_af')

        elif self._phase == 'row_scan_snap':
            self._do_quick_af('row_snap_af')

        elif self._phase == 'col_scan_move':
            self._do_quick_af('col_scan_af')

        elif self._phase == 'col_reverse_move':
            self._do_quick_af('col_reverse_af')

        elif self._phase == 'col_scan_snap':
            self._do_quick_af('col_snap_af')

        elif self._phase == 'refine_move':
            self._do_quick_af('refine_af')

        elif self._phase == 'refine_return_for_next':
            self._do_quick_af('refine_return_af')

        elif self._phase == 'refine_return':
            self._do_quick_af('refine_done_af')

    def _on_af_finished(self, success, message):
        if not self._active:
            return
        if self._phase in ('z_spiral_scan', 'refine_spiral_scan'):
            return  # spiral scan handles its own AF

        if self._phase == 'initial_af':
            QTimer.singleShot(200, self._capture_reference)

        elif self._phase == 'row_scan_af':
            QTimer.singleShot(100, lambda: self._scan_capture_and_advance('row'))

        elif self._phase == 'row_reverse_af':
            self._advance_reverse_step()

        elif self._phase == 'row_snap_af':
            self._scan_step = 0
            self._last_scan_motor_pos = self._cur_motor_pos()
            self.progress.emit('orient', 'Orientation scan: col direction...')
            self._scan_capture_and_advance('col')

        elif self._phase == 'col_scan_af':
            QTimer.singleShot(100, lambda: self._scan_capture_and_advance('col'))

        elif self._phase == 'col_reverse_af':
            self._advance_reverse_step()

        elif self._phase == 'col_snap_af':
            QTimer.singleShot(200, self._analyze_and_store_orientation)

        elif self._phase == 'pre_refine_af':
            QTimer.singleShot(100, self._begin_refinement_after_af)

        elif self._phase == 'refine_af':
            QTimer.singleShot(100, self._refine_capture)

        elif self._phase == 'refine_return_af':
            QTimer.singleShot(100, self._refine_detect_and_move)

        elif self._phase == 'refine_done_af':
            self.progress.emit('z_tilt', 'Starting Z-tilt calibration scan...')
            QTimer.singleShot(100, self._start_z_tilt_survey)