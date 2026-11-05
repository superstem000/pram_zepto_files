"""Spiral Scan Controller.

Scans a configurable grid centered on a given position. Full AF from
KNOWN_FOCUS (±10fs) on cold start; per-tile fine AF during the scan.
Performs stitch after each cycle.
"""

import os, json, numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from calibration.orient_calibration import detect_markers_corrected
from vision.feature_detection import save_failed_frame

def generate_spiral_path(size=5, offset_col=0, offset_row=0):
    """Generate inward spiral from (0,0) corner to center.

    All steps are adjacent (Manhattan distance 1).
    Visits outer ring first, spirals inward, ends at center.
    """
    path, visited = [], set()
    # Directions: right, down, left, up — spirals inward clockwise
    dirs = [(0,1),(1,0),(0,-1),(-1,0)]
    di = 0
    r, c = 0, 0
    for _ in range(size*size):
        path.append((c + offset_col, r + offset_row))
        visited.add((c, r))
        dc, dr = dirs[di]
        nc, nr = c+dc, r+dr
        if 0<=nc<size and 0<=nr<size and (nc,nr) not in visited:
            c, r = nc, nr
        else:
            di = (di+1)%4
            dc, dr = dirs[di]
            nc, nr = c+dc, r+dr
            if 0<=nc<size and 0<=nr<size and (nc,nr) not in visited:
                c, r = nc, nr
            else:
                break
    return path


def measure_pixel_pitch_from_markers(markers):
    """Compute grid pitch in pixels directly from observed marker positions.

    Given a list of detected markers with 'col_id', 'row_id', and 'center',
    compute the average pixel displacement per col step and per row step
    using ONLY axis-aligned marker pairs (same row for col pitch, same col
    for row pitch).  Diagonal pairs cannot cleanly decompose into per-axis
    pitch and introduce large systematic errors, so they are ignored.

    Returns (col_pitch_px, row_pitch_px) as numpy arrays, or (None, None)
    for each axis that has no axis-aligned pairs.
    """
    if len(markers) < 2:
        return None, None

    col_deltas = []  # pixel displacement per col step (from same-row pairs)
    row_deltas = []  # pixel displacement per row step (from same-col pairs)

    for i, m1 in enumerate(markers):
        for m2 in markers[i+1:]:
            dcol = m2['col_id'] - m1['col_id']
            drow = m2['row_id'] - m1['row_id']
            dpx = np.array([m2['center'][0] - m1['center'][0],
                            m2['center'][1] - m1['center'][1]])

            # Pure col movement (same row) — clean measurement
            if dcol != 0 and drow == 0:
                col_deltas.append(dpx / dcol)
            # Pure row movement (same col) — clean measurement
            elif drow != 0 and dcol == 0:
                row_deltas.append(dpx / drow)
            # Diagonal pairs are intentionally skipped:
            # dpx/dcol would include the row contribution and vice versa,
            # causing ~10%+ magnitude error and wrong cross-term direction.

    col_pitch = np.mean(col_deltas, axis=0) if col_deltas else None
    row_pitch = np.mean(row_deltas, axis=0) if row_deltas else None

    return col_pitch, row_pitch


class SpiralScanController(QObject):
    progress = pyqtSignal(str, str)
    finished = pyqtSignal(bool, str)
    capture_ready = pyqtSignal(int, int, str)
    SETTLE_MS = 200
    DISCARD_FRAMES = 1
    GRID_SIZE = 31          # full grid size (for detection/calibration)
    SCAN_SIZE = 5           # 5×5 scan
    SKIP_THRESHOLD_FS = 2.0
    MAX_FINE_ERROR_FS = 5.0
    MAX_RECOVERY_ATTEMPTS = 3
    MOVE_SANITY_FACTOR = 1.8
    WALK_JUMP_SIZE = 2
    WALK_ESCALATE_NO_PROGRESS = 4   # full AF after this many stalled walk checks
    WALK_MAX_NO_PROGRESS = 8        # abort walk if still stuck after this many
    AF_EVERY_N_TILES = 1
    LEARN_RATE = 0.02
    DZ_PER_COL = -0.220
    DZ_PER_ROW = 0.024
    DZ_PER_TILE = -0.03
    AF_MODE = 'minimal'
    SCAN_SPEED_MULTIPLIER = 0.5

    # ── Retry caps (uniform = 5 attempts before skip/halt) ────────────────
    MAX_LOCALIZE_RETRIES  = 5    # detection/localization failures during spiral
    MAX_MOVE_CAP_RETRIES  = 5    # MOVE CAP target-out-of-bounds loop
    MAX_WALK_RETRIES      = 5    # detection failures during walk-to-target
    MAX_COLD_AF_RETRIES   = 5    # detect at safe-center after cold AF
    MAX_CONSECUTIVE_SKIPS = 4    # halt the scan if N tiles skip in a row
    # AF range escalation per retry (fs); indexed by retry_count - 1.
    AF_RANGE_STEPS_FS     = [1.0, 2.5, 5.0, 7.5, 10.0]
    DEAD_TILE_RESTARTS = 5

    def __init__(self, calibration, af_controller, move_to_ctrl,
                 camera_preview, mc, motors_config, config, parent=None):
        super().__init__(parent)
        self.calibration = calibration
        self.af_controller = af_controller
        self.move_to_ctrl = move_to_ctrl
        self.camera_preview = camera_preview
        self.mc = mc
        self.MOTORS = motors_config
        self.config = config
        self._active = False
        self._stop_requested = False
        self._saved_speed_mult = 1.0
        self._state = 'idle'
        self._cycle_count = 0
        self._phase_sub = ''
        self._tile_target_x = 0.0
        self._tile_target_y = 0.0
        self._tile_target_z = 0.0
        self._current_target = (0, 0)
        self._after_move_cb = None
        self._move_start_col = 0
        self._move_start_row = 0
        self._learn_anchor_col = 0
        self._learn_anchor_row = 0
        self._recovery_attempts = 0
        self._last_recovery_error = None
        self._fine_callback = None
        self._last_known_col = 0
        self._last_known_row = 0
        self._has_known_position = False
        self._consecutive_af_failures = 0
        self._last_good_z = 0.0  # ADDED: Z position at last successful detection
        self._tiles_since_af = 0
        self._last_move_was_noop = False
        self._walk_no_progress = 0
        self._walk_last_infer = None
        self._spiral_path: List[Tuple[int, int]] = []
        self._spiral_idx = 0
        self._camera_settings = {}
        self._cached_markers = None  # reuse detection from fine_correct
        self._in_return_phase = False
        self._output_parent = os.path.expanduser("~/Desktop/Users/scans")
        self._output_base = None  # set per-session in start()
        self._z_log_path = ""
        # Z correction state — load from calibration if available (no learning in 5x5)
        self._tile_step_count = 0
        self._last_z_correct_pos = (0, 0)
        cal = self.calibration.get_result()
        if cal.valid and getattr(cal, 'z_tilt_valid', False):
            self._dz_per_col = cal.dz_per_col
            self._dz_per_row = cal.dz_per_row
            self._dz_per_tile = cal.dz_per_tile
            print(f"[Scan] Z-tilt loaded from calibration: "
                  f"col={self._dz_per_col:.4f} row={self._dz_per_row:.4f} "
                  f"tile={self._dz_per_tile:.5f}")
        else:
            self._dz_per_col = self.DZ_PER_COL
            self._dz_per_row = self.DZ_PER_ROW
            self._dz_per_tile = self.DZ_PER_TILE
            print(f"[Scan] Z-tilt using defaults (no calibration data)")
        self._last_af_z = 0.0
        self._last_af_col = 0
        self._last_af_row = 0
        # No least-squares Z learning in 5x5 — not enough data
        self._z_samples = []
        self._cumulative_tiles = 0
        self.move_to_ctrl.finished.connect(self._on_move_finished)
        self.af_controller.finished.connect(self._on_af_finished)
        self._last_fine_frame = None
        self._cycle_start_time = None
        self._scan_interval_min = 5        # default 5 minutes
        self._center_col = 15              # center of the 31x31 grid (indices 0..30)
        self._center_row = 15
        self._scan_size = self.SCAN_SIZE   # configurable per-session (odd, >=3)
        self._target_cycles = 0            # 0 = unlimited; >0 = stop after N cycles
        # Corner bounds: record XY at each corner to clamp motion limits
        self._corner_positions = {}
        self._corners_set = set()
        self._scan_bounds_applied = False
        self._single_tile_af_retries = 0
        self._scan_move_cap_active = False
        # Dead-region skip state
        self._last_good_tile = None     # (col,row,x,y,z) — single anchor slot
        self._dead_restarts = 0
        self._failed_tiles = set()
        self._post_skip = False         # suppress Z/pitch learning on first tile after a skip
        self._pending_z_sample = None   # staged at periodic AF, committed only on confirmation
        self._skip_candidate = None
        self._in_walk = False           # walk phase has its own bounded give-up
        self._walk_retries = 0          # cap-bounded walk detection failures
        self._cold_af_retries = 0       # cap-bounded cold-AF detect failures
        self._consecutive_skips = 0     # tiles skipped in a row (capped → halt)
        self._orientation = "normal"  # updated in start()

    def is_active(self): return self._active

    def _detect(self, frame):
        """Detect markers with orientation correction."""
        return detect_markers_corrected(
            frame, self._orientation,
            threshold=self.calibration.DETECTION_THRESHOLD)

    BASE_AF_RANGE = 0.4

    def _get_af_range(self):
        """AF range grows with consecutive failures. Steeper escalation now that
        retry caps are 5 instead of ~10 — see AF_RANGE_STEPS_FS."""
        n = self._consecutive_af_failures
        if n <= 0:
            return self.BASE_AF_RANGE
        idx = min(n - 1, len(self.AF_RANGE_STEPS_FS) - 1)
        return self.AF_RANGE_STEPS_FS[idx]

    def _start_escalating_af(self):
        """Start AF with range based on failure count, centered on last good Z."""
        af_range = self._get_af_range()
        center_z = self._last_good_z if self._consecutive_af_failures > 0 and self._last_good_z > 0 else None
        if af_range >= 10.0:
            print(f"[Scan] Full AF (failures={self._consecutive_af_failures})")
            self.af_controller.start(self._camera_settings)
        else:
            print(f"[Scan] AF ±{af_range:.1f}fs around Z={center_z or 'current'} (failures={self._consecutive_af_failures})")
            self.af_controller.start_quick(self._camera_settings, custom_range=af_range, center_z=center_z)

    def _set_scan_speed(self):
        """Restore scan speed for tile-to-tile moves."""
        try:
            self.mc.set_speed_multiplier(self.SCAN_SPEED_MULTIPLIER)
        except Exception:
            pass

    def _check_move_cap(self, dx_fs, dy_fs):
        """Reject an XY move if it would land outside the active scan/travel
        bounds (primary) or exceeds the 3-tile magnitude cap (backup).
        Returns True if BLOCKED."""
        if not self._scan_move_cap_active:
            return False
        cal = self.calibration.get_result()
        if not cal.valid:
            return False

        # Primary: a target outside the scanned region can only come from a
        # bad anchor/detection. Never issue it — the motion layer would
        # silently clamp the axis and MoveTo would stall instead of recover.
        try:
            lim = self.move_to_ctrl.motion.limits
            cur_x = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
            cur_y = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))
            tgt_x = cur_x + float(dx_fs)
            tgt_y = cur_y + float(dy_fs)
            eps = 1e-6
            if (tgt_x < lim.x0 - eps or tgt_x > lim.x1 + eps or
                    tgt_y < lim.y0 - eps or tgt_y > lim.y1 + eps):
                print(f"[Scan] MOVE CAP: target ({tgt_x:.1f},{tgt_y:.1f})fs "
                      f"outside scan bounds X[{lim.x0:.1f},{lim.x1:.1f}] "
                      f"Y[{lim.y0:.1f},{lim.y1:.1f}] — blocked")
                return True
        except Exception:
            pass

        # Backup: >3 tiles of correction in a single move ⇒ bad anchor.
        col_fs = cal.grid_pitch_col_fs
        row_fs = cal.grid_pitch_row_fs
        max_x_per_tile = max(abs(col_fs[0]), abs(row_fs[0]))
        max_y_per_tile = max(abs(col_fs[1]), abs(row_fs[1]))
        limit_x = max_x_per_tile * 3.0
        limit_y = max_y_per_tile * 3.0
        if abs(dx_fs) > limit_x or abs(dy_fs) > limit_y:
            print(f"[Scan] MOVE CAP: blocked ({dx_fs:.1f},{dy_fs:.1f})fs, "
                  f"limit ({limit_x:.1f},{limit_y:.1f})fs")
            return True
        return False

    def configure(self, interval_min=None, center_col=None, center_row=None,
                  scan_size=None, num_cycles=None):
        """Configure scan interval, center, size, and target cycle count.

        scan_size: odd int >=3 (e.g. 3, 5, 7). Clamped so the scan fits within
                   the GRID_SIZE (centered on center_col/center_row).
        num_cycles: 0 = unlimited (default); >0 = stop after N cycles.
        """
        if interval_min is not None:
            self._scan_interval_min = max(1, min(30, interval_min))
        if scan_size is not None:
            s = int(scan_size)
            if s < 3: s = 3
            if s > self.GRID_SIZE: s = self.GRID_SIZE
            if s % 2 == 0: s += 1  # force odd
            self._scan_size = s
        if center_col is not None and center_row is not None:
            half = self._scan_size // 2
            lo, hi = half, self.GRID_SIZE - 1 - half
            center_col = max(lo, min(hi, center_col))
            center_row = max(lo, min(hi, center_row))
            self._center_col = center_col
            self._center_row = center_row
        if num_cycles is not None:
            self._target_cycles = max(0, int(num_cycles))

    def start(self, camera_settings, output_dir=""):
        if self._active: return
        if not self.camera_preview.is_running():
            self.finished.emit(False, "Start live preview first."); return
        if not self.calibration.get_result().valid:
            self.finished.emit(False, "Calibrate stage first."); return
        self._active = True
        self._stop_requested = False
        self._camera_settings = camera_settings

        # Load plate orientation for marker ID correction
        try:
            self._orientation = self.calibration.get_orientation()
        except AttributeError:
            from calibration.orient_calibration import load_orientation
            self._orientation = load_orientation(self.config)
        print(f"[Scan] Orientation: {self._orientation}")
        self._cycle_count = 0
        self._has_known_position = False
        self._consecutive_af_failures = 0
        self._tiles_since_af = 0
        self._tile_step_count = 0
        # Reset corner bounds for fresh recording on first cycle
        self._corner_positions = {}
        self._corners_set = set()
        self._scan_bounds_applied = False
        self._single_tile_af_retries = 0
        self._walk_retries = 0
        self._cold_af_retries = 0
        self._consecutive_skips = 0
        self._cumulative_tiles = 0
        self._z_samples = []  # fresh samples each scan session
        self._last_good_tile = None
        self._dead_restarts = 0
        self._failed_tiles = set()
        self._post_skip = False
        self._pending_z_sample = None
        self._in_walk = False
        if output_dir: self._output_parent = output_dir
        session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._output_base = os.path.join(self._output_parent, f"demo_{session_ts}")
        os.makedirs(self._output_base, exist_ok=True)
        print(f"[Scan] Session output: {self._output_base}")
        self._z_log_path = os.path.join(self._output_base, "z_log.csv")
        with open(self._z_log_path, 'w') as f:
            f.write("timestamp,col,row,z_fs,state\n")

        # Comprehensive scan log for visualization/analysis
        self._scan_log_path = os.path.join(self._output_base, "scan_log.csv")
        with open(self._scan_log_path, 'w') as f:
            f.write("timestamp,cycle,tile_idx,col,row,x_fs,y_fs,z_fs,"
                    "error_fs_x,error_fs_y,error_mag,"
                    "dz_per_col,dz_per_row,dz_per_tile,"
                    "af_z,af_score,event\n")

        # Save user's speed and set scan speed for tile moves
        try:
            self._saved_speed_mult = self.mc.get_speed_multiplier()
            self.mc.set_speed_multiplier(self.SCAN_SPEED_MULTIPLIER)
        except Exception:
            self._saved_speed_mult = 1.0

        # Move to safe center first, then cold AF
        from calibration.orient_calibration import SAFE_CENTER_X_FS, SAFE_CENTER_Y_FS
        print(f"[Scan] Moving to safe center ({SAFE_CENTER_X_FS}, {SAFE_CENTER_Y_FS})...")
        self.progress.emit('moving', 'Moving to center...')
        self._state = 'move_to_center'
        self.move_to_ctrl.start(SAFE_CENTER_X_FS, SAFE_CENTER_Y_FS, 0.0)

    def stop(self, reason="Stopped by user"):
        if not self._active: return
        self._stop_requested = True; self._active = False; self._state = 'idle'
        self._after_move_cb = None
        self._fine_callback = None
        self._scan_move_cap_active = False
        # Halt motors immediately without going through MoveTo/AF signal chains
        try:
            self.move_to_ctrl.motion.stop_immediate()
        except Exception:
            pass
        # Restore scan bounds to defaults
        self._restore_scan_bounds()
        # Restore user's speed AFTER halting motors
        try:
            self.mc.set_speed_multiplier(self._saved_speed_mult)
        except Exception:
            pass
        print(f"[Scan] Stopped: {reason}")
        self.finished.emit(False, reason)

    def _log_z(self, col, row, z_fs, state=""):
        try:
            with open(self._z_log_path, 'a') as f:
                f.write(f"{datetime.now().strftime('%H:%M:%S.%f')},{col},{row},{z_fs:.4f},{state}\n")
        except Exception as e:
            print(f"[Scan] Z log error: {e}")

    def _log_scan(self, col, row, event, error_fs=None, af_z=None, af_score=None):
        """Log a scan event with full metadata for later visualization."""
        try:
            # Tag return-phase events so they can be filtered in visualization
            if self._in_return_phase and not event.startswith('return'):
                event = f"return_{event}"
            ts = datetime.now().strftime('%H:%M:%S.%f')
            x = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
            y = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))
            z = float(getattr(self.MOTORS, "z_current_position_full_step", 0.0))
            err_x = f"{error_fs[0]:.3f}" if error_fs is not None else ""
            err_y = f"{error_fs[1]:.3f}" if error_fs is not None else ""
            err_m = f"{float(np.linalg.norm(error_fs)):.3f}" if error_fs is not None else ""
            az = f"{af_z:.4f}" if af_z is not None else ""
            asc = f"{af_score:.1f}" if af_score is not None else ""
            with open(self._scan_log_path, 'a') as f:
                f.write(f"{ts},{self._cycle_count},{self._tile_step_count},"
                        f"{col},{row},{x:.3f},{y:.3f},{z:.4f},"
                        f"{err_x},{err_y},{err_m},"
                        f"{self._dz_per_col:.6f},{self._dz_per_row:.6f},{self._dz_per_tile:.6f},"
                        f"{az},{asc},{event}\n")
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════
    # Initial AF at start position
    # ═══════════════════════════════════════════════════════════════

    def _initial_af(self):
        if self._stop_requested: return
        col, row = self._center_col, self._center_row
        print(f"[Scan] At center ({col},{row}), performing quick AF")
        self.progress.emit('af', f'Autofocus at ({col},{row})...')
        self._state = 'initial_af'
        self.af_controller.start_quick(self._camera_settings, mode='tight')

    def _detect_and_walk_to_center(self):
        """After cold AF, detect current tile position and walk to scan center.
        Capped at MAX_COLD_AF_RETRIES — no fallback exists pre-acquisition, so
        on cap we halt with a clear error rather than looping forever."""
        if self._stop_requested: return

        def _cold_fail(reason):
            self._cold_af_retries += 1
            if self._cold_af_retries > self.MAX_COLD_AF_RETRIES:
                self.stop(f"Cannot detect at scan center after "
                          f"{self._cold_af_retries - 1} retries ({reason})")
                return True
            return False

        cal = self.calibration.get_result()
        frame = self.camera_preview.flush_and_capture(
            self._camera_settings, discard=self.DISCARD_FRAMES)
        if frame is None:
            if _cold_fail("no frame"): return
            print(f"[Scan] No frame after cold AF — retrying "
                  f"({self._cold_af_retries}/{self.MAX_COLD_AF_RETRIES})")
            QTimer.singleShot(200, self._detect_and_walk_to_center)
            return
        markers = self._detect(frame)
        if not markers:
            if _cold_fail("no markers"): return
            print(f"[Scan] No markers after cold AF — retrying AF "
                  f"({self._cold_af_retries}/{self.MAX_COLD_AF_RETRIES})")
            save_failed_frame(frame, "demo_coldaf_no_markers")
            self._state = 'cold_af'
            self.af_controller.start(self._camera_settings)
            return
        markers = self.calibration._filter_suspicious_markers(markers)
        if not markers:
            if _cold_fail("all filtered"): return
            print(f"[Scan] All markers filtered — retrying AF "
                  f"({self._cold_af_retries}/{self.MAX_COLD_AF_RETRIES})")
            save_failed_frame(frame, "demo_coldaf_all_filtered")
            self._state = 'cold_af'
            self.af_controller.start(self._camera_settings)
            return
        self._cold_af_retries = 0   # success — reset budget
        cur_col, cur_row, _, _ = self._infer_tile_at_ideal(cal, markers)
        self._last_known_col = cur_col
        self._last_known_row = cur_row
        self._has_known_position = True
        self._last_z_correct_pos = (cur_col, cur_row)
        self._current_target = (cur_col, cur_row)
        print(f"[Scan] Detected position: ({cur_col},{cur_row}), "
              f"walking to center ({self._center_col},{self._center_row})")
        self.progress.emit('walking',
            f'Walking from ({cur_col},{cur_row}) to ({self._center_col},{self._center_row})...')
        self._walk_to_target_then(self._center_col, self._center_row, self._initial_af)

    # ═══════════════════════════════════════════════════════════════
    # Expected move distance from calibration
    # ═══════════════════════════════════════════════════════════════

    def _expected_move_mag(self, cal, dcol, drow):
        expected_fs = dcol * cal.grid_pitch_col_fs + drow * cal.grid_pitch_row_fs
        return float(np.linalg.norm(expected_fs))

    # ═══════════════════════════════════════════════════════════════
    # Z drift correction for a tile move
    # ═══════════════════════════════════════════════════════════════

    def _compute_dz_for_move(self, dcol, drow):
        return self._dz_per_col * dcol + self._dz_per_row * drow + self._dz_per_tile

    # ═══════════════════════════════════════════════════════════════
    # Quadrant-based marker selection for position inference
    # ═══════════════════════════════════════════════════════════════

    def _infer_tile_at_ideal(self, cal, markers):
        cx, cy = cal.image_center_x, cal.image_center_y

        # Use observed pixel pitch from current markers if available
        obs_col_px, obs_row_px = measure_pixel_pitch_from_markers(markers)
        col_px = obs_col_px if obs_col_px is not None else cal.grid_pitch_col_px
        row_px = obs_row_px if obs_row_px is not None else cal.grid_pitch_row_px

        ideal_px = np.array([cx, cy]) - 0.5 * col_px - 0.5 * row_px
        A = np.column_stack([col_px, row_px])

        votes = {}
        for m in markers:
            mx, my = m['center'][0], m['center'][1]
            delta_from_center = np.array([mx - cx, my - cy])
            try:
                grid_pos = np.linalg.solve(A, delta_from_center)
            except np.linalg.LinAlgError:
                continue
            dcol_from_ur = round(grid_pos[0] + 0.5)
            drow_from_ur = round(grid_pos[1] + 0.5)
            inferred_col = m['col_id'] - dcol_from_ur
            inferred_row = m['row_id'] - drow_from_ur
            dist = (mx - ideal_px[0])**2 + (my - ideal_px[1])**2
            key = (inferred_col, inferred_row)
            if key not in votes:
                votes[key] = []
            votes[key].append((m, dist, dcol_from_ur, drow_from_ur))

        if not votes:
            ref_m = min(markers, key=lambda m:
                        (m['center'][0]-ideal_px[0])**2 + (m['center'][1]-ideal_px[1])**2)
            return ref_m['col_id'], ref_m['row_id'], ref_m, False

        best_key = max(votes.keys(),
                       key=lambda k: (len(votes[k]), -min(v[1] for v in votes[k])))
        best_entries = votes[best_key]
        best_entry = min(best_entries, key=lambda e: e[1])
        best_m, best_dist, dcol_off, drow_off = best_entry
        cur_col, cur_row = best_key

        exact = False
        for m in markers:
            if m['col_id'] == cur_col and m['row_id'] == cur_row:
                best_m = m
                exact = True
                break

        if exact:
            print(f"[Scan] Infer: anchor ({best_m['col_id']},{best_m['row_id']}) "
                  f"[exact match] -> cur = ({cur_col},{cur_row}) "
                  f"[{len(best_entries)} votes]")
        else:
            print(f"[Scan] Infer: anchor ({best_m['col_id']},{best_m['row_id']}) "
                  f"[quadrant ({dcol_off},{drow_off})] "
                  f"-> cur = ({cur_col},{cur_row}) [no exact, {len(best_entries)} votes]")

        return int(cur_col), int(cur_row), best_m, exact

    # ═══════════════════════════════════════════════════════════════
    # Multi-tile move with virtual anchor
    # ═══════════════════════════════════════════════════════════════

    def _detect_and_move_multi(self, dest_col, dest_row, max_tiles, callback):
        if self._stop_requested: return None
        self._after_move_cb = callback
        self._phase_sub = ''
        self._last_move_was_noop = False
        # NOTE: _current_target is NOT set here — only after we confirm position

        cal = self.calibration.get_result()
        if not cal.valid:
            self.stop("Calibration invalid"); return None

        # Use cached detection from previous fine_correct if available
        if self._cached_markers is not None:
            markers = self._cached_markers
            self._cached_markers = None
        else:
            frame = self.camera_preview.flush_and_capture(
                self._camera_settings, discard=self.DISCARD_FRAMES)
            if frame is None:
                print(f"[Scan] Detection failed (no frame) — retrying")
                QTimer.singleShot(100, lambda: self._detect_and_move_multi(
                    dest_col, dest_row, max_tiles, callback))
                return None

            markers = self._detect(frame)
            if not markers:
                p = save_failed_frame(frame, "demo_multi_no_markers")
                print(f"[Scan] Detection failed (no markers) — saved {p}")
                self._on_localize_fail(lambda: self._detect_and_move_multi(
                    dest_col, dest_row, max_tiles, callback))
                return None
            expected_pos = (self._last_known_col, self._last_known_row) if self._has_known_position else None
            markers = self.calibration._filter_suspicious_markers(markers, expected_pos=expected_pos)
            if not markers:
                p = save_failed_frame(frame, "demo_multi_all_filtered")
                print(f"[Scan] Detection failed (all filtered) — saved {p}")
                self._on_localize_fail(lambda: self._detect_and_move_multi(
                    dest_col, dest_row, max_tiles, callback))
                return None

        cur_col, cur_row, ref_m, exact = self._infer_tile_at_ideal(cal, markers)

        # Sanity check: position jump
        if self._has_known_position:
            dcol_jump = abs(cur_col - self._last_known_col)
            drow_jump = abs(cur_row - self._last_known_row)
            jump_limit = max_tiles + 1
            if dcol_jump > jump_limit or drow_jump > jump_limit:
                print(f"[Scan] SANITY: jumped from ({self._last_known_col},{self._last_known_row}) "
                      f"to ({cur_col},{cur_row}) — refocusing")
                self._last_known_col = cur_col
                self._last_known_row = cur_row
                self._state = 'sanity_af'
                self._after_move_cb = lambda: self._detect_and_move_multi(
                    dest_col, dest_row, max_tiles, callback)
                self._start_escalating_af()
                return None

        # Virtual anchor if no exact match
        if not exact:
            max_tiles = 1
            ref_px = np.array([ref_m['center'][0], ref_m['center'][1]])
            dcol_ref = ref_m['col_id'] - cur_col
            drow_ref = ref_m['row_id'] - cur_row
            # Use observed pixel pitch for virtual anchor computation
            obs_col_px, obs_row_px = measure_pixel_pitch_from_markers(markers)
            vc_px = obs_col_px if obs_col_px is not None else cal.grid_pitch_col_px
            vr_px = obs_row_px if obs_row_px is not None else cal.grid_pitch_row_px
            virtual_px = ref_px - (dcol_ref * vc_px + drow_ref * vr_px)
            ref_m = {
                'col_id': cur_col,
                'row_id': cur_row,
                'center': (float(virtual_px[0]), float(virtual_px[1])),
                'confidence': ref_m.get('confidence', 1.0),
            }
            print(f"[Scan] Virtual anchor ({cur_col},{cur_row}) at px "
                  f"({virtual_px[0]:.0f},{virtual_px[1]:.0f})")

        self._last_known_col = cur_col
        self._last_known_row = cur_row
        self._has_known_position = True
        self._consecutive_af_failures = 0
        self._dead_restarts = 0
        self._last_good_z = float(getattr(self.MOTORS, "z_current_position_full_step", 0.0))  # ADDED

        total_dcol = dest_col - cur_col
        total_drow = dest_row - cur_row

        if total_dcol == 0 and total_drow == 0:
            self._current_target = (dest_col, dest_row)
            self._last_move_was_noop = True
            QTimer.singleShot(10, callback)
            return (cur_col, cur_row)

        dcol = max(-max_tiles, min(max_tiles, total_dcol))
        drow = max(-max_tiles, min(max_tiles, total_drow))
        target_col = cur_col + dcol
        target_row = cur_row + drow

        motor_delta, mag = self._compute_move_delta(cal, ref_m, dcol, drow, markers=markers)

        if mag < self.SKIP_THRESHOLD_FS:
            # "Already there" shortcut only trusted with a real anchor;
            # with a virtual anchor, a sub-threshold magnitude is a
            # degenerate-geometry artifact, so we still command a move.
            if exact:
                print(f"[Scan] ({cur_col},{cur_row})->({target_col},{target_row}) "
                      f"only {mag:.1f}fs, already there")
                self._current_target = (target_col, target_row)
                self._last_move_was_noop = True
                QTimer.singleShot(10, callback)
                return (cur_col, cur_row)
            obs_col_px, obs_row_px = measure_pixel_pitch_from_markers(markers)
            col_px = obs_col_px if obs_col_px is not None else cal.grid_pitch_col_px
            row_px = obs_row_px if obs_row_px is not None else cal.grid_pitch_row_px
            grid_dpx = dcol * col_px + drow * row_px
            motor_delta = cal.px_to_motor @ (-grid_dpx)
            mag = float(np.linalg.norm(motor_delta))
            print(f"[Scan] ({cur_col},{cur_row})->({target_col},{target_row}) "
                  f"virtual anchor sub-threshold — forcing grid step "
                  f"mag={mag:.1f}fs")

        # Global move cap: if active, block any XY move exceeding 2 tiles per axis
        if self._check_move_cap(motor_delta[0], motor_delta[1]):
            self._single_tile_af_retries += 1
            if self._single_tile_af_retries > self.MAX_MOVE_CAP_RETRIES:
                print(f"[Scan] MOVE CAP: {self._single_tile_af_retries-1} retries "
                      f"exceeded — treating as dead tile")
                self._single_tile_af_retries = 0
                self._skip_dead_tile()
                return None
            print(f"[Scan] MOVE CAP: AF and retry "
                  f"(attempt {self._single_tile_af_retries}/{self.MAX_MOVE_CAP_RETRIES})")
            self._state = 'sanity_af'
            self._after_move_cb = lambda: self._detect_and_move_multi(
                dest_col, dest_row, max_tiles, callback)
            self._start_escalating_af()
            return None

        # During walking (max_tiles > 1): additional relative sanity check
        expected_mag = self._expected_move_mag(cal, dcol, drow)
        if max_tiles > 1 and expected_mag > 0.5:
            if mag > expected_mag * self.MOVE_SANITY_FACTOR:
                print(f"[Scan] MOVE SANITY: mag={mag:.1f}fs but expected~{expected_mag:.1f}fs — refocusing")
                self._state = 'sanity_af'
                self._after_move_cb = lambda: self._detect_and_move_multi(
                    dest_col, dest_row, max_tiles, callback)
                self._start_escalating_af()
                return None

        self._current_target = (target_col, target_row)
        self._move_start_col = cur_col
        self._move_start_row = cur_row
        self._learn_anchor_col = ref_m['col_id']
        self._learn_anchor_row = ref_m['row_id']
        self._recovery_attempts = 0
        self._last_recovery_error = None

        cur_x = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
        cur_y = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))
        cur_z = float(getattr(self.MOTORS, "z_current_position_full_step", 0.0))
        target_x = cur_x + motor_delta[0]
        target_y = cur_y + motor_delta[1]
        # Predictive Z correction based on tilt model
        dz = self._compute_dz_for_move(dcol, drow)
        target_z = cur_z + dz

        self._tile_target_x = target_x
        self._tile_target_y = target_y
        self._tile_target_z = target_z

        print(f"[Scan] ({cur_col},{cur_row})->({target_col},{target_row}) "
              f"d=({dcol},{drow}) mag={mag:.1f}fs "
              f"X:{cur_x:.1f}->{target_x:.1f} Y:{cur_y:.1f}->{target_y:.1f} dz={dz:.3f}")

        self._state = 'moving'
        need_x = abs(target_x - cur_x) > 0.05
        need_y = abs(target_y - cur_y) > 0.05
        need_z = abs(dz) > 0.001

        if need_x or need_y:
            self._phase_sub = 'tile_xy'
            self.move_to_ctrl.start(target_x, target_y, target_z if need_z else cur_z)
        elif need_z:
            self._phase_sub = 'tile_z'
            self.move_to_ctrl.start(cur_x, cur_y, target_z)
        else:
            QTimer.singleShot(10, callback)
        return (cur_col, cur_row)

    def _detect_and_move_one_step(self, dest_col, dest_row, callback):
        return self._detect_and_move_multi(dest_col, dest_row, 1, callback)

    def _compute_move_delta(self, cal, ref_m, dcol, drow, markers=None):
        # Use observed pixel pitch if markers available
        if markers is not None:
            obs_col_px, obs_row_px = measure_pixel_pitch_from_markers(markers)
        else:
            obs_col_px, obs_row_px = None, None
        col_px = obs_col_px if obs_col_px is not None else cal.grid_pitch_col_px
        row_px = obs_row_px if obs_row_px is not None else cal.grid_pitch_row_px

        grid_dpx = dcol * col_px + drow * row_px
        cx, cy = cal.image_center_x, cal.image_center_y
        ideal_px = np.array([cx, cy]) - 0.5 * col_px - 0.5 * row_px
        ref_px = np.array([ref_m['center'][0], ref_m['center'][1]])
        offset_px = ideal_px - ref_px
        total_dpx = -grid_dpx + offset_px
        motor_delta = cal.px_to_motor @ total_dpx
        return motor_delta, float(np.linalg.norm(motor_delta))

    # ═══════════════════════════════════════════════════════════════
    # Fine correction with virtual anchor + continuous learning
    # ═══════════════════════════════════════════════════════════════

    def _fine_correct(self, callback):
        if self._stop_requested: return
        self._fine_callback = callback

        cal = self.calibration.get_result()
        frame = self.camera_preview.capture_to_array(self._camera_settings)
        if frame is None:
            QTimer.singleShot(50, lambda: self._fine_correct(callback)); return

        markers = self._detect(frame)
        if not markers:
            p = save_failed_frame(frame, "demo_fine_no_markers")
            print(f"[Scan] Fine: no markers — saved {p}")
            self._on_localize_fail(lambda: self._fine_correct(callback)); return
        expected_pos = self._current_target
        markers = self.calibration._filter_suspicious_markers(markers, expected_pos=expected_pos)
        if not markers:
            p = save_failed_frame(frame, "demo_fine_all_filtered")
            print(f"[Scan] Fine: all filtered — saved {p}")
            self._on_localize_fail(lambda: self._fine_correct(callback)); return

        cx, cy = cal.image_center_x, cal.image_center_y
        target_col, target_row = self._current_target

        # Use observed pixel pitch from current markers
        obs_col_px, obs_row_px = measure_pixel_pitch_from_markers(markers)
        col_px = obs_col_px if obs_col_px is not None else cal.grid_pitch_col_px
        row_px = obs_row_px if obs_row_px is not None else cal.grid_pitch_row_px

        def marker_priority(m):
            tile_dist = abs(m['col_id'] - target_col) + abs(m['row_id'] - target_row)
            px_dist = (m['center'][0] - cx)**2 + (m['center'][1] - cy)**2
            return (tile_dist, px_dist)

        target_m = min(markers, key=marker_priority)
        landed_col, landed_row = target_m['col_id'], target_m['row_id']

        # Virtual anchor if landed marker isn't the target
        if landed_col != target_col or landed_row != target_row:
            ref_px = np.array([target_m['center'][0], target_m['center'][1]])
            dcol_ref = landed_col - target_col
            drow_ref = landed_row - target_row
            virtual_px = ref_px - (dcol_ref * col_px + drow_ref * row_px)
            target_m = {
                'col_id': target_col,
                'row_id': target_row,
                'center': (float(virtual_px[0]), float(virtual_px[1])),
                'confidence': target_m.get('confidence', 1.0),
            }
            landed_col, landed_row = target_col, target_row

        ideal_px = np.array([cx, cy]) - 0.5 * col_px - 0.5 * row_px
        dcol_landed = target_col - landed_col
        drow_landed = target_row - landed_row
        expected_px = ideal_px - (dcol_landed * col_px + drow_landed * row_px)

        actual_px = np.array([target_m['center'][0], target_m['center'][1]])
        error_px = expected_px - actual_px
        error_fs = cal.px_to_motor @ error_px
        error_mag = float(np.linalg.norm(error_fs))

        print(f"[Scan] Fine: landed ({landed_col},{landed_row}) "
              f"target ({target_col},{target_row}) "
              f"error={error_mag:.2f}fs ({error_fs[0]:.2f},{error_fs[1]:.2f})")

        # Log first landing error for analysis
        if self._recovery_attempts == 0:
            self._log_scan(target_col, target_row, 'landing',
                          error_fs=error_fs)

        # ── Continuous learning (XY calibration) ──
        # CRITICAL: learn from the FIRST landing error (recovery_attempts == 0),
        # which reflects the systematic pitch bias. Post-correction residuals
        # are just noise and don't carry the signal we need.
        if self._recovery_attempts == 0 and not self._post_skip:
            stored_ideal_px = np.array([cx, cy]) - 0.5 * cal.grid_pitch_col_px - 0.5 * cal.grid_pitch_row_px
            stored_expected_px = stored_ideal_px - (dcol_landed * cal.grid_pitch_col_px + drow_landed * cal.grid_pitch_row_px)
            stored_miss_px = actual_px - stored_expected_px
            stored_error_mag = float(np.linalg.norm(cal.px_to_motor @ (stored_expected_px - actual_px)))

            if stored_error_mag > 0.5 and stored_error_mag < 20.0:
                try:
                    dcol_move = target_col - self._learn_anchor_col
                    drow_move = target_row - self._learn_anchor_row

                    if abs(dcol_move) > 0 and drow_move == 0:
                        raw_correction = stored_miss_px / dcol_move
                        correction = raw_correction * self.LEARN_RATE
                        max_change = np.linalg.norm(cal.grid_pitch_col_px) * 0.05
                        if np.linalg.norm(correction) > max_change:
                            correction = correction / np.linalg.norm(correction) * max_change
                        cal.grid_pitch_col_px = cal.grid_pitch_col_px + correction
                        cal.grid_pitch_col_fs = cal.px_to_motor @ cal.grid_pitch_col_px
                        print(f"[Scan] Learn: col_px += ({correction[0]:.2f},{correction[1]:.2f})")
                    elif abs(drow_move) > 0 and dcol_move == 0:
                        raw_correction = stored_miss_px / drow_move
                        correction = raw_correction * self.LEARN_RATE
                        max_change = np.linalg.norm(cal.grid_pitch_row_px) * 0.05
                        if np.linalg.norm(correction) > max_change:
                            correction = correction / np.linalg.norm(correction) * max_change
                        cal.grid_pitch_row_px = cal.grid_pitch_row_px + correction
                        cal.grid_pitch_row_fs = cal.px_to_motor @ cal.grid_pitch_row_px
                        print(f"[Scan] Learn: row_px += ({correction[0]:.2f},{correction[1]:.2f})")
                    elif abs(dcol_move) > 0 and abs(drow_move) > 0:
                        total_steps = abs(dcol_move) + abs(drow_move)
                        col_weight = abs(dcol_move) / total_steps
                        row_weight = abs(drow_move) / total_steps
                        col_correction = (stored_miss_px * col_weight / dcol_move) * self.LEARN_RATE
                        row_correction = (stored_miss_px * row_weight / drow_move) * self.LEARN_RATE
                        max_col = np.linalg.norm(cal.grid_pitch_col_px) * 0.03
                        max_row = np.linalg.norm(cal.grid_pitch_row_px) * 0.03
                        if np.linalg.norm(col_correction) > max_col:
                            col_correction = col_correction / np.linalg.norm(col_correction) * max_col
                        if np.linalg.norm(row_correction) > max_row:
                            row_correction = row_correction / np.linalg.norm(row_correction) * max_row
                        cal.grid_pitch_col_px += col_correction
                        cal.grid_pitch_row_px += row_correction
                        cal.grid_pitch_col_fs = cal.px_to_motor @ cal.grid_pitch_col_px
                        cal.grid_pitch_row_fs = cal.px_to_motor @ cal.grid_pitch_row_px
                        print(f"[Scan] Learn diag: col += ({col_correction[0]:.2f},"
                              f"{col_correction[1]:.2f}), row += ({row_correction[0]:.2f},"
                              f"{row_correction[1]:.2f})")

                    if self._tile_step_count % 10 == 0:
                        from calibration.stage_calibration import save_calibration
                        save_calibration(cal, self.config)
                except Exception as e:
                    print(f"[Scan] Learning error: {e}")

        # Error above threshold — corrective motor move
        if error_mag > 2.0:
            self._recovery_attempts += 1

            # Bounded blind-nudge loop: too many non-converging nudges ⇒
            # hand off to escalating AF, which can widen half-indefinitely.
            if self._recovery_attempts > self.MAX_RECOVERY_ATTEMPTS:
                print(f"[Scan] Fine correction not converging after "
                      f"{self.MAX_RECOVERY_ATTEMPTS} attempts "
                      f"(err {error_mag:.1f}fs) — AF and retry")
                self._recovery_attempts = 0
                self._on_localize_fail(lambda: QTimer.singleShot(
                    self.SETTLE_MS, lambda: self._fine_correct(callback)))
                return

            # Check move cap before executing correction
            if self._check_move_cap(error_fs[0], error_fs[1]):
                print(f"[Scan] Fine correction blocked by move cap — AF and retry")
                self._recovery_attempts = 0
                self._on_localize_fail(lambda: QTimer.singleShot(
                    self.SETTLE_MS, lambda: self._fine_correct(callback)))
                return

            print(f"[Scan] Corrective move: error {error_mag:.1f}fs "
                  f"({error_fs[0]:.2f},{error_fs[1]:.2f}) "
                  f"(attempt {self._recovery_attempts}/{self.MAX_RECOVERY_ATTEMPTS})")
            self._cached_markers = None  # position will change, invalidate cache
            cur_x = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
            cur_y = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))
            cur_z = float(getattr(self.MOTORS, "z_current_position_full_step", 0.0))
            self._state = 'moving'
            self._phase_sub = 'tile_xy'
            self._tile_target_x = cur_x + error_fs[0]
            self._tile_target_y = cur_y + error_fs[1]
            self._tile_target_z = cur_z
            self._after_move_cb = lambda: QTimer.singleShot(
                self.SETTLE_MS, lambda: self._fine_correct(callback))
            self.move_to_ctrl.start(self._tile_target_x, self._tile_target_y, cur_z)
            return

        self._recovery_attempts = 0
        self._last_recovery_error = None
        self._last_fine_frame = frame
        # Cache markers for next _detect_and_move_multi to skip redundant detection
        self._cached_markers = markers

        # ── Confirmed-good tile: update single anchor, commit staged Z sample ──
        self._dead_restarts = 0
        self._single_tile_af_retries = 0   # MOVE CAP cap budget renewed
        self._consecutive_skips = 0        # successful tile breaks the skip streak
        gx = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
        gy = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))
        gz = float(getattr(self.MOTORS, "z_current_position_full_step", 0.0))
        self._last_good_tile = (target_col, target_row, gx, gy, gz)

        # Commit the Z sample staged at periodic AF — only now that the tile is
        # confirmed. Drop it entirely on the first tile after a skip so the
        # post-jump discontinuity never enters the tilt fit.
        if self._pending_z_sample is not None:
            if not self._post_skip:
                s = self._pending_z_sample
                self._cumulative_tiles += s['dtiles']
                self._z_samples.append({'col': s['col'], 'row': s['row'],
                                        'tiles': self._cumulative_tiles, 'z': s['z']})
                self._fit_z_tilt()
            self._pending_z_sample = None
        self._post_skip = False

        QTimer.singleShot(10, callback)

    def _recovery_af_then_fine(self):
        if self._stop_requested: return
        self._state = 'recovery_af'
        self._start_escalating_af()

    # ═══════════════════════════════════════════════════════════════
    # Dead-region skip
    # ═══════════════════════════════════════════════════════════════

    def _on_localize_fail(self, retry_cb):
        """Single funnel for every failure that escalates AF range. Each call
        is one range-increasing restart; capped per-context:
          - spiral acquisition: MAX_LOCALIZE_RETRIES → _skip_dead_tile()
          - walk phase:         MAX_WALK_RETRIES    → give up on precise
                                target, fire walk's _walk_final_cb so the
                                next phase proceeds from current position.
        Outer guard MAX_CONSECUTIVE_SKIPS halts the run only when many
        consecutive tile skips suggest systemic detection failure."""
        if self._stop_requested:
            return
        self._pending_z_sample = None  # this tile was never confirmed
        self._consecutive_af_failures += 1
        in_spiral_step = self._scan_move_cap_active and not self._in_walk
        if in_spiral_step:
            self._dead_restarts += 1
            if self._dead_restarts >= self.MAX_LOCALIZE_RETRIES:
                print(f"[Scan] {self._dead_restarts} range-increasing restarts "
                      f"at target {self._current_target} — declaring dead, skipping")
                self._skip_dead_tile()
                return
        elif self._in_walk:
            self._walk_retries += 1
            if self._walk_retries >= self.MAX_WALK_RETRIES:
                # Don't halt — give up on the exact walk target and let the
                # next phase skip/infer tiles via its own cap machinery.
                print(f"[Scan] Walk: {self._walk_retries} retries — "
                      f"giving up on precise target ({self._walk_target_col},"
                      f"{self._walk_target_row}), proceeding from current position")
                self._walk_retries = 0
                self._in_walk = False
                cb = self._walk_final_cb
                self._walk_final_cb = None
                if cb:
                    QTimer.singleShot(10, cb)
                return
        self._state = 'sanity_af'
        self._after_move_cb = retry_cb
        self._start_escalating_af()

    def _skip_dead_tile(self):
        """Give up on the current target. Return to the last confirmed good
        tile (blind, cap-bypassing), AF there, then open-loop hop to the next
        spiral tile and re-enter the normal flow. The anchor stays pinned to
        the last good tile, so consecutive skips (a multi-tile dead region)
        each hop from the same known origin and errors don't accumulate.

        Halts the scan if MAX_CONSECUTIVE_SKIPS tiles skip in a row — at that
        point detection is systemically broken and continuing wastes the run.
        """
        if self._stop_requested:
            return
        if self._last_good_tile is None:
            self.stop("Dead tile with no good anchor to fall back to")
            return

        self._consecutive_skips += 1
        if self._consecutive_skips >= self.MAX_CONSECUTIVE_SKIPS:
            self.stop(f"{self._consecutive_skips} consecutive tile skips — "
                      f"detection appears systemically broken, halting")
            return

        self._failed_tiles.add(self._current_target)
        self._log_scan(self._current_target[0], self._current_target[1], 'skipped')
        print(f"[Scan] Skipping dead tile {self._current_target} "
              f"({self._consecutive_skips}/{self.MAX_CONSECUTIVE_SKIPS} consecutive)")

        # Reset failure counters and any staged learning.
        self._dead_restarts = 0
        self._single_tile_af_retries = 0
        self._consecutive_af_failures = 0
        self._recovery_attempts = 0
        self._pending_z_sample = None
        self._cached_markers = None

        # Out of tiles? Finish the cycle.
        if self._spiral_idx >= len(self._spiral_path):
            print(f"[Scan] No tiles left after skip — finishing scan")
            self._scan_move_cap_active = False
            self._walk_to_target_then(self._center_col, self._center_row, self._return_af)
            return

        # Consume the next candidate (mirrors _advance_spiral's read+increment).
        self._skip_candidate = self._spiral_path[self._spiral_idx]
        self._spiral_idx += 1

        gcol, grow, gx, gy, gz = self._last_good_tile
        print(f"[Scan] Returning to anchor ({gcol},{grow}) "
              f"X={gx:.1f} Y={gy:.1f} Z={gz:.3f}, then hopping to {self._skip_candidate}")
        self.progress.emit('skipping', f'Dead region — skipping to {self._skip_candidate}')

        # Blind absolute move back to the anchor. Direct move_to_ctrl.start
        # bypasses _check_move_cap entirely; the anchor is in-grid so
        # motion.limits will not clamp it.
        self._state = 'moving'
        self._phase_sub = ''
        self._tile_target_x, self._tile_target_y, self._tile_target_z = gx, gy, gz
        self._after_move_cb = self._skip_af_at_anchor
        self.move_to_ctrl.start(gx, gy, gz)

    def _skip_af_at_anchor(self):
        if self._stop_requested: return
        print(f"[Scan] Skip: AF at anchor")
        self._state = 'skip_af'
        self._after_move_cb = self._skip_hop_to_candidate
        self.af_controller.start_quick(self._camera_settings)

    def _skip_hop_to_candidate(self):
        if self._stop_requested: return
        cal = self.calibration.get_result()
        gcol, grow, gx, gy, gz = self._last_good_tile
        cand_col, cand_row = self._skip_candidate
        dcol = cand_col - gcol
        drow = cand_row - grow
        delta_fs = dcol * cal.grid_pitch_col_fs + drow * cal.grid_pitch_row_fs
        dz = self._dz_per_col * dcol + self._dz_per_row * drow  # tilt-aware; AF refines
        tx = gx + float(delta_fs[0])
        ty = gy + float(delta_fs[1])
        tz = gz + dz
        print(f"[Scan] Skip: open-loop hop ({gcol},{grow})->({cand_col},{cand_row}) "
              f"d=({dcol},{drow}) -> X={tx:.1f} Y={ty:.1f} Z={tz:.3f}")
        # Direct move — cap bypassed; in-grid target stays within motion.limits.
        self._state = 'moving'
        self._phase_sub = ''
        self._tile_target_x, self._tile_target_y, self._tile_target_z = tx, ty, tz
        self._after_move_cb = self._skip_af_at_candidate
        self.move_to_ctrl.start(tx, ty, tz)

    def _skip_af_at_candidate(self):
        if self._stop_requested: return
        print(f"[Scan] Skip: AF at candidate {self._skip_candidate}")
        self._state = 'skip_af'
        self._after_move_cb = self._skip_reenter
        self.af_controller.start_quick(self._camera_settings)

    def _skip_reenter(self):
        if self._stop_requested: return
        cand_col, cand_row = self._skip_candidate
        # Pin every continuity reference to the candidate so the jump is
        # invisible to Z prediction, the every-2-tile Z correction, pitch
        # learning, and the sanity-jump check.
        self._current_target = (cand_col, cand_row)
        self._last_known_col, self._last_known_row = cand_col, cand_row
        self._has_known_position = True
        self._learn_anchor_col, self._learn_anchor_row = cand_col, cand_row
        self._last_z_correct_pos = (cand_col, cand_row)
        self._last_af_col, self._last_af_row = cand_col, cand_row
        self._last_af_z = float(getattr(self.MOTORS, "z_current_position_full_step", 0.0))
        self._last_good_z = self._last_af_z
        self._tiles_since_af = 0
        self._post_skip = True   # suppress Z/pitch learning for this first tile
        self._pending_z_sample = None
        self._recovery_attempts = 0
        self._cached_markers = None
        self._in_walk = False
        print(f"[Scan] Skip: re-entering normal flow at {self._skip_candidate}")
        self._detect_and_move_one_step(cand_col, cand_row, self._on_spiral_arrived)

    # ═══════════════════════════════════════════════════════════════
    # Signal handlers
    # ═══════════════════════════════════════════════════════════════

    def _on_move_finished(self, success, message):
        if not self._active: return
        if self.af_controller.is_active(): return
        # Handle move-to-center before cold AF
        if self._state == 'move_to_center':
            if not success:
                self.stop(f"Move to center failed: {message}")
                return
            print(f"[Scan] At safe center, starting cold AF...")
            self.progress.emit('af', 'Finding focus...')
            self._state = 'cold_af'
            self.af_controller.start(self._camera_settings)
            return
        if self._state not in ('moving',): return
        print(f"[Scan] Move done: ok={success} sub={self._phase_sub}")

        cur_x = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
        cur_y = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))
        cur_z = float(getattr(self.MOTORS, "z_current_position_full_step", 0.0))

        if self._phase_sub == 'tile_xy':
            if abs(self._tile_target_z - cur_z) > 0.001:
                self._phase_sub = 'tile_z'
                self.move_to_ctrl.start(cur_x, cur_y, self._tile_target_z)
                return

        if self._phase_sub == 'tile_x':
            if abs(self._tile_target_y - cur_y) > 0.05:
                self._phase_sub = 'tile_y'
                self.move_to_ctrl.start(cur_x, self._tile_target_y, cur_z)
                return
            if abs(self._tile_target_z - cur_z) > 0.001:
                self._phase_sub = 'tile_z'
                self.move_to_ctrl.start(cur_x, cur_y, self._tile_target_z)
                return

        if self._phase_sub == 'tile_y':
            if abs(self._tile_target_z - cur_z) > 0.001:
                self._phase_sub = 'tile_z'
                self.move_to_ctrl.start(cur_x, cur_y, self._tile_target_z)
                return

        self._phase_sub = ''
        self._state = 'idle'
        cb = self._after_move_cb
        self._after_move_cb = None
        if cb: QTimer.singleShot(10, cb)

    def _on_af_finished(self, success, message):
        if not self._active: return

        # Walk AF
        if self._state == 'walk_af':
            self._state = 'idle'
            cb = self._after_move_cb
            self._after_move_cb = None
            if cb: QTimer.singleShot(10, cb)
            return

        # Skip-chain AF (return-to-anchor AF and post-hop AF) — generic
        # AF-then-callback; never touches Z learning or tile counters.
        if self._state == 'skip_af':
            self._state = 'idle'
            cb = self._after_move_cb
            self._after_move_cb = None
            if cb: QTimer.singleShot(10, cb)
            return

        if self._state not in ('initial_af', 'recovery_af', 'sanity_af', 'periodic_af', 'cold_af'):
            return

        cur_z = float(getattr(self.MOTORS, "z_current_position_full_step", 0.0))
        print(f"[Scan] AF done: ok={success} state={self._state} Z={cur_z:.3f}")
        col, row = self._current_target
        self._log_z(col, row, cur_z, self._state)

        score = getattr(self.af_controller, '_last_score',
                 getattr(self.af_controller, 'last_score', 99))

        # Cold start AF done — detect position and walk to center
        if self._state == 'cold_af':
            self._state = 'idle'
            self._last_af_z = cur_z
            self._last_good_z = cur_z  # ADDED
            if not success:
                self.stop("Cold start AF failed — could not find focus")
                return
            print(f"[Scan] Cold AF done at Z={cur_z:.2f}fs, detecting position...")
            self.progress.emit('walking', 'Detecting position...')
            # Detect where we are and walk to scan center
            QTimer.singleShot(self.SETTLE_MS, self._detect_and_walk_to_center)
            return

        if self._state == 'initial_af':
            self._state = 'idle'
            self._tiles_since_af = 0
            self._last_af_z = cur_z

            self._last_af_col = self._center_col
            self._last_af_row = self._center_row
            self._last_z_correct_pos = (self._last_af_col, self._last_af_row)
            cb = self._after_move_cb
            self._after_move_cb = None
            if cb:
                print(f"[Scan] Return AF done")
                QTimer.singleShot(10, cb)
            else:
                print(f"[Scan] Initial AF done, starting 5×5 scan")
                QTimer.singleShot(10, self._start_spiral)
            return

        if self._state == 'sanity_af':
            self._state = 'idle'
            self._last_z_correct_pos = self._current_target
            cb = self._after_move_cb
            self._after_move_cb = None
            if cb: QTimer.singleShot(10, cb)
            return

        if self._state == 'recovery_af':
            self._state = 'idle'
            cb = self._fine_callback
            QTimer.singleShot(self.SETTLE_MS,
                              lambda: self._fine_correct(cb))
            return

        if self._state == 'periodic_af':
            self._state = 'idle'

            col, row = self._current_target

            # Stage the Z sample — do NOT commit/fit yet. It is added to the
            # tilt model only once _fine_correct confirms this tile is real, so
            # a dead tile (AF locked on a scratch) never pollutes the fit.
            self._pending_z_sample = {
                'col': col, 'row': row, 'z': cur_z,
                'dtiles': self._tiles_since_af,
            }

            # Log prediction error for diagnostics
            predicted_dz = (self._dz_per_col * (col - self._last_af_col) +
                           self._dz_per_row * (row - self._last_af_row) +
                           self._dz_per_tile * self._tiles_since_af)
            actual_dz = cur_z - self._last_af_z
            error = actual_dz - predicted_dz
            print(f"[Scan] Z-sample staged: ({col},{row}) "
                  f"Z={cur_z:.3f} pred_err={error:.3f}")

            # Log AF to scan log
            af_score = max(self.af_controller._scores) if self.af_controller._scores else 0.0
            self._log_scan(col, row, 'periodic_af', af_z=cur_z, af_score=af_score)

            self._last_af_z = cur_z
            self._last_af_col = col
            self._last_af_row = row
            self._tiles_since_af = 0
            self._last_z_correct_pos = self._current_target
            cb = self._after_move_cb
            self._after_move_cb = None
            if cb: QTimer.singleShot(10, cb)
            return

    # ═══════════════════════════════════════════════════════════════
    # Walk to target — multi-tile jumps with AF at each stop
    # ═══════════════════════════════════════════════════════════════

    def _walk_to_target_then(self, target_col, target_row, callback):
        if self._stop_requested: return
        self._in_walk = True
        self._walk_target_col = target_col
        self._walk_target_row = target_row
        self._walk_final_cb = callback
        self._walk_no_progress = 0
        self._walk_retries = 0   # fresh cap budget per walk
        self._walk_last_infer = None
        self._walk_step()

    def _walk_step(self):
        if self._stop_requested: return
        self._detect_and_move_multi(self._walk_target_col, self._walk_target_row,
                                    self.WALK_JUMP_SIZE, self._walk_check)

    def _walk_check(self):
        if self._stop_requested: return
        frame = self.camera_preview.flush_and_capture(
            self._camera_settings, discard=self.DISCARD_FRAMES)
        if frame is None:
            QTimer.singleShot(10, self._walk_step); return
        markers = self._detect(frame)
        markers = self.calibration._filter_suspicious_markers(markers) if markers else []
        if markers:
            self._walk_retries = 0   # detection recovered; reset cap
            cal = self.calibration.get_result()
            cur_col, cur_row, _, _ = self._infer_tile_at_ideal(cal, markers)
            if cur_col == self._walk_target_col and cur_row == self._walk_target_row:
                self._last_known_col = self._walk_target_col
                self._last_known_row = self._walk_target_row
                self._last_z_correct_pos = (self._walk_target_col, self._walk_target_row)
                self._walk_no_progress = 0
                self._in_walk = False
                QTimer.singleShot(10, self._walk_final_cb)
                return
            # Stall safety net: inferred position not advancing toward target.
            if (cur_col, cur_row) == self._walk_last_infer:
                self._walk_no_progress += 1
            else:
                self._walk_no_progress = 0
                self._walk_last_infer = (cur_col, cur_row)
            if self._walk_no_progress >= self.WALK_MAX_NO_PROGRESS:
                self.stop(f"Walk stalled at ({cur_col},{cur_row}) — "
                          f"cannot reach ({self._walk_target_col},"
                          f"{self._walk_target_row})")
                return
            if self._walk_no_progress >= self.WALK_ESCALATE_NO_PROGRESS:
                print(f"[Scan] Walk no progress x{self._walk_no_progress} "
                      f"at ({cur_col},{cur_row}) — escalating AF")
                self._consecutive_af_failures += 1
                self._state = 'walk_af'
                self._after_move_cb = self._walk_step
                self._start_escalating_af()
                return
        # No markers (or none survived filtering): cap-bounded detection-fail loop.
        self._walk_retries += 1
        if self._walk_retries >= self.MAX_WALK_RETRIES:
            # Same skip-and-continue policy as the _on_localize_fail path.
            print(f"[Scan] Walk: {self._walk_retries} retries during "
                  f"_walk_check — giving up on precise target ("
                  f"{self._walk_target_col},{self._walk_target_row}), "
                  f"proceeding from current position")
            self._walk_retries = 0
            self._in_walk = False
            cb = self._walk_final_cb
            self._walk_final_cb = None
            if cb:
                QTimer.singleShot(10, cb)
            return
        # Not there yet — AF then continue
        self._state = 'walk_af'
        self._after_move_cb = self._walk_step
        self.af_controller.start_quick(self._camera_settings)

    # ═══════════════════════════════════════════════════════════════
    # 5×5 Spiral — centered scan with stitch, return to center
    # ═══════════════════════════════════════════════════════════════

    def _start_spiral(self):
        if self._stop_requested or not self._active:
            return
        # 5×5 grid centered on _center_col/_center_row
        offset_col = self._center_col - self._scan_size // 2
        offset_row = self._center_row - self._scan_size // 2
        self._spiral_path = generate_spiral_path(
            self._scan_size, offset_col, offset_row)
        self._spiral_idx = 0
        self._cycle_count += 1
        self._tiles_since_af = 0
        self._tile_step_count = 0
        self._last_z_correct_pos = (self._center_col, self._center_row)
        # Seed the anchor with the center (reached via walk + AF + fine-correct)
        cx_ = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
        cy_ = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))
        cz_ = float(getattr(self.MOTORS, "z_current_position_full_step", 0.0))
        self._last_good_tile = (self._center_col, self._center_row, cx_, cy_, cz_)
        self._dead_restarts = 0
        self._post_skip = False
        self._pending_z_sample = None
        self._scan_move_cap_active = True
        print(f"[Scan] === 5×5 scan cycle #{self._cycle_count} "
              f"center=({self._center_col},{self._center_row}) "
              f"(dz: col={self._dz_per_col:.4f} row={self._dz_per_row:.4f} "
              f"tile={self._dz_per_tile:.5f}) ===")
        self._cycle_start_time = datetime.now()
        self._advance_spiral()

    def _advance_spiral(self):
        if self._stop_requested: return
        if self._spiral_idx >= len(self._spiral_path):
            print(f"[Scan] 5×5 scan complete — {len(self._spiral_path)} tiles")
            self.progress.emit('scanning', 'Scan complete, stitching...')
            self._scan_move_cap_active = False

            # Return to center for next cycle
            self._walk_to_target_then(self._center_col, self._center_row,
                                      self._return_af)
            return
        self._in_return_phase = False
        col, row = self._spiral_path[self._spiral_idx]
        self._spiral_idx += 1
        self.progress.emit('scanning',
            f'Scan: ({col},{row}) [{self._spiral_idx}/{len(self._spiral_path)}]')
        if self._spiral_idx == 1:
            self._walk_to_target_then(col, row, self._on_spiral_arrived)
        else:
            self._detect_and_move_one_step(col, row, self._on_spiral_arrived)

    def _return_af(self):
        if self._stop_requested: return
        print(f"[Scan] Back at center, performing AF")
        self.progress.emit('af', 'Autofocus at center...')
        self._state = 'initial_af'
        self._after_move_cb = self._return_fine_correct
        self.af_controller.start_quick(self._camera_settings)

    def _return_fine_correct(self):
        """Fine-correct position after returning to center tile."""
        if self._stop_requested: return
        print(f"[Scan] Fine-correcting at center after return")
        self.progress.emit('correcting', 'Fine-correcting at center...')
        QTimer.singleShot(self.SETTLE_MS,
                          lambda: self._fine_correct(self._wait_for_next_cycle))

    def _wait_for_next_cycle(self):
        if self._stop_requested: return
        # Target-cycles cap: stop instead of scheduling next cycle.
        if self._target_cycles > 0 and self._cycle_count >= self._target_cycles:
            print(f"[Scan] Target {self._target_cycles} cycle(s) complete — stopping.")
            self.stop(f"Reached target ({self._target_cycles} cycles)")
            return
        target_interval_ms = self._scan_interval_min * 60 * 1000
        elapsed_ms = int((datetime.now() - self._cycle_start_time).total_seconds() * 1000)
        remaining_ms = max(1000, target_interval_ms - elapsed_ms)
        remaining_min = remaining_ms / 60000
        print(f"[Scan] Cycle took {elapsed_ms/1000:.0f}s, waiting {remaining_min:.1f} min for next...")
        self.progress.emit('waiting', f'Next scan in {remaining_min:.1f} minutes...')
        if self._stop_requested or not self._active:
            return
        QTimer.singleShot(remaining_ms, self._start_spiral)

    def _on_spiral_arrived(self):
        if self._stop_requested: return
        self._in_walk = False
        self._tiles_since_af += 1
        self._tile_step_count += 1
        self._single_tile_af_retries = 0  # move succeeded, reset retry counter

        # Z correction every 2 tiles
        if self._tile_step_count % 2 == 0:
            col, row = self._current_target
            prev_col, prev_row = self._last_z_correct_pos
            dcol = col - prev_col
            drow = row - prev_row
            dz = self._dz_per_col * dcol + self._dz_per_row * drow + self._dz_per_tile * 2
            if abs(dz) > 0.01:
                cur_x = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
                cur_y = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))
                cur_z = float(getattr(self.MOTORS, "z_current_position_full_step", 0.0))
                target_z = cur_z + dz
                print(f"[Scan] Z-correct: ({prev_col},{prev_row})->({col},{row}) "
                      f"dz={dz:.3f} Z:{cur_z:.3f}->{target_z:.3f}")
                self._last_z_correct_pos = (col, row)
                self._state = 'moving'
                self._phase_sub = 'tile_z'
                self._tile_target_x = cur_x
                self._tile_target_y = cur_y
                self._tile_target_z = target_z
                self._after_move_cb = self._after_z_correct
                self.move_to_ctrl.start(cur_x, cur_y, target_z)
                return
            self._last_z_correct_pos = (col, row)

        # Periodic AF
        if self._tiles_since_af >= self.AF_EVERY_N_TILES:
            print(f"[Scan] Periodic AF after {self._tiles_since_af} tiles")
            self._state = 'periodic_af'
            self._after_move_cb = lambda: QTimer.singleShot(
                self.SETTLE_MS, lambda: self._fine_correct(self._do_spiral_capture))
            self._start_escalating_af()
            return

        QTimer.singleShot(self.SETTLE_MS,
                          lambda: self._fine_correct(self._do_spiral_capture))

    def _after_z_correct(self):
        if self._stop_requested: return
        if self._tiles_since_af >= self.AF_EVERY_N_TILES:
            print(f"[Scan] Periodic AF after {self._tiles_since_af} tiles")
            self._state = 'periodic_af'
            self._after_move_cb = lambda: QTimer.singleShot(
                self.SETTLE_MS, lambda: self._fine_correct(self._do_spiral_capture))
            self._start_escalating_af()
            return
        QTimer.singleShot(self.SETTLE_MS,
                          lambda: self._fine_correct(self._do_spiral_capture))

    # ═══════════════════════════════════════════════════════════════
    # Z-tilt learning from AF samples
    # ═══════════════════════════════════════════════════════════════

    def _fit_z_tilt(self):
        """Fit z = dz_per_col*col + dz_per_row*row + dz_per_tile*tiles + z0
        using all accumulated AF samples via least-squares.

        Everything stays frozen until we have enough diverse data to fit
        all parameters reliably. This means waiting until the spiral has
        doubled back so tile count decorrelates from position.
        """
        MAX_SAMPLES = 200
        DZ_LEARN_RATE = 0.05
        MIN_SAMPLES = 40

        # Trim list to cap memory
        if len(self._z_samples) > MAX_SAMPLES:
            self._z_samples = self._z_samples[-MAX_SAMPLES:]

        samples = self._z_samples
        n = len(samples)
        if n < MIN_SAMPLES:
            return

        cols = [s['col'] for s in samples]
        rows = [s['row'] for s in samples]
        tiles = [s['tiles'] for s in samples]

        col_spread = max(cols) - min(cols)
        row_spread = max(rows) - min(rows)

        can_fit_col = col_spread >= 2
        can_fit_row = row_spread >= 2

        # Don't fit anything until we can fit tile — we want all three
        # to be determined together to avoid one absorbing another's effect.
        can_fit_tile = False
        if can_fit_col and can_fit_row:
            A_cr = np.column_stack([
                np.array(cols), np.array(rows), np.ones(n)
            ])
            t_arr = np.array(tiles)
            try:
                fit_cr, _, _, _ = np.linalg.lstsq(A_cr, t_arr, rcond=None)
                pred_tiles = A_cr @ fit_cr
                tile_resid = t_arr - pred_tiles
                r_squared = 1.0 - (np.var(tile_resid) / max(np.var(t_arr), 1e-10))
                if r_squared < 0.95:
                    can_fit_tile = True
                    print(f"[Scan] Tile fitting enabled (R²={r_squared:.3f}, {n} samples)")
                else:
                    print(f"[Scan] Tile still correlated with position (R²={r_squared:.3f})")
            except Exception:
                pass

        if not (can_fit_col and can_fit_row and can_fit_tile):
            return

        # Fit all three together
        A = np.zeros((n, 4))
        b = np.zeros(n)
        for i, s in enumerate(samples):
            A[i, 0] = s['col']
            A[i, 1] = s['row']
            A[i, 2] = s['tiles']
            A[i, 3] = 1.0
            b[i] = s['z']

        try:
            result, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

            pred = A @ result
            rms = float(np.sqrt(np.mean((b - pred)**2)))

            fit_col = float(result[0])
            fit_row = float(result[1])
            fit_tile = float(result[2])

            self._dz_per_col += DZ_LEARN_RATE * (fit_col - self._dz_per_col)
            self._dz_per_row += DZ_LEARN_RATE * (fit_row - self._dz_per_row)
            self._dz_per_tile += DZ_LEARN_RATE * (fit_tile - self._dz_per_tile)

            print(f"[Scan] Z-fit ({n} samples, rms={rms:.3f}): "
                  f"col={self._dz_per_col:.4f} row={self._dz_per_row:.4f} "
                  f"tile={self._dz_per_tile:.5f}")

            # Persist to calibration
            cal = self.calibration.get_result()
            cal.dz_per_col = self._dz_per_col
            cal.dz_per_row = self._dz_per_row
            cal.dz_per_tile = self._dz_per_tile
            if self._tile_step_count % 10 == 0:
                from calibration.stage_calibration import save_calibration
                save_calibration(cal, self.config)

        except Exception as e:
            print(f"[Scan] Z-fit error: {e}")

    # ═══════════════════════════════════════════════════════════════
    # Corner position recording & scan bounds
    # ═══════════════════════════════════════════════════════════════

    def _get_scan_corners(self):
        """Return the four corner tile coordinates of the current scan grid."""
        c0 = self._center_col - self._scan_size // 2
        r0 = self._center_row - self._scan_size // 2
        half = (self._scan_size - 1)
        return {
            (c0, r0),
            (c0 + half, r0),
            (c0, r0 + half),
            (c0 + half, r0 + half),
        }

    def _maybe_record_corner(self, col, row):
        """If this tile is a corner and hasn't been recorded yet, save its XY
        and immediately update scan bounds with what we know so far."""
        corners = self._get_scan_corners()
        if (col, row) not in corners:
            return
        if (col, row) in self._corners_set:
            return

        cur_x = float(getattr(self.MOTORS, "x_current_position_full_step", 0.0))
        cur_y = float(getattr(self.MOTORS, "y_current_position_full_step", 0.0))
        self._corner_positions[(col, row)] = (cur_x, cur_y)
        self._corners_set.add((col, row))
        print(f"[Scan] Corner ({col},{row}) recorded at X={cur_x:.1f} Y={cur_y:.1f} "
              f"[{len(self._corners_set)}/4]")

        # Update bounds with every corner — tighten as we learn more
        self._apply_scan_bounds()

    def _apply_scan_bounds(self):
        """Set motion limits from recorded corner positions + half-tile margin.
        Called incrementally — works with 1, 2, 3, or 4 corners."""
        if not self._corner_positions:
            return

        cal = self.calibration.get_result()
        # 0.8 of feature distance as margin (absorbs inter-cycle anchor jitter)
        col_mag = float(np.linalg.norm(cal.grid_pitch_col_fs)) * 0.8
        row_mag = float(np.linalg.norm(cal.grid_pitch_row_fs)) * 0.8
        margin = max(col_mag, row_mag)

        xs = [pos[0] for pos in self._corner_positions.values()]
        ys = [pos[1] for pos in self._corner_positions.values()]

        # With fewer than 4 corners, use config defaults for unknown edges
        motion_ctrl = self.move_to_ctrl.motion
        m = getattr(self.config, "motor", None)
        default_x_min = float(getattr(m, "x_full_step_min", 0.0)) if m else 0.0
        default_x_max = float(getattr(m, "x_full_step_max", 6000.0)) if m else 6000.0
        default_y_min = float(getattr(m, "y_full_step_min", 0.0)) if m else 0.0
        default_y_max = float(getattr(m, "y_full_step_max", 5200.0)) if m else 5200.0

        # Tighten each edge only if we have corner data for it
        x_min = min(xs) - margin
        x_max = max(xs) + margin
        y_min = min(ys) - margin
        y_max = max(ys) + margin

        # If we have fewer than 4 corners, keep unknown edges at defaults
        # (don't artificially constrain directions we haven't visited yet)
        if len(self._corners_set) < 4:
            x_min = min(x_min, default_x_min)
            x_max = max(x_max, default_x_max)
            y_min = min(y_min, default_y_min)
            y_max = max(y_max, default_y_max)

        old_limits = motion_ctrl.limits
        motion_ctrl.limits = type(old_limits)(
            x0=x_min, x1=x_max,
            y0=y_min, y1=y_max,
            z0=old_limits.z0, z1=old_limits.z1,
        )
        self._scan_bounds_applied = True

        print(f"[Scan] Scan bounds updated ({len(self._corners_set)}/4 corners): "
              f"X=[{x_min:.1f}, {x_max:.1f}] Y=[{y_min:.1f}, {y_max:.1f}] "
              f"(margin={margin:.1f}fs)")

    def _restore_scan_bounds(self):
        """Restore original motion limits when scan stops."""
        if not self._scan_bounds_applied:
            return
        # Reload limits from config
        motion_ctrl = self.move_to_ctrl.motion
        m = getattr(self.config, "motor", None)
        if m is not None:
            old_limits = motion_ctrl.limits
            motion_ctrl.limits = type(old_limits)(
                x0=float(getattr(m, "x_full_step_min", 0.0)),
                x1=float(getattr(m, "x_full_step_max", 6000.0)),
                y0=float(getattr(m, "y_full_step_min", 0.0)),
                y1=float(getattr(m, "y_full_step_max", 5200.0)),
                z0=old_limits.z0, z1=old_limits.z1,
            )
        self._scan_bounds_applied = False
        print(f"[Scan] Scan bounds restored to defaults")

    def _do_spiral_capture(self):
        if self._stop_requested: return
        col, row = self._current_target
        # Record corner positions after fine correction (position is accurate here)
        self._maybe_record_corner(col, row)
        self._log_scan(col, row, 'capture')
        self._capture_current_tile(col, row)
        self._advance_spiral()

    def _stitch_cycle(self):
        """Stitch all tiles from the current cycle using shared marker alignment."""
        if self._stop_requested: return
        try:
            from PIL import Image

            cycle_dir = os.path.join(self._output_base, f"cycle_{self._cycle_count:04d}")
            if not os.path.isdir(cycle_dir):
                print(f"[Scan] Stitch: no cycle dir {cycle_dir}")
                return

            # Load all tile images and detect markers
            tile_data = {}  # (col, row) -> {'path': ..., 'markers': [...], 'image': ...}
            for fn in os.listdir(cycle_dir):
                if not fn.startswith('tile_') or not fn.endswith('.jpg'):
                    continue
                parts = fn.split('_')
                col, row = int(parts[1]), int(parts[2])
                path = os.path.join(cycle_dir, fn)
                img = np.array(Image.open(path).convert('RGB'))
                markers = self._detect(img)
                tile_data[(col, row)] = {
                    'path': path,
                    'markers': markers,
                    'image': img,
                }
                print(f"[Scan] Stitch: loaded ({col},{row}) with {len(markers)} markers")

            if len(tile_data) < 2:
                print(f"[Scan] Stitch: only {len(tile_data)} tiles, skipping")
                return

            # Build marker lookup per tile: (col_id, row_id) -> pixel center
            tile_markers = {}
            for tile_pos, data in tile_data.items():
                marker_map = {}
                for m in data['markers']:
                    marker_map[(m['col_id'], m['row_id'])] = np.array([m['center'][0], m['center'][1]])
                tile_markers[tile_pos] = marker_map

            # Compute pairwise offsets from shared markers
            # offset[(a, b)] = where tile_b's origin is relative to tile_a's origin
            offsets = {}
            for pos_a in tile_data:
                for pos_b in tile_data:
                    if pos_a >= pos_b:
                        continue
                    # Check adjacency (Manhattan distance 1)
                    dc = abs(pos_a[0] - pos_b[0])
                    dr = abs(pos_a[1] - pos_b[1])
                    if dc + dr > 2:  # allow diagonal neighbors too
                        continue

                    shared = []
                    for mid, px_a in tile_markers[pos_a].items():
                        if mid in tile_markers[pos_b]:
                            px_b = tile_markers[pos_b][mid]
                            shared.append(px_a - px_b)  # offset = where b's pixel maps in a's frame

                    if shared:
                        avg_offset = np.mean(shared, axis=0)
                        offsets[(pos_a, pos_b)] = avg_offset
                        print(f"[Scan] Stitch: ({pos_a})->({pos_b}) "
                              f"offset=({avg_offset[0]:.1f},{avg_offset[1]:.1f}) "
                              f"from {len(shared)} shared markers")

            if not offsets:
                print(f"[Scan] Stitch: no shared markers between any tiles")
                return

            # Build global positions via BFS from the first tile
            start_tile = (self._center_col, self._center_row)
            if start_tile not in tile_data:
                start_tile = next(iter(tile_data))

            global_pos = {start_tile: np.array([0.0, 0.0])}
            queue = [start_tile]
            visited = {start_tile}

            while queue:
                current = queue.pop(0)
                for (a, b), offset in offsets.items():
                    if a == current and b not in visited:
                        # b's origin = a's origin + offset
                        global_pos[b] = global_pos[a] + offset
                        visited.add(b)
                        queue.append(b)
                    elif b == current and a not in visited:
                        # a's origin = b's origin - offset
                        global_pos[a] = global_pos[current] - offset
                        visited.add(a)
                        queue.append(a)

            print(f"[Scan] Stitch: placed {len(global_pos)}/{len(tile_data)} tiles")

            # Compute canvas size
            h, w = next(iter(tile_data.values()))['image'].shape[:2]
            min_x = min(p[0] for p in global_pos.values())
            min_y = min(p[1] for p in global_pos.values())
            max_x = max(p[0] for p in global_pos.values()) + w
            max_y = max(p[1] for p in global_pos.values()) + h

            # Shift so everything is positive
            for k in global_pos:
                global_pos[k][0] -= min_x
                global_pos[k][1] -= min_y

            canvas_w = int(max_x - min_x + 1)
            canvas_h = int(max_y - min_y + 1)
            print(f"[Scan] Stitch: canvas {canvas_w}x{canvas_h}")

            # Place tiles with alpha blending in overlap regions
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
            weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)

            for tile_pos in sorted(global_pos.keys()):
                if tile_pos not in tile_data:
                    continue
                ox = int(round(global_pos[tile_pos][0]))
                oy = int(round(global_pos[tile_pos][1]))
                img = tile_data[tile_pos]['image'].astype(np.float32)
                th, tw = img.shape[:2]


                # Create weight mask — 1.0 in center, smooth cosine fade at edges
                # Use a wide margin and raised-cosine (Hann) profile for seamless blending
                margin = min(1200, tw // 3, th // 3)  # blend zone — up to 1/3 of tile
                wx = np.ones(tw, dtype=np.float32)
                wy = np.ones(th, dtype=np.float32)
                ramp_x = min(margin, tw // 2)
                ramp_y = min(margin, th // 2)
                # Raised cosine: 0.5*(1 - cos(pi*t)) smoothly goes 0→1
                if ramp_x > 0:
                    t = np.linspace(0, 1, ramp_x)
                    cosine_ramp = 0.5 * (1.0 - np.cos(np.pi * t))
                    wx[:ramp_x] = cosine_ramp
                    wx[-ramp_x:] = cosine_ramp[::-1]
                if ramp_y > 0:
                    t = np.linspace(0, 1, ramp_y)
                    cosine_ramp = 0.5 * (1.0 - np.cos(np.pi * t))
                    wy[:ramp_y] = cosine_ramp
                    wy[-ramp_y:] = cosine_ramp[::-1]
                weight = wy[:, None] * wx[None, :]  # 2D weight mask

                # Clip to canvas bounds
                src_x0 = max(0, -ox)
                src_y0 = max(0, -oy)
                src_x1 = min(tw, canvas_w - ox)
                src_y1 = min(th, canvas_h - oy)
                dst_x0 = max(0, ox)
                dst_y0 = max(0, oy)
                dst_x1 = dst_x0 + (src_x1 - src_x0)
                dst_y1 = dst_y0 + (src_y1 - src_y0)

                w_region = weight[src_y0:src_y1, src_x0:src_x1]
                canvas[dst_y0:dst_y1, dst_x0:dst_x1] += img[src_y0:src_y1, src_x0:src_x1] * w_region[:, :, None]
                weight_sum[dst_y0:dst_y1, dst_x0:dst_x1] += w_region

            # Normalize by total weight
            mask = weight_sum > 0
            for c in range(3):
                canvas[:, :, c][mask] /= weight_sum[mask]

            canvas = np.clip(canvas, 0, 255).astype(np.uint8)

            # Save
            out_path = os.path.join(self._output_base,
                                    f"stitched_cycle_{self._cycle_count:04d}.jpg")
            Image.fromarray(canvas).save(out_path, quality=95)
             # Save reduced preview
            preview = Image.fromarray(canvas)
            preview = preview.resize((canvas_w // 4, canvas_h // 4), Image.LANCZOS)
            preview_path = os.path.join(self._output_base,
                                        f"stitched_cycle_{self._cycle_count:04d}_preview.jpg")
            preview.save(preview_path, quality=80)
            print(f"[Scan] Stitched image saved: {out_path} ({canvas_w}x{canvas_h})")
            self.progress.emit('stitching', f'Stitched cycle {self._cycle_count}')
            print(f"[Scan] Stitched image saved: {out_path} ({canvas_w}x{canvas_h})")
            self.progress.emit('stitching', f'Stitched cycle {self._cycle_count}')

        except Exception as e:
            print(f"[Scan] Stitch error: {e}")
            import traceback
            traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════
    # Capture
    # ═══════════════════════════════════════════════════════════════
    def _capture_current_tile(self, col, row):
        try:
            frame = getattr(self, '_last_fine_frame', None)
            if frame is None:
                frame = self.camera_preview.capture_to_array(self._camera_settings)
            if frame is None: return
            self._last_fine_frame = None

            d = os.path.join(self._output_base, f"cycle_{self._cycle_count:04d}")
            os.makedirs(d, exist_ok=True)
            fn = f"tile_{col:02d}_{row:02d}_{datetime.now().strftime('%H%M%S_%f')[:-3]}.jpg"
            p = os.path.join(d, fn)
            from PIL import Image
            Image.fromarray(frame).save(p, quality=self._camera_settings.get('quality', 95))
            print(f"[Scan] Captured ({col},{row}) -> {fn}")
            self.capture_ready.emit(col, row, p)
        except Exception as e:
            print(f"[Scan] Capture error: {e}")