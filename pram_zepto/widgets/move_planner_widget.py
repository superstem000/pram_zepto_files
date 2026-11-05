import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QComboBox, QLineEdit,
    QHBoxLayout, QVBoxLayout, QFormLayout, QGridLayout,
    QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMessageBox, QInputDialog,
    QSizePolicy
)

from widgets.widget_collapsible_box import CollapsibleBox
from calibration.orient_calibration import SAFE_CENTER_X_FS, SAFE_CENTER_Y_FS


# ═══════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════

@dataclass
class RecipeStep:
    """One step in a recipe. Type determines how it's executed.

    Types:
        'move'                — go to absolute XYZ (microsteps) at given speed
        'microfluidics'       — legacy full hardcoded sequence (push1+push2)
        'microfluidics_push1' — first push (approach → slow push → retract)
        'microfluidics_push2' — second push + final Z retract
        'calibration'         — run full orient calibration to completion;
                                doubles as the incubation window between
                                microfluidics pushes.
        'spiral_scan'         — run a spiral scan, then advance.
    """
    step_type: str = "move"
    x_us: int = 0  # X in microsteps (mres=32)
    y_us: int = 0
    z_us: int = 0
    speed: float = 1.0  # speed multiplier
    dwell_ms: int = 0
    override_z_limit: bool = False
    run_calibration_during_dwell: bool = False
    num_cycles: int = 1  # spiral_scan: 0 = until stopped
    scan_size: int = 5   # spiral_scan: odd N for an N×N scan
    note: str = ""


@dataclass
class MoveRecipe:
    name: str
    steps: List[RecipeStep] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Microfluidics hardcoded sequence (all positions in microsteps)
# ═══════════════════════════════════════════════════════════════

MICROFLUIDICS_PUSH1 = [
    # Move to push position 1
    RecipeStep(step_type="move", x_us=35867, y_us=19270, z_us=22589,
               speed=2.67, override_z_limit=True),
    # Fast approach
    RecipeStep(step_type="move", x_us=21000, y_us=19270, z_us=22589,
               speed=0.02, override_z_limit=True),
    # Slow push
    RecipeStep(step_type="move", x_us=10677, y_us=19270, z_us=22589,
               speed=0.02, override_z_limit=True),
    # Fast retract to safe park position
    RecipeStep(step_type="move", x_us=35867, y_us=19270, z_us=22589,
               speed=2.67, override_z_limit=True),
]

MICROFLUIDICS_PUSH2 = [
    # Move to push position 2
    RecipeStep(step_type="move", x_us=35867, y_us=36986, z_us=22589,
               speed=2.67, override_z_limit=True),
    # Fast approach
    RecipeStep(step_type="move", x_us=21000, y_us=36986, z_us=22589,
               speed=0.02, override_z_limit=True),
    # Slow push
    RecipeStep(step_type="move", x_us=10677, y_us=36986, z_us=22589,
               speed=0.02, override_z_limit=True),
    # Fast retract to safe park position
    RecipeStep(step_type="move", x_us=35867, y_us=36986, z_us=22589,
               speed=2.67, override_z_limit=True),
    # Final Z retract
    RecipeStep(step_type="move", x_us=35867, y_us=36986, z_us=0,
               speed=2.67, override_z_limit=True),
]

# Legacy: step_type="microfluidics" expands to push1 + push2 with no gap.
MICROFLUIDICS_SEQUENCE = MICROFLUIDICS_PUSH1 + MICROFLUIDICS_PUSH2


# ═══════════════════════════════════════════════════════════════
# Widget
# ═══════════════════════════════════════════════════════════════

MRES = 32  # locked microstepping resolution

class MovePlannerWidget(QWidget):
    """
    Sequence builder for automated stage control.

    Supports three step types:
      - Move: go to absolute XYZ (microsteps) at a given speed
      - Microfluidics: hardcoded sub-sequence of fine movements
      - Spiral Scan: runs one full spiral-scan cycle, then advances
    """

    def __init__(
        self,
        parent_layout,
        config,
        stepper_widget,
        recipes_path: Optional[str] = None,
        mode: str = 'full',
        parent=None,
    ):
        super().__init__(parent)

        self.parent_layout = parent_layout
        self.config = config
        self.stepper = stepper_widget
        self.mode = mode

        if recipes_path is None:
            cfg_dir = os.path.dirname(getattr(self.config, "cfg_path", "") or "")
            if cfg_dir and os.path.isdir(cfg_dir):
                recipes_path = os.path.join(cfg_dir, "move_recipes.json")
            else:
                recipes_path = os.path.join(os.getcwd(), "move_recipes.json")
        self.recipes_path = recipes_path

        self._recipes: Dict[str, MoveRecipe] = {}
        self._active_recipe_name: Optional[str] = None

        # Execution state
        self._exec_active = False
        self._exec_index = -1
        self._pending_dwell_ms = 0
        self._pending_cal_during_dwell = False  # set per-step from RecipeStep
        self._micro_seq_index = -1  # sub-index within microfluidics sequence
        self._micro_seq_source = []  # which sequence we're executing (PUSH1, PUSH2, or full)
        self._saved_speed_mult = 1.0
        self._saved_z_limit = None  # saved z_global_max for restore

        self._dwell_timer = QTimer(self)
        self._dwell_timer.setSingleShot(True)
        self._dwell_timer.timeout.connect(self._on_dwell_timeout)

        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._update_countdown)
        self._dwell_remaining_ms = 0

        # Calibration-during-dwell hooks. The host screen calls
        # set_orient_cal_controller() and set_camera_settings_provider() to
        # wire these up. When unset, run_calibration_during_dwell is a no-op.
        self._orient_cal = None
        self._camera_settings_provider = None
        self._cal_running_for_step = False  # we kicked off cal for this dwell
        self._waiting_for_cal = False       # dwell expired, holding for cal
        self._cal_prep_moving = False       # planner-driven move to SAFE_CENTER before cal
        self._cal_skip_move_to_center = False  # passed to orient_cal.start()

        # Spiral-scan-step hooks. The host screen calls
        # set_spiral_scan_controller() and set_spiral_scan_settings_provider()
        # to wire these up. When unset, spiral_scan steps are skipped.
        self._spiral_scan = None
        self._spiral_scan_settings_provider = None
        self._spiral_scan_running_for_step = False  # we kicked off scan for this step

        self._build_move_planner_ui()

        if self.mode == 'full':
            try:
                self.stepper.move_to_ctrl.finished.connect(self._on_move_to_finished)
            except Exception:
                pass

        self._load_recipes_from_disk()
        self._refresh_recipe_combo()

    # =========================
    # Public hooks (set by host screen)
    # =========================

    def set_orient_cal_controller(self, orient_cal):
        """Wire the orientation calibration controller. When a recipe step
        has run_calibration_during_dwell=True, the planner calls
        orient_cal.start(settings) at the start of that dwell and waits for
        it to finish (or for the dwell to expire, whichever is later)."""
        self._orient_cal = orient_cal

    def set_camera_settings_provider(self, provider):
        """Set a no-arg callable returning the current camera settings dict.
        Required for run_calibration_during_dwell to actually start cal AND
        for spiral_scan steps."""
        self._camera_settings_provider = provider

    def set_spiral_scan_controller(self, spiral_scan):
        """Wire the spiral-scan controller. When a recipe step has
        step_type=='spiral_scan', the planner calls spiral_scan.configure(...)
        and spiral_scan.start(settings), then advances when one cycle
        completes (detected by progress emitting phase='waiting')."""
        self._spiral_scan = spiral_scan

    def set_spiral_scan_settings_provider(self, provider):
        """Set a no-arg callable returning (interval_min, center_col, center_row)
        for the spiral scan. Typically reads the stepper widget's scan
        settings UI so the planner mirrors the Spiral Scan button."""
        self._spiral_scan_settings_provider = provider

    def recipe_needs_calibration(self) -> bool:
        """True if the currently-loaded recipe contains any step that requests
        calibration during its dwell. Used by the host screen to gate Run
        Recipe behind 'live preview must be on'."""
        try:
            steps = self._read_table_steps()
        except Exception:
            return False
        for s in steps:
            if getattr(s, "run_calibration_during_dwell", False):
                return True
            if s.step_type == "calibration":
                return True
            # Legacy microfluidics step expands into MICROFLUIDICS_SEQUENCE
            if s.step_type == "microfluidics":
                for sub in MICROFLUIDICS_SEQUENCE:
                    if getattr(sub, "run_calibration_during_dwell", False):
                        return True
        return False

    def recipe_needs_spiral_scan(self) -> bool:
        """True if the recipe contains any spiral_scan step. Host screen
        uses this to gate Run Recipe behind 'live preview must be on' and
        'calibration must exist'."""
        try:
            steps = self._read_table_steps()
        except Exception:
            return False
        return any(s.step_type == "spiral_scan" for s in steps)

    # =========================
    # UI
    # =========================

    def _build_move_planner_ui(self):
        box = CollapsibleBox("Sequence Planner")

        panel = QFrame()
        panel_layout = QHBoxLayout(panel)
        panel_layout.setContentsMargins(25, 25, 25, 35)
        panel_layout.setSpacing(35)

        # ==========================
        # LEFT COLUMN
        # ==========================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)

        # 1. Recipe Selection
        recipe_row = QHBoxLayout()
        recipe_lbl = QLabel("Recipe:")
        recipe_lbl.setStyleSheet("font-weight: bold;")
        self.recipe_combo = QComboBox()
        self.recipe_combo.setMinimumHeight(45)
        self.recipe_combo.currentTextChanged.connect(self._on_recipe_selected)
        recipe_row.addWidget(recipe_lbl)
        recipe_row.addWidget(self.recipe_combo, 1)
        left_layout.addLayout(recipe_row)

        # 2. Recipe Name Edit
        self.recipe_name_edit = QLineEdit()
        self.recipe_name_edit.setPlaceholderText("New recipe name (optional)")
        self.recipe_name_edit.setMinimumHeight(45)
        left_layout.addWidget(self.recipe_name_edit)

        # 3. Step type selector
        type_layout = QVBoxLayout()
        type_layout.setSpacing(8)

        type_lbl = QLabel("Add:")
        type_lbl.setStyleSheet("font-weight: bold;")
        type_layout.addWidget(type_lbl)

        self.btn_add_move = QPushButton("Move Step")
        self.btn_add_move.setMinimumHeight(150)
        self.btn_add_move.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_add_move.clicked.connect(self._on_add_move_step)
        type_layout.addWidget(self.btn_add_move)

        # Push1 / Calibration / Push2 — three buttons replacing the single microfluidics step.
        self.btn_add_push1 = QPushButton("µFluidics Push 1")
        self.btn_add_push1.setMinimumHeight(80)
        self.btn_add_push1.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_add_push1.clicked.connect(self._on_add_microfluidics_push1)
        type_layout.addWidget(self.btn_add_push1)

        self.btn_add_calibration = QPushButton("Calibration")
        self.btn_add_calibration.setMinimumHeight(80)
        self.btn_add_calibration.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        self.btn_add_calibration.clicked.connect(self._on_add_calibration)
        type_layout.addWidget(self.btn_add_calibration)

        self.btn_add_push2 = QPushButton("µFluidics Push 2")
        self.btn_add_push2.setMinimumHeight(80)
        self.btn_add_push2.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_add_push2.clicked.connect(self._on_add_microfluidics_push2)
        type_layout.addWidget(self.btn_add_push2)

        # Kept for backwards-compat: btn_add_microfluidics reference points to
        # the push1 button so any external code that still references it (rare)
        # doesn't crash.
        self.btn_add_microfluidics = self.btn_add_push1

        # Spiral scan add: button + size/cycles inputs that the step will use.
        # Inline so the user doesn't have to go change the manual-mode scan
        # settings on the stepper widget every time.
        spiral_block = QVBoxLayout()
        spiral_block.setSpacing(4)
        self.btn_add_spiral = QPushButton("Spiral Scan")
        self.btn_add_spiral.setMinimumHeight(110)
        self.btn_add_spiral.setStyleSheet("background-color: #1565C0; color: white; font-weight: bold;")
        self.btn_add_spiral.clicked.connect(self._on_add_spiral_scan)
        spiral_block.addWidget(self.btn_add_spiral)

        # Container height must accommodate the global stylesheet's
        # min-height: 45px + padding + border on QSpinBox/QComboBox (~55px
        # painted). Clamping smaller clips the value text to the bottom.
        spiral_inputs = QWidget()
        spiral_inputs.setFixedHeight(60)
        sh = QHBoxLayout(spiral_inputs)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.setSpacing(8)
        sh.addWidget(QLabel("Size:"))
        self.spiral_step_size_combo = QComboBox()
        self.spiral_step_size_combo.addItems(["3", "5", "7", "9", "11", "15", "21", "31"])
        self.spiral_step_size_combo.setCurrentText("5")
        self.spiral_step_size_combo.setFixedWidth(85)
        self.spiral_step_size_combo.setFixedHeight(55)
        sh.addWidget(self.spiral_step_size_combo)
        sh.addWidget(QLabel("Cycles:"))
        self.spiral_step_cycles_spin = QSpinBox()
        self.spiral_step_cycles_spin.setRange(0, 999)  # 0 = ∞ (runs until manually stopped)
        self.spiral_step_cycles_spin.setValue(1)
        self.spiral_step_cycles_spin.setFixedWidth(85)
        self.spiral_step_cycles_spin.setFixedHeight(55)
        self.spiral_step_cycles_spin.setSpecialValueText("∞")
        sh.addWidget(self.spiral_step_cycles_spin)
        sh.addStretch()
        spiral_block.addWidget(spiral_inputs)

        type_layout.addLayout(spiral_block)

        left_layout.addLayout(type_layout)

        # 4. Move input fields
        left_layout.addSpacing(10)

        self.x_us = QSpinBox()
        self.y_us = QSpinBox()
        self.z_us = QSpinBox()
        self.speed_spin = QDoubleSpinBox()
        self.dwell_ms = QSpinBox()

        for sb in (self.x_us, self.y_us, self.z_us):
            sb.setRange(0, 200000)
            sb.setSingleStep(32)
            sb.setMinimumHeight(45)

        self.speed_spin.setDecimals(2)
        self.speed_spin.setRange(0.01, 2.67)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setPrefix("x")
        self.speed_spin.setMinimumHeight(45)

        self.dwell_ms.setRange(0, 60_000_000)
        self.dwell_ms.setSingleStep(100)
        self.dwell_ms.setSuffix(" ms")
        self.dwell_ms.setMinimumHeight(45)

        try:
            self.x_us.setRange(0, int(float(getattr(self.stepper.MOTORS, "x_full_step_max", 6000)) * MRES))
            self.y_us.setRange(0, int(float(getattr(self.stepper.MOTORS, "y_full_step_max", 5200)) * MRES))
            self.z_us.setRange(0, int(float(getattr(self.stepper.MOTORS, "z_full_step_max", 1200)) * MRES))
        except Exception:
            pass

        try:
            self.speed_spin.setRange(
                self.stepper.mc.get_min_multiplier(),
                self.stepper.mc.get_max_multiplier()
            )
        except Exception:
            pass

        for label_text, widget in [
            ("X (µsteps):", self.x_us),
            ("Y (µsteps):", self.y_us),
            ("Z (µsteps):", self.z_us),
            ("Speed:", self.speed_spin),
            ("Dwell:", self.dwell_ms),
        ]:
            row = QHBoxLayout()
            row.setSpacing(15)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(120)
            row.addWidget(lbl)
            row.addWidget(widget, 1)
            left_layout.addLayout(row)

        self.btn_use_current = QPushButton("Use Current Position")
        self.btn_use_current.setMinimumHeight(45)
        self.btn_use_current.clicked.connect(self._on_use_current)
        left_layout.addWidget(self.btn_use_current)

        left_layout.addSpacing(10)

        # 5. Table
        # Columns: #, Type, X, Y, Z, Speed, Dwell
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["#", "Type", "X", "Y", "Z", "Speed", "Dwell"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(250)
        self.table.verticalHeader().setDefaultSectionSize(40)

        # Column widths
        self.table.setColumnWidth(0, 35)   # #
        self.table.setColumnWidth(1, 110)  # Type
        self.table.setColumnWidth(2, 70)   # X
        self.table.setColumnWidth(3, 70)   # Y
        self.table.setColumnWidth(4, 70)   # Z
        self.table.setColumnWidth(5, 60)   # Speed
        self.table.setColumnWidth(6, 70)   # Dwell

        left_layout.addWidget(self.table)

        # Status
        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setStyleSheet("color: gray; margin-top: 5px;")
        left_layout.addWidget(self.status_lbl)

        # ==========================
        # RIGHT COLUMN
        # ==========================
        right_widget = QWidget()
        right_widget.setFixedWidth(220)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self.btn_new = QPushButton("New")
        self.btn_new.clicked.connect(self._on_new_recipe)
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._on_save_recipe)
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._on_delete_recipe)

        right_layout.addWidget(self.btn_new)
        right_layout.addWidget(self.btn_save)
        right_layout.addWidget(self.btn_delete)
        right_layout.addSpacing(30)

        # Execute (full mode only)
        self.btn_execute = None
        self.btn_stop = None
        if self.mode == 'full':
            lbl_exec = QLabel("Execute")
            lbl_exec.setStyleSheet("font-weight: bold;")
            right_layout.addWidget(lbl_exec)

            self.btn_execute = QPushButton("Run Recipe")
            self.btn_execute.setMinimumHeight(55)
            self.btn_execute.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
            self.btn_execute.clicked.connect(self._on_execute)
            right_layout.addWidget(self.btn_execute)

            self.btn_stop = QPushButton("Stop")
            self.btn_stop.setMinimumHeight(55)
            self.btn_stop.setEnabled(False)
            self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
            self.btn_stop.clicked.connect(self._on_stop)
            right_layout.addWidget(self.btn_stop)
            right_layout.addSpacing(25)

        # Step controls
        lbl_steps = QLabel("Steps")
        lbl_steps.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(lbl_steps)

        self.btn_remove_step = QPushButton("Remove")
        self.btn_remove_step.clicked.connect(self._on_remove_step)
        self.btn_move_up = QPushButton("Move Up")
        self.btn_move_up.clicked.connect(lambda: self._on_reorder(-1))
        self.btn_move_down = QPushButton("Move Down")
        self.btn_move_down.clicked.connect(lambda: self._on_reorder(+1))
        self.btn_clear_steps = QPushButton("Clear All")
        self.btn_clear_steps.clicked.connect(self._on_clear_steps)

        right_layout.addWidget(self.btn_remove_step)
        right_layout.addWidget(self.btn_move_up)
        right_layout.addWidget(self.btn_move_down)
        right_layout.addWidget(self.btn_clear_steps)
        right_layout.addStretch()

        panel_layout.addWidget(left_widget, 1)
        panel_layout.addWidget(right_widget, 0)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(panel)
        box.setContentLayout(content_layout)

        self.parent_layout.addWidget(box)
        self.move_planner_group = box

    # =========================
    # Add steps
    # =========================

    def _cur_x_fs(self) -> float:
        return float(getattr(self.stepper.MOTORS, "x_current_position_full_step", 0.0))

    def _cur_y_fs(self) -> float:
        return float(getattr(self.stepper.MOTORS, "y_current_position_full_step", 0.0))

    def _cur_z_fs(self) -> float:
        return float(getattr(self.stepper.MOTORS, "z_current_position_full_step", 0.0))

    def _on_use_current(self):
        try:
            x_fs = float(getattr(self.stepper.MOTORS, "x_current_position_full_step", 0.0))
            y_fs = float(getattr(self.stepper.MOTORS, "y_current_position_full_step", 0.0))
            z_fs = float(getattr(self.stepper.MOTORS, "z_current_position_full_step", 0.0))
            self.x_us.setValue(int(round(x_fs * MRES)))
            self.y_us.setValue(int(round(y_fs * MRES)))
            self.z_us.setValue(int(round(z_fs * MRES)))
        except Exception:
            pass
        self._set_status("Loaded current position.")

    def _on_add_move_step(self):
        step = RecipeStep(
            step_type="move",
            x_us=self.x_us.value(),
            y_us=self.y_us.value(),
            z_us=self.z_us.value(),
            speed=self.speed_spin.value(),
            dwell_ms=self.dwell_ms.value(),
        )
        self._add_step_to_table(step)
        self._set_status("Added move step.")

    def _on_add_microfluidics(self):
        """Legacy single-step microfluidics — kept for any external callers.
        New recipes should use push1 + calibration + push2 as three steps."""
        step = RecipeStep(
            step_type="microfluidics",
            dwell_ms=self.dwell_ms.value(),
            note="Microfluidics routine (legacy: push1 + push2 no gap)",
        )
        self._add_step_to_table(step)
        self._set_status("Added microfluidics (legacy) step.")

    def _on_add_microfluidics_push1(self):
        step = RecipeStep(
            step_type="microfluidics_push1",
            dwell_ms=self.dwell_ms.value(),
            note="µFluidics Push 1 (approach → slow push → retract)",
        )
        self._add_step_to_table(step)
        self._set_status("Added µFluidics Push 1 step.")

    def _on_add_microfluidics_push2(self):
        step = RecipeStep(
            step_type="microfluidics_push2",
            dwell_ms=self.dwell_ms.value(),
            note="µFluidics Push 2 (approach → slow push → retract → Z retract)",
        )
        self._add_step_to_table(step)
        self._set_status("Added µFluidics Push 2 step.")

    def _on_add_calibration(self):
        step = RecipeStep(
            step_type="calibration",
            dwell_ms=self.dwell_ms.value(),
            note="Full orient calibration (~30 min; doubles as incubation)",
        )
        self._add_step_to_table(step)
        self._set_status("Added Calibration step.")

    def _on_add_spiral_scan(self):
        # Each spiral_scan step carries its own size + cycles.
        # cycles = 0 → runs until manually stopped (recipe will not advance).
        try:
            size = int(self.spiral_step_size_combo.currentText())
        except Exception:
            size = 5
        try:
            cyc = int(self.spiral_step_cycles_spin.value())
        except Exception:
            cyc = 1
        if cyc < 0: cyc = 0
        cyc_text = "∞" if cyc == 0 else f"{cyc} cycle{'s' if cyc != 1 else ''}"
        step = RecipeStep(
            step_type="spiral_scan",
            dwell_ms=self.dwell_ms.value(),
            num_cycles=cyc,
            scan_size=size,
            note=f"Spiral scan ({size}×{size}, {cyc_text})",
        )
        self._add_step_to_table(step)
        self._set_status(f"Added spiral scan step ({size}×{size}, {cyc_text}).")

    def _add_step_to_table(self, step: RecipeStep):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self._set_table_row(r, step)
        self.table.selectRow(r)

    def _set_table_row(self, r: int, step: RecipeStep):
        self.table.setItem(r, 0, QTableWidgetItem(str(r + 1)))

        if step.step_type == "move":
            type_item = QTableWidgetItem("Move")
            self.table.setItem(r, 2, QTableWidgetItem(str(step.x_us)))
            self.table.setItem(r, 3, QTableWidgetItem(str(step.y_us)))
            self.table.setItem(r, 4, QTableWidgetItem(str(step.z_us)))
            self.table.setItem(r, 5, QTableWidgetItem(f"x{step.speed:.2f}"))
            self.table.setItem(r, 6, QTableWidgetItem(str(step.dwell_ms) if step.dwell_ms else ""))
        elif step.step_type == "microfluidics":
            type_item = QTableWidgetItem("µFluidics")
            type_item.setBackground(Qt.GlobalColor.darkYellow)
            for c in (2, 3, 4, 5):
                self.table.setItem(r, c, QTableWidgetItem("—"))
            self.table.setItem(r, 6, QTableWidgetItem(str(step.dwell_ms) if step.dwell_ms else ""))
        elif step.step_type == "microfluidics_push1":
            type_item = QTableWidgetItem("µFluidics 1")
            type_item.setBackground(Qt.GlobalColor.darkYellow)
            for c in (2, 3, 4, 5):
                self.table.setItem(r, c, QTableWidgetItem("—"))
            self.table.setItem(r, 6, QTableWidgetItem(str(step.dwell_ms) if step.dwell_ms else ""))
        elif step.step_type == "microfluidics_push2":
            type_item = QTableWidgetItem("µFluidics 2")
            type_item.setBackground(Qt.GlobalColor.darkYellow)
            for c in (2, 3, 4, 5):
                self.table.setItem(r, c, QTableWidgetItem("—"))
            self.table.setItem(r, 6, QTableWidgetItem(str(step.dwell_ms) if step.dwell_ms else ""))
        elif step.step_type == "calibration":
            type_item = QTableWidgetItem("Calibrate")
            type_item.setBackground(Qt.GlobalColor.darkMagenta)
            for c in (2, 3, 4, 5):
                self.table.setItem(r, c, QTableWidgetItem("—"))
            self.table.setItem(r, 6, QTableWidgetItem(str(step.dwell_ms) if step.dwell_ms else ""))
        elif step.step_type == "spiral_scan":
            type_item = QTableWidgetItem("Scan")
            type_item.setBackground(Qt.GlobalColor.darkBlue)
            # Use X column for grid size, Y column for cycle count; Z/Speed stay "—"
            self.table.setItem(r, 2, QTableWidgetItem(f"{step.scan_size}×{step.scan_size}"))
            cyc_text = f"{step.num_cycles} cyc" if step.num_cycles > 0 else "∞"
            self.table.setItem(r, 3, QTableWidgetItem(cyc_text))
            self.table.setItem(r, 4, QTableWidgetItem("—"))
            self.table.setItem(r, 5, QTableWidgetItem("—"))
            self.table.setItem(r, 6, QTableWidgetItem(str(step.dwell_ms) if step.dwell_ms else ""))

        self.table.setItem(r, 1, type_item)

        # Store the full step data as user data on the type item for retrieval
        type_item.setData(Qt.ItemDataRole.UserRole, step.__dict__)

    def _read_table_steps(self) -> List[RecipeStep]:
        steps: List[RecipeStep] = []
        for r in range(self.table.rowCount()):
            type_item = self.table.item(r, 1)
            if type_item is None:
                continue
            data = type_item.data(Qt.ItemDataRole.UserRole)
            if data:
                steps.append(RecipeStep(**data))
            else:
                # Legacy fallback: try reading as move step
                try:
                    x = int(self.table.item(r, 2).text())
                    y = int(self.table.item(r, 3).text())
                    z = int(self.table.item(r, 4).text())
                    speed_text = self.table.item(r, 5).text().replace("x", "")
                    speed = float(speed_text) if speed_text else 1.0
                    dwell_text = self.table.item(r, 6).text()
                    dwell = int(dwell_text) if dwell_text else 0
                    steps.append(RecipeStep(step_type="move", x_us=x, y_us=y,
                                            z_us=z, speed=speed, dwell_ms=dwell))
                except Exception:
                    pass
        return steps

    # =========================
    # Step controls
    # =========================

    def _on_remove_step(self):
        row = self._selected_row()
        if row < 0:
            return
        self.table.removeRow(row)
        self._renumber()

    def _on_reorder(self, delta: int):
        row = self._selected_row()
        if row < 0:
            return
        new_row = row + int(delta)
        if new_row < 0 or new_row >= self.table.rowCount():
            return
        # Read both steps
        steps = self._read_table_steps()
        steps[row], steps[new_row] = steps[new_row], steps[row]
        # Re-render
        self.table.setRowCount(0)
        for i, st in enumerate(steps):
            self.table.insertRow(i)
            self._set_table_row(i, st)
        self.table.selectRow(new_row)

    def _on_clear_steps(self):
        if QMessageBox.question(self, "Clear Steps", "Clear all steps?") != QMessageBox.StandardButton.Yes:
            return
        self.table.setRowCount(0)
        self._set_status("Cleared steps.")

    def _renumber(self):
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0) is None:
                self.table.setItem(r, 0, QTableWidgetItem(""))
            self.table.item(r, 0).setText(str(r + 1))

    def _selected_row(self) -> int:
        items = self.table.selectedItems()
        if not items:
            return -1
        return items[0].row()

    def _set_status(self, msg: str):
        self.status_lbl.setText(msg)

    # =========================
    # Recipe persistence
    # =========================

    def _load_recipes_from_disk(self):
        self._recipes = {}
        if not self.recipes_path or not os.path.exists(self.recipes_path):
            self._set_status("No recipes file yet.")
            return
        try:
            with open(self.recipes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, blob in (data or {}).items():
                steps = []
                for st in blob.get("steps", []):
                    # Handle both old format (x_fs) and new format (x_us)
                    if "x_fs" in st and "x_us" not in st:
                        # Legacy: convert full-steps to microsteps
                        steps.append(RecipeStep(
                            step_type=st.get("step_type", "move"),
                            x_us=int(round(float(st.get("x_fs", 0)) * MRES)),
                            y_us=int(round(float(st.get("y_fs", 0)) * MRES)),
                            z_us=int(round(float(st.get("z_fs", 0)) * MRES)),
                            speed=float(st.get("speed", 1.0)),
                            dwell_ms=int(st.get("dwell_ms", 0) or 0),
                            override_z_limit=bool(st.get("override_z_limit", False)),
                            run_calibration_during_dwell=bool(
                                st.get("run_calibration_during_dwell", False)),
                            num_cycles=int(st.get("num_cycles", 1) or 1),
                            scan_size=int(st.get("scan_size", 5) or 5),
                            note=str(st.get("note", "") or ""),
                        ))
                    else:
                        steps.append(RecipeStep(
                            step_type=st.get("step_type", "move"),
                            x_us=int(st.get("x_us", 0)),
                            y_us=int(st.get("y_us", 0)),
                            z_us=int(st.get("z_us", 0)),
                            speed=float(st.get("speed", 1.0)),
                            dwell_ms=int(st.get("dwell_ms", 0) or 0),
                            override_z_limit=bool(st.get("override_z_limit", False)),
                            run_calibration_during_dwell=bool(
                                st.get("run_calibration_during_dwell", False)),
                            num_cycles=int(st.get("num_cycles", 1) or 1),
                            scan_size=int(st.get("scan_size", 5) or 5),
                            note=str(st.get("note", "") or ""),
                        ))
                self._recipes[name] = MoveRecipe(name=name, steps=steps)
        except Exception as e:
            QMessageBox.warning(self, "Move Planner", f"Failed to load recipes:\n{e}")

    def _save_recipes_to_disk(self):
        try:
            os.makedirs(os.path.dirname(self.recipes_path), exist_ok=True)
        except Exception:
            pass
        try:
            out = {}
            for name, r in self._recipes.items():
                out[name] = {
                    "name": r.name,
                    "steps": [asdict(s) for s in r.steps]
                }
            with open(self.recipes_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Move Planner", f"Failed to save recipes:\n{e}")

    def _refresh_recipe_combo(self):
        current = self.recipe_combo.currentText().strip()
        self.recipe_combo.blockSignals(True)
        self.recipe_combo.clear()
        names = sorted(self._recipes.keys())
        self.recipe_combo.addItems(names)
        self.recipe_combo.blockSignals(False)
        if current and current in self._recipes:
            self.recipe_combo.setCurrentText(current)
        elif names:
            self.recipe_combo.setCurrentIndex(0)
        else:
            self._active_recipe_name = None
            self.table.setRowCount(0)

    def _render_recipe_steps(self, recipe: Optional[MoveRecipe]):
        self.table.setRowCount(0)
        if recipe is None:
            return
        for i, st in enumerate(recipe.steps):
            self.table.insertRow(i)
            self._set_table_row(i, st)

    # =========================
    # Recipe actions
    # =========================

    def _on_recipe_selected(self, name: str):
        name = (name or "").strip()
        if not name or name not in self._recipes:
            self._active_recipe_name = None
            self.table.setRowCount(0)
            return
        self._active_recipe_name = name
        self._render_recipe_steps(self._recipes[name])

    def _on_new_recipe(self):
        if self._exec_active:
            QMessageBox.information(self, "Planner", "Stop execution first.")
            return
        name = self.recipe_name_edit.text().strip()
        if not name:
            name, ok = QInputDialog.getText(self, "New Recipe", "Recipe name:")
            if not ok:
                return
            name = name.strip()
        if not name:
            return
        if name in self._recipes:
            QMessageBox.warning(self, "Planner", f"'{name}' already exists.")
            return
        self._recipes[name] = MoveRecipe(name=name, steps=[])
        self._save_recipes_to_disk()
        self._refresh_recipe_combo()
        self.recipe_combo.setCurrentText(name)
        self.recipe_name_edit.clear()

    def _on_save_recipe(self):
        if self._exec_active:
            QMessageBox.information(self, "Planner", "Stop execution first.")
            return
        name = self._active_recipe_name
        typed = self.recipe_name_edit.text().strip()
        if typed:
            name = typed
        if not name:
            name, ok = QInputDialog.getText(self, "Save Recipe", "Recipe name:")
            if not ok:
                return
            name = name.strip()
        if not name:
            return
        steps = self._read_table_steps()
        self._recipes[name] = MoveRecipe(name=name, steps=steps)
        self._save_recipes_to_disk()
        self._refresh_recipe_combo()
        self.recipe_combo.setCurrentText(name)
        self.recipe_name_edit.clear()
        self._set_status(f"Saved '{name}' ({len(steps)} steps).")

    def _on_delete_recipe(self):
        if self._exec_active:
            return
        name = self.recipe_combo.currentText().strip()
        if not name or name not in self._recipes:
            return
        if QMessageBox.question(self, "Delete", f"Delete '{name}'?") != QMessageBox.StandardButton.Yes:
            return
        del self._recipes[name]
        self._save_recipes_to_disk()
        self._refresh_recipe_combo()

    # =========================
    # Z-limit override
    # =========================

    def _override_z_limit(self):
        """Temporarily raise z_global_max to allow high-Z moves."""
        try:
            motion = self.stepper.motion
            self._saved_z_limit = motion._sb_z_global_max
            motion._sb_z_global_max = 1200.0  # max possible
            print(f"[Planner] Z limit overridden: {self._saved_z_limit} -> 1200")
        except Exception as e:
            print(f"[Planner] Z limit override failed: {e}")

    def _restore_z_limit(self):
        """Restore z_global_max to its original value."""
        if self._saved_z_limit is not None:
            try:
                self.stepper.motion._sb_z_global_max = self._saved_z_limit
                print(f"[Planner] Z limit restored: {self._saved_z_limit}")
            except Exception:
                pass
            self._saved_z_limit = None

    # =========================
    # Execution
    # =========================

    def _on_execute(self):
        if self.mode != 'full' or self.btn_execute is None:
            return
        if self._exec_active:
            return
        if self.table.rowCount() == 0:
            QMessageBox.warning(self, "Planner", "Add at least one step.")
            return
        if not hasattr(self.stepper, "move_to_ctrl"):
            QMessageBox.critical(self, "Planner", "No move controller.")
            return
        try:
            if self.stepper.homing.is_active() or self.stepper.move_to_ctrl.is_active():
                QMessageBox.information(self, "Busy", "Motors are already moving.")
                return
        except Exception:
            pass

        # Save current speed
        try:
            self._saved_speed_mult = self.stepper.mc.get_speed_multiplier()
        except Exception:
            self._saved_speed_mult = 1.0

        self._exec_active = True
        self._exec_index = -1
        self._micro_seq_index = -1
        self._micro_seq_source = []
        if self.btn_execute:
            self.btn_execute.setEnabled(False)
        if self.btn_stop:
            self.btn_stop.setEnabled(True)
        self._set_status("Executing recipe...")
        self._advance_exec_step()

    def _on_stop(self):
        if self.mode != 'full':
            return
        if not self._exec_active:
            return
        self._dwell_timer.stop()
        self._countdown_timer.stop()
        self._exec_active = False
        self._exec_index = -1
        self._micro_seq_index = -1
        self._micro_seq_source = []
        self._cal_prep_moving = False
        self._cal_skip_move_to_center = False
        # Stop any running cal we kicked off during a dwell
        self._cleanup_cal_during_dwell(stop_cal=True)
        # Stop any running spiral scan we kicked off
        self._cleanup_spiral_scan(stop_scan=True)
        self._restore_z_limit()
        # Restore speed
        try:
            self.stepper.mc.set_speed_multiplier(self._saved_speed_mult)
        except Exception:
            pass
        try:
            self.stepper._on_stop()
        except Exception:
            pass
        if self.btn_execute:
            self.btn_execute.setEnabled(True)
        if self.btn_stop:
            self.btn_stop.setEnabled(False)
        self._set_status("Execution stopped.")

    def _advance_exec_step(self):
        if not self._exec_active:
            return

        # If we're in the middle of a microfluidics sub-sequence, continue it
        if self._micro_seq_index >= 0:
            self._advance_microfluidics()
            return

        self._exec_index += 1
        steps = self._read_table_steps()

        if self._exec_index >= len(steps):
            self._finish_execution("Recipe complete.")
            return

        step = steps[self._exec_index]
        self.table.selectRow(self._exec_index)


        if step.step_type == "move":
            self._execute_move_step(step)
        elif step.step_type == "microfluidics":
            # Legacy: run full push1 + push2 as a single expanded sequence.
            self._pending_dwell_ms = step.dwell_ms
            self._pending_cal_during_dwell = False
            self._micro_seq_source = list(MICROFLUIDICS_SEQUENCE)
            self._micro_seq_index = 0
            self._set_status(f"Step {self._exec_index + 1}: Microfluidics (legacy)...")
            self._advance_microfluidics()
        elif step.step_type == "microfluidics_push1":
            self._pending_dwell_ms = step.dwell_ms
            self._pending_cal_during_dwell = False
            self._micro_seq_source = list(MICROFLUIDICS_PUSH1)
            self._micro_seq_index = 0
            self._set_status(f"Step {self._exec_index + 1}: µFluidics Push 1...")
            self._advance_microfluidics()
        elif step.step_type == "microfluidics_push2":
            self._pending_dwell_ms = step.dwell_ms
            self._pending_cal_during_dwell = False
            self._micro_seq_source = list(MICROFLUIDICS_PUSH2)
            self._micro_seq_index = 0
            self._set_status(f"Step {self._exec_index + 1}: µFluidics Push 2...")
            self._advance_microfluidics()
        elif step.step_type == "calibration":
            # Move to SAFE_CENTER as a plain planner-driven move, then hand
            # off to the dwell-with-cal path with OrientCal skipping its own
            # move-to-center. Decouples move completion from AF start.
            self._pending_dwell_ms = 0
            self._pending_cal_during_dwell = False
            self._cal_prep_moving = True
            self._override_z_limit()
            try:
                self.stepper.mc.set_speed_multiplier(1.0)
            except Exception:
                pass
            self._set_status(
                f"Step {self._exec_index + 1}: Moving to calibration center..."
            )
            try:
                self.stepper._preempt_all_motion()
                self.stepper._all_btn_enabled(False)
                self.stepper.move_to_ctrl.start(
                    SAFE_CENTER_X_FS, SAFE_CENTER_Y_FS, 0.0
                )
            except Exception as e:
                self._cal_prep_moving = False
                self._finish_execution(f"Move to cal center failed: {e}")
        elif step.step_type == "spiral_scan":
            self._pending_dwell_ms = step.dwell_ms
            self._pending_cal_during_dwell = False
            self._execute_spiral_scan_step(step)

    def _execute_move_step(self, step: RecipeStep):
        """Execute a single move step."""
        x_fs = float(step.x_us) / MRES
        y_fs = float(step.y_us) / MRES
        z_fs = float(step.z_us) / MRES

        self._pending_dwell_ms = step.dwell_ms
        self._pending_cal_during_dwell = bool(
            getattr(step, "run_calibration_during_dwell", False))

        # Override Z limit if needed
        if step.override_z_limit:
            self._override_z_limit()
        else:
            self._restore_z_limit()

        # Set speed for this step
        try:
            self.stepper.mc.set_speed_multiplier(step.speed)
        except Exception:
            pass

        self._set_status(
            f"Step {self._exec_index + 1}: Move ({step.x_us},{step.y_us},{step.z_us}) "
            f"x{step.speed:.2f}"
        )

        try:
            self.stepper._preempt_all_motion()
            self.stepper._all_btn_enabled(False)
            self.stepper.move_to_ctrl.start(x_fs, y_fs, z_fs)
        except Exception as e:
            self._restore_z_limit()
            self._finish_execution(f"Move failed: {e}")

    # =========================
    # Microfluidics sub-sequence
    # =========================

    def _advance_microfluidics(self):
        """Execute the next step in the currently-selected microfluidics
        sub-sequence (either PUSH1, PUSH2, or the legacy full sequence,
        depending on which step_type triggered execution)."""
        if not self._exec_active:
            return
        seq = self._micro_seq_source
        if self._micro_seq_index >= len(seq):
            # Sub-sequence complete
            self._micro_seq_index = -1
            self._micro_seq_source = []
            self._restore_z_limit()
            self._set_status(f"Step {self._exec_index + 1}: Microfluidics complete.")
            self._handle_dwell_or_advance()
            return

        sub_step = seq[self._micro_seq_index]
        self._micro_seq_index += 1

        self._set_status(
            f"Step {self._exec_index + 1}: µFluidics "
            f"[{self._micro_seq_index}/{len(seq)}]"
        )

        self._execute_move_step(sub_step)

    # =========================
    # Spiral scan step
    # =========================

    def _execute_spiral_scan_step(self, step: RecipeStep):
        """Kick off the spiral scan controller. Advance once one full cycle
        completes (detected via progress signal phase='waiting'). The post-step
        dwell, if any, runs after the cycle completes the same way it would
        for any other step."""
        if not self._exec_active:
            return

        if self._spiral_scan is None:
            print("[Planner] spiral_scan step requested but no spiral_scan "
                  "controller wired up — skipping.")
            self._set_status(f"Step {self._exec_index + 1}: Spiral Scan: no controller, skipping.")
            self._handle_dwell_or_advance()
            return

        if self._spiral_scan.is_active():
            print("[Planner] spiral_scan step requested but a scan is already "
                  "active — skipping.")
            self._set_status(f"Step {self._exec_index + 1}: Spiral Scan: already active, skipping.")
            self._handle_dwell_or_advance()
            return

        # Resolve camera settings
        camera_settings = {}
        try:
            if callable(self._camera_settings_provider):
                camera_settings = self._camera_settings_provider() or {}
        except Exception as e:
            print(f"[Planner] Camera settings provider failed: {e}")
            camera_settings = {}

        # Resolve scan settings (interval, center_col, center_row)
        interval_min = 5
        center_col = 15
        center_row = 15
        try:
            if callable(self._spiral_scan_settings_provider):
                vals = self._spiral_scan_settings_provider() or {}
                # Provider may return a dict or a 3-tuple
                if isinstance(vals, dict):
                    interval_min = int(vals.get('interval_min', interval_min))
                    center_col = int(vals.get('center_col', center_col))
                    center_row = int(vals.get('center_row', center_row))
                else:
                    interval_min = int(vals[0])
                    center_col = int(vals[1])
                    center_row = int(vals[2])
        except Exception as e:
            print(f"[Planner] Spiral scan settings provider failed: {e}, "
                  f"using defaults ({interval_min}min, {center_col},{center_row})")

        # Configure the scan — scan_size + num_cycles come from the step itself
        # (so each spiral_scan step in a sequence can have its own settings).
        try:
            self._spiral_scan.configure(
                interval_min=interval_min,
                center_col=center_col,
                center_row=center_row,
                scan_size=int(getattr(step, 'scan_size', 5)),
                num_cycles=int(getattr(step, 'num_cycles', 1)),
            )
        except Exception as e:
            print(f"[Planner] spiral_scan.configure failed: {e}")
            self._finish_execution(f"Spiral scan configure failed: {e}")
            return

        # Connect to progress (stop after one full cycle enters 'waiting')
        # and to finished (errors / external stops).
        try:
            self._spiral_scan.progress.connect(self._on_spiral_scan_progress)
        except Exception:
            pass
        try:
            self._spiral_scan.finished.connect(self._on_spiral_scan_finished)
        except Exception:
            pass

        self._spiral_scan_running_for_step = True

        self._set_status(
            f"Step {self._exec_index + 1}: Spiral Scan "
            f"(center=({center_col},{center_row}), interval={interval_min}min)..."
        )

        # Quiesce barrier: cal-during-dwell's AF drives Z directly past
        # MoveTo. Stop it and wait for all motion to idle before scanning.
        try:
            self._cleanup_cal_during_dwell(stop_cal=True)
        except Exception:
            pass
        try:
            self.stepper._preempt_all_motion()
        except Exception:
            pass

        self._pending_spiral_camera_settings = camera_settings
        self._spiral_quiesce_polls = 0
        if getattr(self, "_spiral_quiesce_timer", None) is None:
            self._spiral_quiesce_timer = QTimer(self)
            self._spiral_quiesce_timer.setInterval(100)
            self._spiral_quiesce_timer.timeout.connect(self._poll_spiral_quiesce)
        self._set_status(
            f"Step {self._exec_index + 1}: Spiral Scan — waiting for "
            f"calibration/motion to settle..."
        )
        self._spiral_quiesce_timer.start()

    # Max wait for cal/motion to settle before starting the scan anyway
    # (100ms poll * 40 = 4s). Bounded so a stuck controller can't hang the recipe.
    SPIRAL_QUIESCE_MAX_POLLS = 40

    def _poll_spiral_quiesce(self):
        """Wait until OrientCal/AF/homing/MoveTo all report idle, then start
        the spiral scan. The cal-during-dwell AF drives Z directly outside the
        safety layer, so starting the scan's move while it is still active
        causes simultaneous unsafe XY+Z motion."""
        if not self._exec_active or not self._spiral_scan_running_for_step:
            self._spiral_quiesce_timer.stop()
            return

        busy = False
        try:
            if self._orient_cal is not None and self._orient_cal.is_active():
                busy = True
            if self.stepper.homing.is_active() or self.stepper.move_to_ctrl.is_active():
                busy = True
            af = getattr(self._spiral_scan, "af_controller", None)
            if af is not None and af.is_active():
                busy = True
        except Exception:
            pass

        self._spiral_quiesce_polls += 1
        timed_out = self._spiral_quiesce_polls >= self.SPIRAL_QUIESCE_MAX_POLLS
        if busy and not timed_out:
            return

        self._spiral_quiesce_timer.stop()
        if busy and timed_out:
            print("[Planner] WARNING: cal/motion still active after quiesce "
                  "timeout — starting spiral scan anyway")
        else:
            print("[Planner] Quiesce complete (motion idle) — starting spiral scan.")

        try:
            self._spiral_scan.start(self._pending_spiral_camera_settings)
            print(f"[Planner] Spiral scan started for recipe step.")
        except Exception as e:
            print(f"[Planner] spiral_scan.start failed: {e}")
            self._cleanup_spiral_scan(stop_scan=False)
            self._finish_execution(f"Spiral scan start failed: {e}")

    def _on_spiral_scan_progress(self, phase, message):
        """Status passthrough while the scan runs. Termination after N cycles
        is owned by the controller itself (configure(num_cycles=N) makes it
        stop after N completed cycles), which fires finished → advance."""
        if not self._spiral_scan_running_for_step:
            return
        if not self._exec_active:
            return
        if phase == 'waiting':
            self._set_status(
                f"Step {self._exec_index + 1}: Spiral Scan running. {message}"
            )

    def _on_spiral_scan_finished(self, success, message):
        """Handle the scan's finished signal. Fires when:
          - we asked it to stop after one cycle (success=False, but expected),
          - the scan errored,
          - the user pressed stop elsewhere.
        Either way: clean up signal connections and advance the recipe."""
        if not self._spiral_scan_running_for_step:
            return
        if not self._exec_active:
            self._cleanup_spiral_scan(stop_scan=False)
            return

        print(f"[Planner] Spiral scan finished: ok={success} msg={message}")
        self._cleanup_spiral_scan(stop_scan=False)
        # Advance through normal dwell-or-advance path.
        self._handle_dwell_or_advance()

    def _cleanup_spiral_scan(self, stop_scan: bool):
        """Disconnect our slots and optionally stop the running scan."""
        if self._spiral_scan is not None:
            try:
                self._spiral_scan.progress.disconnect(self._on_spiral_scan_progress)
            except Exception:
                pass
            try:
                self._spiral_scan.finished.disconnect(self._on_spiral_scan_finished)
            except Exception:
                pass
            if stop_scan:
                try:
                    if self._spiral_scan.is_active():
                        self._spiral_scan.stop("Recipe stopped")
                except Exception:
                    pass
        self._spiral_scan_running_for_step = False

    # =========================
    # Move completion handler
    # =========================

    def _on_move_to_finished(self, success: bool, message: str):
        if not self._exec_active:
            return

        # If a dwell is currently running, this move-finished is from
        # something OTHER than our recipe (e.g. orientation calibration
        # moving the stage during the dwell). The dwell timer / cal-finished
        # signal is in charge of advancing — not move_to_ctrl.
        if self._dwell_timer.isActive() or self._waiting_for_cal \
                or self._cal_running_for_step:
            return

        # Same idea for an active spiral scan: it issues its own moves.
        if self._spiral_scan_running_for_step:
            return

        if not success:
            self._restore_z_limit()
            try:
                self.stepper.mc.set_speed_multiplier(self._saved_speed_mult)
            except Exception:
                pass
            # If this was the pre-cal move to SAFE_CENTER, clear the flag so
            # a stray retry can't re-enter cal handoff.
            self._cal_prep_moving = False
            self._finish_execution(message or "Move failed.")
            return

        # Standalone-calibration pre-move to SAFE_CENTER just completed.
        # Hand off to the working cal-during-dwell path but tell OrientCal
        # to skip its own move-to-center (we already drove there).
        if self._cal_prep_moving:
            self._cal_prep_moving = False
            self._pending_dwell_ms = 0
            self._pending_cal_during_dwell = True
            self._cal_skip_move_to_center = True
            self._set_status(
                f"Step {self._exec_index + 1}: Calibration (~30 min)..."
            )
            self._start_dwell(1)
            return

        # If in microfluidics sub-sequence, check dwell then advance
        if self._micro_seq_index >= 0:
            dwell_ms = int(getattr(self, "_pending_dwell_ms", 0) or 0)
            if dwell_ms > 0:
                self._start_dwell(dwell_ms)
                return
            self._advance_microfluidics()
            return

        # Normal move step complete
        self._handle_dwell_or_advance()

    def _handle_dwell_or_advance(self):
        dwell_ms = int(getattr(self, "_pending_dwell_ms", 0) or 0)
        if dwell_ms > 0:
            self._start_dwell(dwell_ms)
        else:
            self._advance_exec_step()

    def _start_dwell(self, dwell_ms: int):
        """Start a dwell. If the step requested calibration during the dwell,
        kick it off in parallel — the visible countdown timer is unaffected
        and continues to display remaining dwell time as usual."""
        self._dwell_remaining_ms = dwell_ms
        self._pending_dwell_ms = 0
        self._update_countdown()
        self._countdown_timer.start()
        self._dwell_timer.start(dwell_ms)

        if self._pending_cal_during_dwell:
            self._pending_cal_during_dwell = False
            self._kick_off_calibration_during_dwell()

    def _kick_off_calibration_during_dwell(self):
        """Start orientation calibration in parallel with the dwell."""
        if self._orient_cal is None:
            print("[Planner] run_calibration_during_dwell requested but no "
                  "orient_cal controller wired up — skipping cal, dwell continues.")
            return
        if self._orient_cal.is_active():
            print("[Planner] Calibration already active when dwell started; "
                  "tracking it as the dwell's calibration.")
            self._cal_running_for_step = True
            try:
                self._orient_cal.finished.connect(self._on_dwell_calibration_finished)
            except Exception:
                pass
            return

        # Resolve camera settings
        settings = {}
        try:
            if callable(self._camera_settings_provider):
                settings = self._camera_settings_provider() or {}
        except Exception as e:
            print(f"[Planner] Camera settings provider failed: {e}")
            settings = {}

        # Connect first so we don't miss a fast finish
        try:
            self._orient_cal.finished.connect(self._on_dwell_calibration_finished)
        except Exception:
            pass

        # Set flag before start() in case start() synchronously emits finished.
        self._cal_running_for_step = True

        # Defer actual start to the next event loop tick: we're reached from
        # inside a move_to_ctrl.finished emission, and starting cal
        # synchronously would let OrientCal's own slot misinterpret the
        # stale signal as its move-to-center completing.
        QTimer.singleShot(0, lambda s=settings: self._deferred_cal_start(s))

    def _deferred_cal_start(self, settings):
        """Start orient cal on a clean stack (see _kick_off_calibration_during_dwell).
        Bail if the recipe was stopped before this fired."""
        if not self._exec_active or not self._cal_running_for_step:
            return
        if self._orient_cal is None:
            return
        # If the planner already drove the stage to SAFE_CENTER (standalone
        # calibration step), tell OrientCal to skip its internal
        # move-to-center and go straight to AF. Cleared after use.
        skip = getattr(self, '_cal_skip_move_to_center', False)
        self._cal_skip_move_to_center = False
        try:
            self._orient_cal.start(settings, skip_move_to_center=skip)
            print(f"[Planner] Orientation calibration started "
                  f"(skip_move_to_center={skip}).")
        except Exception as e:
            self._cal_running_for_step = False
            print(f"[Planner] Failed to start calibration during dwell: {e}")
            try:
                self._orient_cal.finished.disconnect(self._on_dwell_calibration_finished)
            except Exception:
                pass

    def _on_dwell_timeout(self):
        """Dwell timer expired. If a calibration we kicked off is still
        running, hold off advancing until cal finishes. Otherwise advance."""
        self._countdown_timer.stop()
        if self._cal_running_for_step and self._orient_cal is not None \
                and self._orient_cal.is_active():
            self._waiting_for_cal = True
            # Status hint, but we don't touch the countdown label format —
            # remaining time has reached zero, so this just sits at 00:00.
            self._set_status(
                f"Step {self._exec_index + 1}: dwell complete, "
                f"waiting for calibration to finish..."
            )
            return
        self._advance_after_dwell()

    def _on_dwell_calibration_finished(self, *args):
        """Slot for orient_cal.finished. Signature is (success, message, result)
        but we accept *args defensively."""
        try:
            self._orient_cal.finished.disconnect(self._on_dwell_calibration_finished)
        except Exception:
            pass

        if not self._cal_running_for_step:
            return  # not ours to handle

        success = bool(args[0]) if len(args) >= 1 else False
        message = str(args[1]) if len(args) >= 2 else ""
        self._cal_running_for_step = False

        # The cal completion path goes through RecordSetupScreen._on_orient_cal_finished
        # → stepper.end_calibration() → _all_btn_enabled(True), which re-enables
        # all motor buttons even though the recipe is still running. Re-disable
        # them here so the user can't poke the stage during the rest of the dwell.
        if self._exec_active:
            try:
                self.stepper._all_btn_enabled(False)
            except Exception:
                pass

        if not self._waiting_for_cal:
            # Cal finished while dwell still running — let the dwell finish on its own
            print(f"[Planner] Calibration finished during dwell "
                  f"(success={success}): {message}")
            return

        # Dwell already expired and we were holding — advance now
        self._waiting_for_cal = False
        print(f"[Planner] Calibration finished, releasing dwell hold "
              f"(success={success}): {message}")
        self._advance_after_dwell()

    def _advance_after_dwell(self):
        """_advance_exec_step routes to microfluidics if we're in a sub-sequence
        (it checks _micro_seq_index >= 0 first)."""
        if not self._exec_active:
            return
        self._advance_exec_step()

    def _cleanup_cal_during_dwell(self, stop_cal: bool):
        """Disconnect our cal-finished slot and optionally stop the running cal."""
        if self._orient_cal is not None:
            try:
                self._orient_cal.finished.disconnect(self._on_dwell_calibration_finished)
            except Exception:
                pass
            if stop_cal:
                try:
                    if self._orient_cal.is_active():
                        self._orient_cal.stop("Recipe stopped")
                except Exception:
                    pass
        self._cal_running_for_step = False
        self._waiting_for_cal = False

    def _update_countdown(self):
        remaining = self._dwell_remaining_ms
        if remaining <= 0:
            self._countdown_timer.stop()
            return
        mins = remaining // 60000
        secs = (remaining % 60000) // 1000
        self._set_status(f"Step {self._exec_index + 1}: waiting {mins:02d}:{secs:02d}")
        self._dwell_remaining_ms -= 1000

    def _finish_execution(self, msg: str):
        self._dwell_timer.stop()
        self._countdown_timer.stop()
        self._exec_active = False
        self._exec_index = -1
        self._micro_seq_index = -1
        self._micro_seq_source = []
        # Disconnect our cal slot but don't stop a running cal — if it's still
        # going when the recipe ends naturally, let it finish.
        self._cleanup_cal_during_dwell(stop_cal=False)
        self._cleanup_spiral_scan(stop_scan=False)
        self._restore_z_limit()
        # Restore speed
        try:
            self.stepper.mc.set_speed_multiplier(self._saved_speed_mult)
        except Exception:
            pass
        if self.btn_execute:
            self.btn_execute.setEnabled(True)
        if self.btn_stop:
            self.btn_stop.setEnabled(False)
        self._set_status(msg)