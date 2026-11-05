from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QFrame, QVBoxLayout, QFormLayout,
    QLabel, QSlider, QSpinBox, QPushButton, QComboBox, QSizePolicy, QMessageBox, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer
from widgets.widget_joystick_xy import JoystickXYWidget
from widgets.widget_joystick_z import JoystickZWidget
from widgets.widget_collapsible_box import CollapsibleBox
from motors.stepper_motors_photoswitch import StepperMotorsPhotoswitch
from config import save_config_atomic
from motors.stepper_motors_motion import StepperMotorsMotion, MotionMode
from motors.stepper_motors_homing import StepperMotorsHoming
from motors.stepper_motors_moveto import StepperMotorsMoveto
from PyQt6.QtCore import pyqtSignal as Signal

class StepperMotorsWidget(QWidget):

    auto_focus_requested = Signal()
    calibrate_requested = Signal()
    calibrate_stop_requested = Signal()
    move_to_feature_requested = Signal(int, int)  # col, row
    spiral_scan_requested = Signal()
    spiral_scan_stop_requested = Signal()
    spiral_scan_settings_changed = Signal(int, int, int, int, int)
    # interval_min, center_col, center_row, scan_size, num_cycles (0=unlimited)

    def __init__(self, parent_layout, config, mc):
        super().__init__()
        self.parent_layout = parent_layout
        self.config = config
        self.config_path = config.cfg_path
        self.MOTORS = getattr(self.config, "motor", {})
        self.mc = mc
        self.mc.signals.step.connect(self._on_step_update)

        self._move_thread = None
        self._move_worker = None

        self._build_stepper_motor_UI()
        self._init_photoelectric_switches()
        self._update_motor_ui_from_config()
        self._sync_mc_logical_from_saved()

        self.motion = StepperMotorsMotion(self.mc, self.config, self.MOTORS, parent=self)
        self.motion.statusChanged.connect(self._set_status)
        self.homing = StepperMotorsHoming(self.mc, self.motion, self.photo_switch_monitor, self.config, self.MOTORS, parent=self)
        self.homing.statusChanged.connect(self._set_status)
        self.homing.axisHomed.connect(self._on_axis_homed_freeze)
        self.homing.finished.connect(self._on_homing_done)
        self.homing.z_home_finished.connect(self._on_z_home_cleanup)
        self.move_to_ctrl = StepperMotorsMoveto(self.motion, self.MOTORS, mc=self.mc, vmax=float(getattr(getattr(self.config,'motion',None),'max_speed_fullstep',400.0)), parent=self)
        self.move_to_ctrl.set_homing(self.homing)
        self.move_to_ctrl.statusChanged.connect(self._set_status)
        self.move_to_ctrl.finished.connect(self._on_move_to_done)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._flush_config_to_disk)
        self._save_debounce_ms = 500  # save at most twice per second while moving

        self._freeze_step_updates = set()

        # Set spinbox range from mc's actual limits
        try:
            self.speed_multiplier_spin.setRange(
                self.mc.get_min_multiplier(),
                self.mc.get_max_multiplier()
            )
        except Exception:
            pass

    def _build_stepper_motor_UI(self):
        box = CollapsibleBox("Stepper Motor Control (XYZ)")

        panel = QFrame()
        # Main Horizontal Layout (Left Col | Right Col)
        panel_layout = QHBoxLayout(panel)
        # Margins: Left, Top, Right, Bottom (Increased Bottom to 40 to prevent cutoff)
        panel_layout.setContentsMargins(20, 20, 20, 60)
        panel_layout.setSpacing(30)

        # ============================================
        # LEFT COLUMN (Status, Sliders, Joysticks)
        # ============================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)

        # 1. Status Row
        status_row = QHBoxLayout()
        motor_status_label = QLabel("Motor Status:")
        motor_status_label.setStyleSheet("font-weight: bold;")
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color: gray;")
        status_row.addWidget(motor_status_label)
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        left_layout.addLayout(status_row)

        # 2. Sliders (Labels above, Slider+Spinbox below)
        self.x_position_label = QLabel("X Pos:")
        self.y_position_label = QLabel("Y Pos:")
        self.z_position_label = QLabel("Z Pos:")

        self.x_bar, self.y_bar, self.z_bar = QSlider(Qt.Orientation.Horizontal), QSlider(Qt.Orientation.Horizontal), QSlider(Qt.Orientation.Horizontal)
        self.x_step, self.y_step, self.z_step = QSpinBox(), QSpinBox(), QSpinBox()

        # --- Microstepping is locked at 32 for all axes ---
        # (handled in StepperMotorHardware.__init__)
        for ax in ("x", "y", "z"):
            mres = self.mc.get_axis_microstepping(ax)
            print(f"[INIT] {ax.upper()} microstepping = {mres} (locked)")

        self._add_axis_control("x", left_layout)
        self._add_axis_control("y", left_layout)
        self._add_axis_control("z", left_layout)

        # Sync sliders and spinboxes
        self.x_bar.valueChanged.connect(self.x_step.setValue)
        self.x_step.valueChanged.connect(self.x_bar.setValue)
        self.y_bar.valueChanged.connect(self.y_step.setValue)
        self.y_step.valueChanged.connect(self.y_bar.setValue)
        self.z_bar.valueChanged.connect(self.z_step.setValue)
        self.z_step.valueChanged.connect(self.z_bar.setValue)

        # 3. Spacer before Joysticks
        left_layout.addSpacing(10)

        # 4. Joysticks
        self._build_joysticks(left_layout)

        # Push everything up
        left_layout.addStretch()


        # ============================================
        # RIGHT COLUMN (Microsteps, Buttons)
        # ============================================
        right_widget = QWidget()
        right_widget.setFixedWidth(240) # Slightly wider for cleaner look
        right_widget.setMinimumHeight(800)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)

        # 1. Speed Multiplier Control
        from PyQt6.QtWidgets import QDoubleSpinBox
        speed_grid = QGridLayout()
        speed_grid.setVerticalSpacing(10)

        speed_label = QLabel("Speed (x):")
        speed_label.setStyleSheet("font-weight: bold;")
        speed_grid.addWidget(speed_label, 0, 0)

        self.speed_multiplier_spin = QDoubleSpinBox()
        self.speed_multiplier_spin.setDecimals(2)
        self.speed_multiplier_spin.setSingleStep(0.1)
        # Range will be set after mc is available
        self.speed_multiplier_spin.setRange(0.01, 2.67)
        self.speed_multiplier_spin.setValue(1.0)
        self.speed_multiplier_spin.setPrefix("x")
        self.speed_multiplier_spin.valueChanged.connect(self._on_speed_multiplier_changed)
        speed_grid.addWidget(self.speed_multiplier_spin, 0, 1)

        # Show actual speed info
        self.speed_info_label = QLabel("")
        self.speed_info_label.setStyleSheet("font-size: 11pt; color: gray;")
        speed_grid.addWidget(self.speed_info_label, 1, 0, 1, 2)
        self._update_speed_info_label()

        right_layout.addLayout(speed_grid)
        right_layout.addSpacing(10)

        # 2. Buttons
        self.home_btn = QPushButton("Home Stage")
        self.home_btn.setMinimumHeight(55)
        self.home_btn.clicked.connect(self._on_home_stage)

        self.move_to_btn = QPushButton("Move to Position")
        self.move_to_btn.setMinimumHeight(55)
        self.move_to_btn.clicked.connect(self._on_move_to)

        self.stop_motor_btn = QPushButton("Stop Motor")
        self.stop_motor_btn.setMinimumHeight(55)
        self.stop_motor_btn.clicked.connect(self._on_stop)

        self.auto_focus_btn = QPushButton("Auto Focus")
        self.auto_focus_btn.setMinimumHeight(55)
        self.auto_focus_btn.clicked.connect(self._on_auto_focus_clicked)

        self._autofocus_active = False

        self.calibrate_btn = QPushButton("Calibrate Stage")
        self.calibrate_btn.setMinimumHeight(55)
        self.calibrate_btn.clicked.connect(self._on_calibrate_clicked)

        self.move_feature_btn = QPushButton("Move to Feature")
        self.move_feature_btn.setMinimumHeight(55)
        self.move_feature_btn.clicked.connect(self._on_move_to_feature_clicked)

        self._calibration_active = False

        self.spiral_scan_btn = QPushButton("▶ Spiral Scan")
        self.spiral_scan_btn.setMinimumHeight(55)
        self.spiral_scan_btn.setStyleSheet("background-color: #1565C0; color: white; font-weight: bold;")
        self.spiral_scan_btn.clicked.connect(self._on_spiral_scan_clicked)
        self._spiral_scan_active = False

        right_layout.addWidget(self.home_btn)
        right_layout.addWidget(self.move_to_btn)
        right_layout.addWidget(self.stop_motor_btn)
        right_layout.addWidget(self.auto_focus_btn)
        right_layout.addWidget(self.calibrate_btn)
        right_layout.addWidget(self.move_feature_btn)
        right_layout.addWidget(self.spiral_scan_btn)

        # Add columns to main panel
        panel_layout.addWidget(left_widget, stretch=1)
        panel_layout.addWidget(right_widget, stretch=0)

        # ---- CollapsibleBox content layout ----
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(panel)

        box.setContentLayout(content_layout)
        self.parent_layout.addWidget(box)
        self.motor_group = box

    def _build_joysticks(self, parent_layout):
        row = QWidget()
        h = QHBoxLayout(row)
        # Ensure explicit height to prevent cutoff
        row.setMinimumHeight(300)

        h.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        h.setContentsMargins(0, 10, 0, 10)
        h.setSpacing(30)

        self.xy_joysticks = JoystickXYWidget()
        self.z_joystick = JoystickZWidget()

        self.xy_joysticks.setFixedSize(200, 200)
        self.xy_joysticks.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.z_joystick.setFixedSize(90, 200)
        self.z_joystick.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        h.addWidget(self.xy_joysticks)
        h.addWidget(self.z_joystick)
        h.addStretch(1)

        parent_layout.addWidget(row)

        # ── Scan Settings (below joysticks) ──
        # Row heights are tuned to match the global stylesheet's input sizing
        # (min-height 45 + padding + border ≈ 55px); forcing a smaller fixed
        # height clips the rendered text.
        LABEL_W   = 140
        INPUT_H   = 55
        ROW_H     = 60
        ROW_SPACE = 18
        VC = Qt.AlignmentFlag.AlignVCenter

        scan_settings = QWidget()
        scan_outer = QVBoxLayout(scan_settings)
        scan_outer.setContentsMargins(0, 5, 0, 0)
        scan_outer.setSpacing(ROW_SPACE)

        scan_label = QLabel("Scan Settings")
        scan_label.setStyleSheet("font-weight: bold;")
        scan_outer.addWidget(scan_label)

        def _label(text, fixed_width=None):
            lab = QLabel(text)
            lab.setFixedHeight(INPUT_H)
            lab.setAlignment(Qt.AlignmentFlag.AlignLeft | VC)
            if fixed_width is not None:
                lab.setFixedWidth(fixed_width)
            return lab

        def _row(label_text, *widgets):
            row = QWidget()
            row.setFixedHeight(ROW_H)
            row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)
            h.addWidget(_label(label_text, LABEL_W), 0, VC)
            for w in widgets:
                # Force every input/suffix to the same height so the row
                # has a single consistent baseline.
                try:
                    w.setFixedHeight(INPUT_H)
                except Exception:
                    pass
                h.addWidget(w, 0, VC)
            h.addStretch()
            return row

        # Row 1: Scan interval
        self.scan_interval_combo = QComboBox()
        self.scan_interval_combo.addItems([str(m) for m in range(5, 35, 5)])
        self.scan_interval_combo.setCurrentText("5")
        self.scan_interval_combo.setFixedWidth(70)
        scan_outer.addWidget(_row(
            "Scan interval:", self.scan_interval_combo, QLabel("min")))

        # Row 2: Scan center (col + row spinboxes)
        self.scan_center_col_spin = QSpinBox()
        self.scan_center_col_spin.setRange(2, 28)
        self.scan_center_col_spin.setValue(15)
        self.scan_center_col_spin.setPrefix("col ")
        self.scan_center_col_spin.setFixedWidth(95)
        self.scan_center_row_spin = QSpinBox()
        self.scan_center_row_spin.setRange(2, 28)
        self.scan_center_row_spin.setValue(15)
        self.scan_center_row_spin.setPrefix("row ")
        self.scan_center_row_spin.setFixedWidth(95)
        scan_outer.addWidget(_row(
            "Scan center:", self.scan_center_col_spin, self.scan_center_row_spin))

        # Row 3: Scan size
        self.scan_size_combo = QComboBox()
        self.scan_size_combo.addItems(["3", "5", "7", "9", "11", "15", "21", "31"])
        self.scan_size_combo.setCurrentText("5")
        self.scan_size_combo.setFixedWidth(70)
        scan_outer.addWidget(_row(
            "Scan size:", self.scan_size_combo, QLabel("× size")))

        # Row 4: Cycle count
        self.scan_cycles_spin = QSpinBox()
        self.scan_cycles_spin.setRange(0, 999)
        self.scan_cycles_spin.setValue(0)
        self.scan_cycles_spin.setFixedWidth(80)
        self.scan_cycles_spin.setSpecialValueText("∞")
        hint = QLabel("(0 = until stopped)")
        hint.setStyleSheet("color: gray; font-size: 10pt;")
        scan_outer.addWidget(_row("Cycles:", self.scan_cycles_spin, hint))

        parent_layout.addWidget(scan_settings)

        self.xy_joysticks.speedCommand.connect(self._on_xy_joy_speed)
        self.xy_joysticks.stopCommand.connect(self._on_xy_joy_stop)
        self.z_joystick.speedCommand.connect(self._on_z_joy_speed)
        self.z_joystick.stopCommand.connect(self._on_z_joy_stop)

        # Set initial joystick max speeds from the current profile
        self._update_joystick_speeds()

    def _add_axis_control(self, axis, parent_layout):
        """Helper to create Label -> [Slider + Spinbox] structure."""
        # Label Row
        lbl = getattr(self, f"{axis}_position_label")
        parent_layout.addWidget(lbl)

        # Control Row (Slider + Spinbox)
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 5) # 5px bottom margin for spacing

        bar = getattr(self, f"{axis}_bar")
        bar.setMinimumWidth(180)

        spin = getattr(self, f"{axis}_step")
        spin.setFixedWidth(100)

        # Set Ranges
        mres = self._mres(axis)
        full_min = float(getattr(self.MOTORS, f"{axis}_full_step_min", 0.0))
        full_max = float(getattr(self.MOTORS, f"{axis}_full_step_max", 20000))
        ms_min = self._full_to_micro(axis, full_min)
        ms_max = self._full_to_micro(axis, full_max)
        print(f"[AXIS {axis.upper()}] full_step_max={full_max}, mres={mres}, slider/spin range={ms_min}..{ms_max}")
        bar.setRange(ms_min, ms_max)
        spin.setRange(ms_min, ms_max)

        row_layout.addWidget(bar)
        row_layout.addSpacing(10) # Space between slider and spinbox
        row_layout.addWidget(spin)

        parent_layout.addLayout(row_layout)

    def _init_photoelectric_switches(self) -> None:
        active_state = getattr(self.config.endstops, "active_state", "low")
        pull_up = bool(getattr(self.config.endstops, "pull_up", True))
        self.axis_gpio = {ax: getattr(self.config.axes[ax], "photoelectric_switch", None) for ax in ("x","y","z")}
        self.axis_gpio = {ax: gpio for ax, gpio in self.axis_gpio.items() if gpio is not None}

        self.photo_switch_monitor = StepperMotorsPhotoswitch(
            self.axis_gpio,
            pull_up=pull_up,
            active_state=active_state,
            parent=self
        )
        self._homing_active = False

    def _update_motor_ui_from_config(self):
        for ax in ("x", "y", "z"):
            fs = float(getattr(self.MOTORS, f"{ax}_current_position_full_step", 0.0))
            ms = self._full_to_micro(ax, fs)
            bar  = getattr(self, f"{ax}_bar")
            step = getattr(self, f"{ax}_step")
            bar.blockSignals(True); step.blockSignals(True)
            bar.setValue(ms);       step.setValue(ms)
            bar.blockSignals(False); step.blockSignals(False)
            self._update_all_position_labels(ax)

    def _update_position_label(self, axis):
        fs = float(getattr(self.MOTORS, f"{axis}_current_position_full_step", 0.0))
        ms = self._full_to_micro(axis, fs)
        txt = f"{axis.upper()} pos.: {fs:.3f} full ({ms} µ)"
        if axis == "x":   self.x_position_label.setText(txt)
        elif axis == "y": self.y_position_label.setText(txt)
        else:             self.z_position_label.setText(txt)

    def _update_all_position_labels(self, axis: str):
        axis = axis.lower()
        fs = float(getattr(self.MOTORS, f"{axis}_current_position_full_step", 0.0))
        mres = max(1, int(self.mc.get_axis_microstepping(axis)))
        ms = int(round(fs * mres))

        bar  = getattr(self, f"{axis}_bar", None)
        step = getattr(self, f"{axis}_step", None)

        if bar is not None:
            bar.blockSignals(True)
            bar.setValue(ms)
            bar.blockSignals(False)

        if step is not None:
            step.blockSignals(True)
            step.setValue(ms)
            step.blockSignals(False)

        self._update_position_label(axis)

    def _sync_mc_logical_from_saved(self):
        for axis in ("x", "y", "z"):
            fs = float(getattr(self.MOTORS, f"{axis}_current_position_full_step", 0.0))
            mres = max(1, int(self.mc.get_axis_microstepping(axis)))
            logical_micro = int(round(fs * mres))
            self.mc.set_logical_position_micro(axis, logical_micro)

    def _on_speed_multiplier_changed(self, value: float):
        """Handle speed multiplier spinbox change."""
        try:
            self.mc.halt_all_speed()
        except Exception:
            pass

        if getattr(self, "_move_thread", None) and self._move_thread.isRunning():
            # Don't change speed mid-move; revert spinbox
            self.speed_multiplier_spin.blockSignals(True)
            self.speed_multiplier_spin.setValue(self.mc.get_speed_multiplier())
            self.speed_multiplier_spin.blockSignals(False)
            return

        self.mc.set_speed_multiplier(value)
        self._update_speed_info_label()
        self._update_joystick_speeds()

    def _update_speed_info_label(self):
        """Update the info label showing current speed."""
        try:
            mult = self.mc.get_speed_multiplier()
            vmax = self.mc.get_current_vmax()
            self.speed_info_label.setText(f"x{mult:.2f}: {vmax:.0f} fs/s")
        except Exception:
            self.speed_info_label.setText("")

    def _update_joystick_speeds(self, profile_name: str = None):
        """Update joystick max speeds based on current speed."""
        try:
            vmax = self.mc.get_current_vmax()
            self.xy_joysticks.set_max_speed(vmax, vmax)
            self.z_joystick.set_max_speed(vmax)
        except Exception:
            pass

    def _sync_speed_ui_from_mc(self):
        """Sync the speed spinbox, info label, and joystick speeds to match
        whatever the hardware controller currently has active.
        Called after homing/moveto finish and restore speed internally."""
        try:
            mult = self.mc.get_speed_multiplier()
            self.speed_multiplier_spin.blockSignals(True)
            self.speed_multiplier_spin.setValue(mult)
            self.speed_multiplier_spin.blockSignals(False)
            self._update_speed_info_label()
            self._update_joystick_speeds()
        except Exception:
            pass

    def _on_xy_joy_speed(self, vx: float, vy: float):
        if self.homing.is_active() or self.move_to_ctrl.is_active():
            return
        vx_cmd = self._clamp_jog_to_limits("x", float(vx))
        vy_cmd = self._clamp_jog_to_limits("y", float(vy))
        self.motion.set_desired_velocity(vx_cmd, vy_cmd, float(self.motion.v_des["z"]), mode=MotionMode.JOYSTICK)
        if abs(vx_cmd) < 1e-6 and abs(vy_cmd) < 1e-6:
            self._set_status("Idle", "gray")
            self._save_timer.start(self._save_debounce_ms)
        else:
            self._set_status("Jogging…", "orange")

    def _on_xy_joy_stop(self):
        self.motion.set_desired_velocity(0.0, 0.0, float(self.motion.v_des["z"]), mode=MotionMode.IDLE)
        self.motion.stop_immediate()
        self._set_status("Idle", "gray")
        self._save_timer.start(self._save_debounce_ms)

    def _on_z_joy_speed(self, vz: float):
        if self.homing.is_active() or self.move_to_ctrl.is_active():
            return
        vz_cmd = self._clamp_jog_to_limits("z", float(vz))
        self.motion.set_desired_velocity(float(self.motion.v_des["x"]), float(self.motion.v_des["y"]), vz_cmd, mode=MotionMode.JOYSTICK)
        if abs(vz_cmd) < 1e-6:
            self._set_status("Idle", "gray")
            self._save_timer.start(self._save_debounce_ms)
        else:
            self._set_status("Jogging…", "orange")

    def _on_z_joy_stop(self):
        self.motion.set_desired_velocity(float(self.motion.v_des["x"]), float(self.motion.v_des["y"]), 0.0, mode=MotionMode.IDLE)
        self.motion.stop_immediate()
        self._set_status("Idle", "gray")
        self._save_timer.start(self._save_debounce_ms)

    def _clamp_jog_to_limits(self, axis: str, v_cmd: float) -> float:
        cur_fs  = float(getattr(self.MOTORS, f"{axis}_current_position_full_step", 0.0))
        min_fs  = float(getattr(self.MOTORS, f"{axis}_full_step_min", 0.0))
        max_fs  = float(getattr(self.MOTORS, f"{axis}_full_step_max", 0.0))
        eps = 1e-6

        if hasattr(self, 'homing') and self.homing.is_active():
            self.homing.stop('Homing stopped by user.')
            return
        if hasattr(self, 'move_to_ctrl') and self.move_to_ctrl.is_active():
            self.move_to_ctrl.stop('Move stopped by user.')

        if getattr(self, "_homing_active", False):
            return v_cmd

        if cur_fs <= (min_fs + eps) and v_cmd < 0: return 0.0
        if cur_fs >= (max_fs - eps) and v_cmd > 0: return 0.0
        return v_cmd

    def _preempt_all_motion(self):
        if hasattr(self, 'homing') and self.homing.is_active():
            self.homing.stop('Preempted by other motion')
        if hasattr(self, 'move_to_ctrl') and self.move_to_ctrl.is_active():
            self.move_to_ctrl.stop('Preempted by other motion')

        if getattr(self, "_homing_active", False):
            try: self._finish_homing(success=False, resason="Preempted")
            except: pass

        try:
            if getattr(self, "_move_worker", None) is not None:
                self._move_worker.request_stop()
        except Exception: pass

        try: self.mc.halt_all_speed()
        except: pass
        try: self.mc.stop_jog_loop()
        except: pass

    def _on_move_to(self):
        if self.homing.is_active() or self.move_to_ctrl.is_active():
            QMessageBox.information(self, "Busy", "Motors are already moving.")
            return
        self._preempt_all_motion()
        tx = self._micro_to_full("x", self.x_step.value())
        ty = self._micro_to_full("y", self.y_step.value())
        tz = self._micro_to_full("z", self.z_step.value())
        self._all_btn_enabled(False)
        self.move_to_ctrl.start(tx, ty, tz)

    def _on_auto_focus_clicked(self):
        """Emit signal so the parent screen (which has camera access) can run autofocus."""
        if self.homing.is_active() or self.move_to_ctrl.is_active():
            QMessageBox.information(self, "Busy", "Motors are already moving.")
            return
        self._preempt_all_motion()
        self._all_btn_enabled(False)
        self._autofocus_active = True
        self.auto_focus_requested.emit()

    def _on_calibrate_clicked(self):
        """Toggle calibration on/off."""
        if self._calibration_active:
            # Stop it
            self.calibrate_stop_requested.emit()
            return

        if self.homing.is_active() or self.move_to_ctrl.is_active():
            QMessageBox.information(self, "Busy", "Motors are already moving.")
            return
        self._preempt_all_motion()
        self._all_btn_enabled(False)
        self._calibration_active = True
        self.calibrate_btn.setText("⏹ Stop Calibration")
        self.calibrate_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        self.calibrate_btn.setEnabled(True)  # Keep stop button enabled
        self.calibrate_requested.emit()

    def _on_move_to_feature_clicked(self):
        """Prompt for target feature (col, row) and emit signal."""
        if self.homing.is_active() or self.move_to_ctrl.is_active():
            QMessageBox.information(self, "Busy", "Motors are already moving.")
            return

        from PyQt6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(
            self, "Move to Feature",
            "Enter target (col, row):\ne.g. 15, 12")
        if not ok or not text.strip():
            return
        try:
            parts = [p.strip() for p in text.split(",")]
            col, row = int(parts[0]), int(parts[1])
            if not (0 <= col <= 31 and 0 <= row <= 31):
                raise ValueError("col and row must be 0–31")
        except (ValueError, IndexError) as e:
            QMessageBox.warning(self, "Invalid Input",
                                f"Enter two integers 0–31 separated by comma.\n{e}")
            return

        self._preempt_all_motion()
        self._all_btn_enabled(False)
        self._calibration_active = True
        self.move_to_feature_requested.emit(col, row)

    def _on_spiral_scan_clicked(self):
        """Toggle spiral scan on/off."""
        if self._spiral_scan_active:
            # Stop it
            self.spiral_scan_stop_requested.emit()
            return

        if self.homing.is_active() or self.move_to_ctrl.is_active():
            QMessageBox.information(self, "Busy", "Motors are already moving.")
            return

        self._preempt_all_motion()
        self._all_btn_enabled(False)
        self._spiral_scan_active = True
        self.spiral_scan_btn.setText("⏹ Stop Scan")
        self.spiral_scan_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        self.spiral_scan_btn.setEnabled(True)  # Keep stop button enabled
        # Send scan settings before starting
        interval = int(self.scan_interval_combo.currentText())
        center_col = self.scan_center_col_spin.value()
        center_row = self.scan_center_row_spin.value()
        scan_size = int(self.scan_size_combo.currentText())
        num_cycles = self.scan_cycles_spin.value()
        self.spiral_scan_settings_changed.emit(
            interval, center_col, center_row, scan_size, num_cycles)
        self.spiral_scan_requested.emit()

    def get_spiral_scan_settings(self) -> dict:
        """Snapshot of current scan settings — used by the planner/recipe step."""
        return {
            'interval_min': int(self.scan_interval_combo.currentText()),
            'center_col': self.scan_center_col_spin.value(),
            'center_row': self.scan_center_row_spin.value(),
            'scan_size': int(self.scan_size_combo.currentText()),
            'num_cycles': self.scan_cycles_spin.value(),
        }

    def end_spiral_scan(self, success: bool, message: str):
        """Called by parent screen when spiral scan finishes or is stopped."""
        self._spiral_scan_active = False
        self._all_btn_enabled(True)
        self._save_timer.start(self._save_debounce_ms)
        self.spiral_scan_btn.setText("▶ Spiral Scan")
        self.spiral_scan_btn.setStyleSheet("background-color: #1565C0; color: white; font-weight: bold;")
        if success:
            self._set_status("Scan done", "#4caf50")
        else:
            self._set_status("Scan: " + message, "orange")

    def end_calibration(self, success: bool, message: str):
        """Called by parent screen when calibration or move-to-feature finishes."""
        self._calibration_active = False
        self._all_btn_enabled(True)
        self._save_timer.start(self._save_debounce_ms)
        self.calibrate_btn.setText("Calibrate Stage")
        self.calibrate_btn.setStyleSheet("")
        if success:
            self._set_status("Cal done", "#4caf50")
        else:
            self._set_status("Cal: " + message, "orange")

    def _on_move_error(self, msg: str):
        try: self.mc.stop_all()
        except: pass
        QMessageBox.critical(self, "Motion Error", msg)

    def _on_move_finished(self):
        if self._move_thread:
            self._move_thread.quit()
            self._move_thread.wait()
        self._all_btn_enabled(True)
        try: self._save_timer.start(self._save_debounce_ms)
        except: pass
        self._set_status("Idle", "gray")
        self._move_thread = None
        self._move_worker = None

    def _on_home_stage(self):
        self._preempt_all_motion()
        self._all_btn_enabled(False)
        self._homing_active = True
        self.homing.start()

    def _on_stop(self):
        self._preempt_all_motion()
        self._autofocus_active = False
        if self._calibration_active:
            self.calibrate_stop_requested.emit()
        self._calibration_active = False
        if self._spiral_scan_active:
            self.spiral_scan_stop_requested.emit()
        try: self.mc.stop_all()
        except: pass
        self._set_status("Idle", "gray")
        self._all_btn_enabled(True)
        self._save_timer.start(self._save_debounce_ms)
        self.calibrate_btn.setText("Calibrate Stage")
        self.calibrate_btn.setStyleSheet("")

    def end_autofocus(self, success: bool, message: str):
        """Called by parent screen when autofocus finishes."""
        self._autofocus_active = False
        self._all_btn_enabled(True)
        self._save_timer.start(self._save_debounce_ms)
        if success:
            self._set_status("AF done", "#4caf50")
        else:
            self._set_status("AF: " + message, "orange")

    def _on_step_update(self, axis: str, logical_micro: int):
        axis = axis.lower()
        if axis in self._freeze_step_updates:
            return
        mres = self._mres(axis) or 1
        fs_abs = float(logical_micro) / float(mres)
        setattr(self.MOTORS, f"{axis}_current_position_full_step", fs_abs)
        ms_abs = self._full_to_micro(axis, fs_abs)
        bar  = getattr(self, f"{axis}_bar")
        step = getattr(self, f"{axis}_step")
        bar.blockSignals(True); step.blockSignals(True)
        bar.setValue(ms_abs);   step.setValue(ms_abs)
        bar.blockSignals(False); step.blockSignals(False)
        self._update_all_position_labels(axis)

    def _on_axis_homed_freeze(self, axis: str):
        axis = axis.lower()
        self._freeze_step_updates.add(axis)
        setattr(self.MOTORS, f"{axis}_current_position_full_step", 0.0)
        self._update_all_position_labels(axis)

    def _on_homing_done(self, success: bool, message: str):
        self._all_btn_enabled(True)
        self._save_timer.start(self._save_debounce_ms)
        self._homing_active = False
        QTimer.singleShot(150, self._freeze_step_updates.clear)
        # Sync UI with restored profile
        self._sync_speed_ui_from_mc()
        # Force positions to true zero after all motion has settled
        if success:
            QTimer.singleShot(200, self._reset_all_home_positions)
        if (not success) and message:
            QMessageBox.warning(self, "Homing", message)

    def _on_z_home_cleanup(self, success, message):
        """Called when z-only homing finishes (from MoveTo z-homing phase)."""
        QTimer.singleShot(150, self._freeze_step_updates.clear)
        if success:
            QTimer.singleShot(200, lambda: self._reset_home_position("z"))

    def _reset_all_home_positions(self):
        """Re-zero all axes after full homing has settled."""
        for axis in ("x", "y", "z"):
            self._reset_home_position(axis)

    def _reset_home_position(self, axis):
        """Re-zero a single axis position after homing has settled."""
        self.mc.set_logical_position_micro(axis, 0)
        setattr(self.MOTORS, f"{axis}_current_position_full_step", 0.0)
        self._update_all_position_labels(axis)

    def _on_move_to_done(self, success: bool, message: str):
        if self._autofocus_active:
            return  # AutoFocusController handles moves during autofocus
        if self._spiral_scan_active:
            return  # SpiralScanController handles moves during spiral scan
        self._all_btn_enabled(True)
        self._save_timer.start(self._save_debounce_ms)
        # Sync UI with restored profile
        self._sync_speed_ui_from_mc()
        if (not success) and message:
            QMessageBox.warning(self, "Move", message)

    def _set_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color};")

    def _flush_config_to_disk(self):
        try:
            save_config_atomic(self.config, self.config_path)
            print(f"[StepperUI] successful")
        except Exception as e:
            print(f"[StepperUI] failed: {e}")

    def _all_btn_enabled(self, state: bool):
        self.home_btn.setEnabled(state)
        self.move_to_btn.setEnabled(state)
        self.auto_focus_btn.setEnabled(state)
        self.calibrate_btn.setEnabled(state)
        self.move_feature_btn.setEnabled(state)
        self.spiral_scan_btn.setEnabled(state)
        self.speed_multiplier_spin.setEnabled(state)
        self.xy_joysticks.setEnabled(state)
        self.z_joystick.setEnabled(state)
        self.x_bar.setEnabled(state)
        self.y_bar.setEnabled(state)
        self.z_bar.setEnabled(state)
        self.x_step.setEnabled(state)
        self.y_step.setEnabled(state)
        self.z_step.setEnabled(state)

    def _mres(self, axis: str) -> int:
        return int(self.mc.get_axis_microstepping(axis))

    def _micro_to_full(self, axis: str, micro: int | float) -> float:
        return float(micro) / max(1, self._mres(axis))

    def _full_to_micro(self, axis: str, full: float) -> int:
        return int(round(float(full) * max(1, self._mres(axis))))