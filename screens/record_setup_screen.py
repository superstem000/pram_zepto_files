"""
Record Setup Screen — configure recording parameters before starting.

Mirrors FreeModeScreen's camera/calibration wiring so the move planner
can run orientation calibration during a microfluidics dwell and drive
the shared SpiralScanController for spiral_scan recipe steps.

Layout:
  - LEFT panel:  live camera preview.
  - RIGHT panel: stepper widget + move planner + camera controls +
                 Start Recording button.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QMessageBox, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from widgets.move_planner_widget import MovePlannerWidget
from widgets.camera_controls_widget import CameraControlsWidget
from widgets.camera_preview_widget import CameraPreviewWidget
from vision.auto_focus import AutoFocusController
from calibration.stage_calibration import StageCalibration
from calibration.orient_calibration import OrientCalibrationController
from calibration.spiral_scan import SpiralScanController


class RecordSetupScreen(QWidget):
    """Record mode setup screen with live preview + recipe planner."""

    back_requested = pyqtSignal()
    start_recording_requested = pyqtSignal()

    def __init__(self, config, mc, shared_stepper_widget, parent=None):
        super().__init__(parent)
        self.config = config
        self.mc = mc
        self.shared_stepper_widget = shared_stepper_widget

        # Canonical preview state lives in self.camera_preview.is_running();
        # don't shadow it with a screen-local flag (it drifts on screen-switch
        # and on any stop_preview call we don't make ourselves).
        self._build_ui()
        self._init_autofocus()
        self._init_calibration()
        self._init_spiral_scan()
        self._wire_planner_to_controllers()

    # ──────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        """Build the record setup UI: live preview on left, controls on right."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── LEFT PANEL (Camera Feed) ──
        self.left_panel = QWidget()
        self.left_panel.setStyleSheet("background-color: black;")

        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)

        self.camera_preview = CameraPreviewWidget(self.config, parent=self)
        left_layout.addWidget(self.camera_preview, stretch=1)

        # ── RIGHT PANEL (Controls) ──
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                border-left: 3px solid #555;
                background-color: transparent;
            }
        """)
        self.scroll_area.setFixedWidth(780)

        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background-color: transparent;")

        self.controls_layout = QVBoxLayout(self.scroll_content)
        self.controls_layout.setContentsMargins(10, 10, 10, 10)
        self.controls_layout.setSpacing(25)

        # Top row: back + screen title
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 10)

        self.back_btn = QPushButton("← Back to Home")
        self.back_btn.setMinimumWidth(200)
        self.back_btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.back_btn.clicked.connect(self._on_back)
        top_layout.addWidget(self.back_btn)

        title = QLabel("🔴 Record Setup")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #f44336; margin-left: 20px;")
        top_layout.addWidget(title)
        top_layout.addStretch()

        self.controls_layout.addWidget(top_row)

        # The shared StepperMotorsWidget is re-parented here in on_enter,
        # matching FreeModeScreen.
        self.stepper_placeholder = QWidget()
        self.stepper_placeholder_layout = QVBoxLayout(self.stepper_placeholder)
        self.stepper_placeholder_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_layout.addWidget(self.stepper_placeholder)

        # Move planner (mode='full' enables the Run Recipe / Stop buttons)
        self._move_planner = MovePlannerWidget(
            self.controls_layout,
            self.config,
            self.shared_stepper_widget,
            mode='full'
        )

        # Camera controls — use 'free' mode so we get Show Live / Apply
        # Settings / Capture / LED buttons (record_setup mode is too stripped
        # down for what this screen now needs).
        self._camera_controls = CameraControlsWidget(
            self.controls_layout,
            self.config,
            mode='free',
            parent=self
        )
        self._camera_controls.live_toggled.connect(self._on_live_toggle)
        self._camera_controls.capture_requested.connect(self._on_capture)
        self._camera_controls.settings_changed.connect(self._on_apply_settings)

        # Start Recording button at the bottom of the right panel
        self.start_btn = QPushButton("▶  Start Recording")
        self.start_btn.setFixedHeight(60)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #c62828;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 8px;
                border: 2px solid #b71c1c;
            }
            QPushButton:hover { background-color: #d32f2f; }
            QPushButton:pressed { background-color: #b71c1c; }
        """)
        self.start_btn.clicked.connect(self._on_start_recording)
        self.controls_layout.addWidget(self.start_btn)

        self.controls_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_content)

        main_layout.addWidget(self.left_panel, stretch=4)
        main_layout.addWidget(self.scroll_area, stretch=0)

    # ──────────────────────────────────────────────────────────────
    # Live preview / capture
    # ──────────────────────────────────────────────────────────────
    def _on_live_toggle(self, start: bool):
        if start:
            self._start_live()
        else:
            self._stop_live()

    def _start_live(self):
        try:
            self.camera_preview.start_preview(self._camera_controls.get_camera_settings())
            self._camera_controls.set_live_state(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start live preview:\n{e}")
            self._camera_controls.set_live_state(False)

    def _stop_live(self):
        try:
            self.camera_preview.stop_preview()
        except Exception as e:
            print(f"RecordSetupScreen: Error stopping preview: {e}")
        self._camera_controls.set_live_state(False)

    def _on_capture(self):
        if not self.camera_preview.is_running():
            QMessageBox.warning(self, "Warning", "Start live preview before capturing.")
            return
        try:
            output_path = self._camera_controls.get_output_path()
            if not output_path:
                return
            settings = self._camera_controls.get_camera_settings()
            self.camera_preview.capture_image(output_path, settings)
            self._camera_controls.set_status(f"Saved: {output_path.split('/')[-1]}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Capture failed:\n{e}")

    def _on_apply_settings(self, settings: dict):
        try:
            self.camera_preview.apply_settings(settings)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply settings:\n{e}")

    # ──────────────────────────────────────────────────────────────
    # Autofocus  (mirrors FreeModeScreen)
    # ──────────────────────────────────────────────────────────────
    def _init_autofocus(self):
        sw = self.shared_stepper_widget
        self._af_controller = AutoFocusController(
            move_to_ctrl=sw.move_to_ctrl,
            camera_preview=self.camera_preview,
            mc=self.mc,
            motors_config=sw.MOTORS,
            config=self.config,
            parent=self
        )
        self._af_controller.progress.connect(self._on_af_progress)
        self._af_controller.finished.connect(self._on_af_finished)

        sw.auto_focus_requested.connect(self._on_af_requested)

    def _on_af_requested(self):
        if not self.isVisible():
            return
        if not self.camera_preview.is_running():
            QMessageBox.warning(self, "Warning",
                                "Start live preview before autofocusing.")
            self.shared_stepper_widget.end_autofocus(False, "No live preview")
            return
        settings = self._camera_controls.get_camera_settings()
        self._af_controller.start(settings)

    def _on_af_progress(self, phase: str, step: int, total: int,
                        z_fs: float, score: float):
        self.shared_stepper_widget._set_status(
            f"AF {phase}: {step}/{total}  Z={z_fs:.1f}  score={score:.0f}",
            "#2196F3"
        )

    def _on_af_finished(self, success: bool, message: str):
        pending = getattr(self, '_pending_cal_message', None)
        if pending:
            self._pending_cal_message = None
            full_msg = f"{pending} | Refocus: {message}" if success else pending
            self.shared_stepper_widget.end_calibration(True, full_msg)
            self._camera_controls.set_status(full_msg)
            return

        self.shared_stepper_widget.end_autofocus(success, message)
        if success:
            self._camera_controls.set_status(f"Autofocus: {message}")
        else:
            if "Cancelled" not in message:
                QMessageBox.warning(self, "Autofocus", message)

    # ──────────────────────────────────────────────────────────────
    # Stage Calibration (mirrors FreeModeScreen)
    # ──────────────────────────────────────────────────────────────
    def _init_calibration(self):
        sw = self.shared_stepper_widget

        self._orient_cal = OrientCalibrationController(
            move_to_ctrl=sw.move_to_ctrl,
            camera_preview=self.camera_preview,
            mc=self.mc,
            motors_model=sw.MOTORS,
            config=self.config,
            af_controller=self._af_controller,
            homing=sw.homing,
            parent=self
        )
        self._orient_cal.progress.connect(self._on_cal_progress)
        self._orient_cal.finished.connect(self._on_orient_cal_finished)

        # StageCalibration handles the move_to_feature detection-preview flow.
        self._calibration = StageCalibration(
            move_to_ctrl=sw.move_to_ctrl,
            camera_preview=self.camera_preview,
            mc=self.mc,
            motors_config=sw.MOTORS,
            config=self.config,
            af_controller=self._af_controller,
            parent=self
        )
        self._calibration.progress.connect(self._on_cal_progress)
        self._calibration.finished.connect(self._on_cal_finished)
        self._calibration.detection_preview.connect(self._on_detection_preview)

        sw.calibrate_requested.connect(self._on_cal_requested)
        sw.calibrate_stop_requested.connect(self._on_cal_stop_requested)
        sw.move_to_feature_requested.connect(self._on_move_to_feature_requested)

    def _on_cal_requested(self):
        """Calibrate Stage button — same flow as FreeModeScreen."""
        if not self.isVisible():
            return
        if not self.camera_preview.is_running():
            QMessageBox.warning(self, "Warning",
                                "Start live preview before calibrating.")
            self.shared_stepper_widget.end_calibration(False, "No live preview")
            return

        reply = QMessageBox.question(
            self, "Confirm Calibration",
            "This will detect plate orientation and re-calibrate the stage.\n"
            "The stage will move to the safe region center automatically.\n\n"
            "Proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            self.shared_stepper_widget.end_calibration(False, "Cancelled")
            return

        settings = self._camera_controls.get_camera_settings()
        self._orient_cal.start(settings)

    def _on_cal_stop_requested(self):
        if not self.isVisible():
            return
        if self._orient_cal.is_active():
            self._orient_cal.stop("Stopped by user")

    def _on_move_to_feature_requested(self, col: int, row: int):
        if not self.isVisible():
            return
        if not self.camera_preview.is_running():
            QMessageBox.warning(self, "Warning",
                                "Start live preview before moving to feature.")
            self.shared_stepper_widget.end_calibration(False, "No live preview")
            return
        if self._orient_cal.get_result().valid:
            self._calibration.result = self._orient_cal.get_result()
        if not self._calibration.get_result().valid:
            QMessageBox.warning(self, "Not Calibrated",
                                "Run 'Calibrate Stage' first.")
            self.shared_stepper_widget.end_calibration(False, "Not calibrated")
            return
        settings = self._camera_controls.get_camera_settings()
        self._calibration.move_to_feature(col, row, settings)

    def _on_orient_cal_finished(self, success: bool, message: str, result):
        if success and result is not None:
            self._calibration.result = result
        self.shared_stepper_widget.end_calibration(success, message)
        if success:
            self._camera_controls.set_status(message)
        else:
            if "Cancelled" not in message:
                QMessageBox.warning(self, "Calibration", message)

    def _on_detection_preview(self, info_str: str, markers):
        reply = QMessageBox.question(
            self, "Detection Results — Confirm Move?",
            info_str + "\n\nProceed with move?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._calibration.confirm_move()
        else:
            self._calibration.cancel_move()

    def _on_cal_progress(self, phase: str, message: str):
        self.shared_stepper_widget._set_status(f"Cal: {message}", "#FF9800")

    def _on_cal_finished(self, success: bool, message: str, result):
        if success and "Arrived near" in message:
            self._pending_cal_message = message
            self.shared_stepper_widget._set_status("Quick refocus...", "#2196F3")
            settings = self._camera_controls.get_camera_settings()
            self._af_controller.start_quick(settings)
            return

        self.shared_stepper_widget.end_calibration(success, message)
        if success:
            self._camera_controls.set_status(message)
        else:
            if "Cancelled" not in message and "cancelled" not in message:
                QMessageBox.warning(self, "Calibration", message)

    # ──────────────────────────────────────────────────────────────
    # Spiral Scan (mirrors FreeModeScreen)
    # ──────────────────────────────────────────────────────────────
    def _init_spiral_scan(self):
        sw = self.shared_stepper_widget
        self._spiral_scan = SpiralScanController(
            calibration=self._orient_cal,
            af_controller=self._af_controller,
            move_to_ctrl=sw.move_to_ctrl,
            camera_preview=self.camera_preview,
            mc=self.mc,
            motors_config=sw.MOTORS,
            config=self.config,
            parent=self
        )
        self._spiral_scan.progress.connect(self._on_spiral_progress)
        self._spiral_scan.finished.connect(self._on_spiral_finished)

        sw.spiral_scan_requested.connect(self._on_spiral_scan_requested)
        sw.spiral_scan_stop_requested.connect(self._on_spiral_scan_stop)
        sw.spiral_scan_settings_changed.connect(self._on_spiral_scan_settings)

    def _on_spiral_scan_requested(self):
        if not self.isVisible():
            return
        if not self.camera_preview.is_running():
            QMessageBox.warning(self, "Warning",
                                "Start live preview before scanning.")
            self.shared_stepper_widget.end_spiral_scan(False, "No live preview")
            return
        if not self._orient_cal.get_result().valid:
            QMessageBox.warning(self, "Not Calibrated",
                                "Run 'Calibrate Stage' first.")
            self.shared_stepper_widget.end_spiral_scan(False, "Not calibrated")
            return
        settings = self._camera_controls.get_camera_settings()
        self._spiral_scan.start(settings)

    def _on_spiral_scan_stop(self):
        if not self.isVisible():
            return
        if self._spiral_scan.is_active():
            self._spiral_scan.stop("Stopped by user")

    def _on_spiral_scan_settings(self, interval_min: int, center_col: int,
                                  center_row: int, scan_size: int, num_cycles: int):
        if not self.isVisible():
            return
        self._spiral_scan.configure(
            interval_min=interval_min,
            center_col=center_col,
            center_row=center_row,
            scan_size=scan_size,
            num_cycles=num_cycles,
        )

    def _on_spiral_progress(self, phase: str, message: str):
        self.shared_stepper_widget._set_status(f"Scan: {message}", "#1565C0")

    def _on_spiral_finished(self, success: bool, message: str):
        self.shared_stepper_widget.end_spiral_scan(success, message)
        if not success and "Stopped" not in message:
            self._camera_controls.set_status(f"Scan error: {message}")

    # ──────────────────────────────────────────────────────────────
    # Wire planner ↔ controllers
    #   * orient_cal for calibration-during-dwell
    #   * spiral_scan for spiral_scan recipe steps
    # ──────────────────────────────────────────────────────────────
    def _wire_planner_to_controllers(self):
        """Give the move planner handles on the cal & spiral-scan controllers
        plus a way to fetch current camera settings and current scan settings.
        Also intercept the Run Recipe button so we can pre-flight: live preview
        must be on if the recipe will try to calibrate during a dwell, or if it
        contains a spiral_scan step."""
        self._move_planner.set_orient_cal_controller(self._orient_cal)
        self._move_planner.set_camera_settings_provider(
            self._camera_controls.get_camera_settings
        )
        # Spiral scan wiring — mirror what the stepper widget's Spiral Scan
        # button does: configure with (interval, center_col, center_row) from
        # the stepper widget's scan settings UI, then start.
        self._move_planner.set_spiral_scan_controller(self._spiral_scan)
        self._move_planner.set_spiral_scan_settings_provider(
            self._get_spiral_scan_settings
        )

        # Pre-flight check: if the recipe needs calibration or contains a
        # spiral_scan step, refuse Run unless live preview is on (and stage
        # is calibrated, for spiral_scan). We do this by replacing the
        # existing click handler on btn_execute.
        btn_exec = getattr(self._move_planner, "btn_execute", None)
        if btn_exec is not None:
            try:
                btn_exec.clicked.disconnect()  # remove the planner's own handler
            except Exception:
                pass
            btn_exec.clicked.connect(self._on_run_recipe_clicked)

    def _get_spiral_scan_settings(self) -> dict:
        """Read scan settings from the shared stepper widget's UI. Returns
        {interval_min, center_col, center_row}. Falls back to defaults if
        the widget isn't available."""
        sw = self.shared_stepper_widget
        try:
            interval_min = int(sw.scan_interval_combo.currentText())
            center_col = int(sw.scan_center_col_spin.value())
            center_row = int(sw.scan_center_row_spin.value())
        except Exception:
            interval_min, center_col, center_row = 5, 13, 13
        return {
            "interval_min": interval_min,
            "center_col": center_col,
            "center_row": center_row,
        }

    def _on_run_recipe_clicked(self):
        """Wrapper around MovePlannerWidget._on_execute that enforces:
          - live preview must be on if the recipe needs calibration during a
            dwell or contains spiral_scan steps,
          - calibration must be valid if the recipe contains spiral_scan steps."""
        try:
            needs_cal = self._move_planner.recipe_needs_calibration()
        except Exception:
            needs_cal = False
        try:
            needs_scan = self._move_planner.recipe_needs_spiral_scan()
        except Exception:
            needs_scan = False

        if (needs_cal or needs_scan) and not self.camera_preview.is_running():
            QMessageBox.warning(
                self, "Live Preview Required",
                "This recipe needs the live camera preview "
                "(calibration-during-dwell and/or spiral scan).\n\n"
                "Start the live camera preview first (Show Live), "
                "then press Run Recipe again."
            )
            return

        if needs_scan and not self._orient_cal.get_result().valid:
            QMessageBox.warning(
                self, "Not Calibrated",
                "This recipe contains a Spiral Scan step but the stage "
                "isn't calibrated yet. Run 'Calibrate Stage' first."
            )
            return

        # Otherwise hand off to the planner's own execute flow
        self._move_planner._on_execute()

    # ──────────────────────────────────────────────────────────────
    # Navigation
    # ──────────────────────────────────────────────────────────────
    def _on_back(self):
        # Stop anything in progress, similar to FreeModeScreen
        if hasattr(self, '_af_controller') and self._af_controller.is_active():
            self._af_controller.stop("Navigated away")
        if hasattr(self, '_orient_cal') and self._orient_cal.is_active():
            self._orient_cal.stop("Navigated away")
        if hasattr(self, '_calibration') and self._calibration.is_active():
            self._calibration.stop("Navigated away")
        if hasattr(self, '_spiral_scan') and self._spiral_scan.is_active():
            self._spiral_scan.stop("Navigated away")
        # Stop the planner if it's running a recipe
        if getattr(self._move_planner, "_exec_active", False):
            try:
                self._move_planner._on_stop()
            except Exception:
                pass

        if self.camera_preview.is_running():
            reply = QMessageBox.question(
                self, "Confirm",
                "Live preview is running. Stop and return to home?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._stop_live()

        self.back_requested.emit()

    def _on_start_recording(self):
        settings = self.get_recording_settings()
        if not settings.get('output_directory'):
            QMessageBox.warning(self, "No Output Directory",
                                "Please select a folder to save recordings.")
            return
        # Recording owns the camera, so release it here first
        if self.camera_preview.is_running():
            self._stop_live()
        self.start_recording_requested.emit()

    # ──────────────────────────────────────────────────────────────
    # Public API consumed by AppMainWindow
    # ──────────────────────────────────────────────────────────────
    def get_recording_settings(self) -> dict:
        return self._camera_controls.get_camera_settings()

    def get_recipe_steps(self) -> list:
        return self._move_planner._read_table_steps()

    def on_enter(self):
        if self.shared_stepper_widget and self.shared_stepper_widget.motor_group:
            self.stepper_placeholder_layout.addWidget(self.shared_stepper_widget.motor_group)
        # Sync the camera-control toggle UI to actual camera state — the
        # preview may have been started or stopped on another screen.
        try:
            self._camera_controls.set_live_state(self.camera_preview.is_running())
        except Exception:
            pass

    def on_leave(self):
        # Mirror FreeModeScreen.on_leave so the camera is fully released
        if hasattr(self, '_af_controller') and self._af_controller.is_active():
            self._af_controller.stop("Left screen")
        if hasattr(self, '_orient_cal') and self._orient_cal.is_active():
            self._orient_cal.stop("Left screen")
        if hasattr(self, '_calibration') and self._calibration.is_active():
            self._calibration.stop("Left screen")
        if hasattr(self, '_spiral_scan') and self._spiral_scan.is_active():
            self._spiral_scan.stop("Left screen")
        if getattr(self._move_planner, "_exec_active", False):
            try:
                self._move_planner._on_stop()
            except Exception:
                pass
        self._stop_live()
        if self.shared_stepper_widget and self.shared_stepper_widget.motor_group:
            self.stepper_placeholder_layout.removeWidget(self.shared_stepper_widget.motor_group)

    def close(self):
        self._stop_live()
        if hasattr(self, 'camera_preview'):
            self.camera_preview.close()
        super().close()