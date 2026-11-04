"""
Free Mode Screen - Manual control with live preview and single capture.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QGridLayout, QSizePolicy, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

from widgets.camera_controls_widget import CameraControlsWidget
from widgets.camera_preview_widget import CameraPreviewWidget
from vision.auto_focus import AutoFocusController
from calibration.stage_calibration import StageCalibration
from calibration.orient_calibration import OrientCalibrationController
from calibration.spiral_scan import SpiralScanController


class FreeModeScreen(QWidget):
    """Free mode screen with motors and single capture."""

    back_requested = pyqtSignal()

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

    def _build_ui(self):
        """Build the free mode UI."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # LEFT PANEL (Camera Feed Area)
        self.left_panel = QWidget()
        self.left_panel.setStyleSheet("background-color: black;")

        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)

        self.camera_preview = CameraPreviewWidget(self.config, parent=self)
        left_layout.addWidget(self.camera_preview, stretch=1)

        # RIGHT PANEL (Controls)
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

        # Back button
        back_row = QWidget()
        back_layout = QHBoxLayout(back_row)
        back_layout.setContentsMargins(0, 0, 0, 10)

        self.back_btn = QPushButton("← Back to Home")
        self.back_btn.setMinimumWidth(200)
        self.back_btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.back_btn.clicked.connect(self._on_back)
        back_layout.addWidget(self.back_btn)
        back_layout.addStretch()

        self.controls_layout.addWidget(back_row)

        self.stepper_placeholder = QWidget()
        self.stepper_placeholder_layout = QVBoxLayout(self.stepper_placeholder)
        self.stepper_placeholder_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_layout.addWidget(self.stepper_placeholder)

        self._camera_controls = CameraControlsWidget(
            self.controls_layout,
            self.config,
            mode='free',
            parent=self
        )

        self._camera_controls.live_toggled.connect(self._on_live_toggle)
        self._camera_controls.capture_requested.connect(self._on_capture)
        self._camera_controls.settings_changed.connect(self._on_apply_settings)

        self.controls_layout.addStretch()

        self.scroll_area.setWidget(self.scroll_content)

        main_layout.addWidget(self.left_panel, stretch=4)
        main_layout.addWidget(self.scroll_area, stretch=0)

    def _on_back(self):
        if hasattr(self, '_af_controller') and self._af_controller.is_active():
            self._af_controller.stop("Navigated away")
        if hasattr(self, '_orient_cal') and self._orient_cal.is_active():
            self._orient_cal.stop("Navigated away")
        if hasattr(self, '_calibration') and self._calibration.is_active():
            self._calibration.stop("Navigated away")
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
            print(f"FreeModeScreen: Error stopping preview: {e}")

        self._camera_controls.set_live_state(False)

    def _on_capture(self):
        # The shared stepper widget broadcasts button signals to BOTH screens
        # (free_mode + record_setup) regardless of which is on top. Guard
        # against this handler firing while the other screen is visible —
        # otherwise our camera_preview (idle) raises a spurious "Start
        # live preview" warning even though the user is operating on the
        # other screen where preview is on.
        if not self.isVisible():
            return
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
        """Apply updated camera settings to the live preview."""
        try:
            self.camera_preview.apply_settings(settings)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply settings:\n{e}")

    # ── Autofocus ──

    def _init_autofocus(self):
        """Set up the autofocus controller and connect signals."""
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
        """Handle Auto Focus button press."""
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
        """Update status during autofocus sweep."""
        self.shared_stepper_widget._set_status(
            f"AF {phase}: {step}/{total}  Z={z_fs:.1f}  score={score:.0f}",
            "#2196F3"
        )

    def _on_af_finished(self, success: bool, message: str):
        """Handle autofocus completion."""
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

    # ── Stage Calibration ──

    def _init_calibration(self):
        """Set up calibration controllers and connect signals.

        OrientCalibrationController: used for full calibration (orientation + cal)
        StageCalibration: kept for move_to_feature (detection preview dialog)
        """
        sw = self.shared_stepper_widget

        # New combined orientation + calibration controller
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
        """Handle Calibrate Stage button press — runs orientation + calibration."""
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
        """Handle Calibrate Stage stop request."""
        if not self.isVisible():
            return
        if self._orient_cal.is_active():
            self._orient_cal.stop("Stopped by user")

    def _on_move_to_feature_requested(self, col: int, row: int):
        """Handle Move to Feature request — triggers detection preview first."""
        if not self.isVisible():
            return
        if not self.camera_preview.is_running():
            QMessageBox.warning(self, "Warning",
                                "Start live preview before moving to feature.")
            self.shared_stepper_widget.end_calibration(False, "No live preview")
            return
        # Sync calibration result from orient_cal if available
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
        """Handle combined orientation + calibration completion."""
        if success and result is not None:
            # Mirror the result into StageCalibration so move_to_feature uses it.
            self._calibration.result = result
        self.shared_stepper_widget.end_calibration(success, message)
        if success:
            self._camera_controls.set_status(message)
        else:
            if "Cancelled" not in message:
                QMessageBox.warning(self, "Calibration", message)

    def _on_detection_preview(self, info_str: str, markers):
        """Show detection results and ask user to confirm or cancel the move."""
        print("[FreeModeScreen] _on_detection_preview slot reached!")
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
        """Update status during calibration."""
        self.shared_stepper_widget._set_status(
            f"Cal: {message}", "#FF9800"
        )

    def _on_cal_finished(self, success: bool, message: str, result):
        """Handle calibration or move-to-feature completion."""
        if success and "Arrived near" in message:
            # Run quick autofocus after move-to-feature landing
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

    # ── Spiral Scan ──

    def _init_spiral_scan(self):
        """Set up the spiral scan controller and connect signals."""
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
        """Handle Spiral Scan button press."""
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
        """Handle Spiral Scan stop request."""
        if not self.isVisible():
            return
        if self._spiral_scan.is_active():
            self._spiral_scan.stop("Stopped by user")

    def _on_spiral_scan_settings(self, interval_min: int, center_col: int,
                                  center_row: int, scan_size: int, num_cycles: int):
        """Apply scan settings before scan starts."""
        self._spiral_scan.configure(
            interval_min=interval_min,
            center_col=center_col,
            center_row=center_row,
            scan_size=scan_size,
            num_cycles=num_cycles,
        )

    def _on_spiral_progress(self, phase: str, message: str):
        """Update status during spiral scan."""
        self.shared_stepper_widget._set_status(
            f"Scan: {message}", "#1565C0"
        )

    def _on_spiral_finished(self, success: bool, message: str):
        """Handle spiral scan completion."""
        self.shared_stepper_widget.end_spiral_scan(success, message)
        if not success and "Stopped" not in message:
            self._camera_controls.set_status(f"Scan error: {message}")

    def on_enter(self):
        if self.shared_stepper_widget and self.shared_stepper_widget.motor_group:
            self.stepper_placeholder_layout.addWidget(self.shared_stepper_widget.motor_group)
        # Sync the camera-control toggle UI to whatever the camera is actually
        # doing right now (the preview may have been started/stopped on a
        # different screen before we landed here).
        try:
            self._camera_controls.set_live_state(self.camera_preview.is_running())
        except Exception:
            pass

    def on_leave(self):
        if hasattr(self, '_af_controller') and self._af_controller.is_active():
            self._af_controller.stop("Left screen")
        if hasattr(self, '_orient_cal') and self._orient_cal.is_active():
            self._orient_cal.stop("Left screen")
        if hasattr(self, '_calibration') and self._calibration.is_active():
            self._calibration.stop("Left screen")
        if hasattr(self, '_spiral_scan') and self._spiral_scan.is_active():
            self._spiral_scan.stop("Left screen")
        self._stop_live()
        if self.shared_stepper_widget and self.shared_stepper_widget.motor_group:
            self.stepper_placeholder_layout.removeWidget(self.shared_stepper_widget.motor_group)

    def close(self):
        self._stop_live()
        if hasattr(self, 'camera_preview'):
            self.camera_preview.close()
        super().close()