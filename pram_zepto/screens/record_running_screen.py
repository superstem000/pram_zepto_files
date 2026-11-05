"""
Record Running Screen - Active recording with live preview, model output, and graph.
"""

import os
import time
import random
import numpy as np
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGridLayout, QSizePolicy, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont

# Matplotlib for the graph
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import RPi.GPIO as GPIO

from widgets.camera_preview_widget import CameraPreviewWidget


class RecordRunningScreen(QWidget):
    """Record running screen with preview, model, and graph."""

    stop_requested = pyqtSignal()
    back_requested = pyqtSignal()

    # Capture interval in milliseconds (10 seconds)
    CAPTURE_INTERVAL_MS = 10000

    def __init__(self, config, mc, parent=None):
        super().__init__(parent)
        self.config = config
        self.mc = mc
        self.THEME = getattr(config, "theme", {})

        # Recording state
        self._is_recording = False
        self._recording_settings = {}
        self._capture_count = 0

        # Graph data
        self._graph_data_time = []
        self._graph_data_counts = []

        # LED setup
        self.led_pin = 20
        self._led_initialized = False

        # Timer for periodic captures (every 10 seconds)
        self._capture_timer = QTimer(self)
        self._capture_timer.setInterval(self.CAPTURE_INTERVAL_MS)
        self._capture_timer.timeout.connect(self._on_capture_timer)

        self._build_ui()

    def _build_ui(self):
        """Build the record running UI."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # LEFT PANEL - Preview, Model Output, Graph
        self.left_panel = QFrame()
        self.left_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {self.THEME.get('bg_color', '#1a1a1a')};
            }}
        """)
        left_layout = QGridLayout(self.left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # Row 0: Live Camera Preview (spans both columns)
        self.camera_preview = CameraPreviewWidget(self.config, parent=self)
        self.camera_preview.setMinimumHeight(300)
        left_layout.addWidget(self.camera_preview, 0, 0, 1, 2)

        # Row 1, Col 0: Model Output
        model_frame = QFrame()
        model_frame.setStyleSheet("""
            QFrame {
                background-color: #222;
                border: 2px solid #444;
                border-radius: 8px;
            }
        """)
        model_layout = QVBoxLayout(model_frame)
        model_layout.setContentsMargins(10, 10, 10, 10)

        model_title = QLabel("Model Output")
        model_title.setStyleSheet("color: #888; font-weight: bold; border: none;")
        model_layout.addWidget(model_title)

        self.model_preview = QLabel("Waiting for capture...")
        self.model_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_preview.setStyleSheet("color: #666; border: none;")
        self.model_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.model_preview.setMinimumSize(200, 150)
        model_layout.addWidget(self.model_preview, stretch=1)

        left_layout.addWidget(model_frame, 1, 0)

        # Row 1, Col 1: Detection Graph
        graph_frame = QFrame()
        graph_frame.setStyleSheet("""
            QFrame {
                background-color: #222;
                border: 2px solid #444;
                border-radius: 8px;
            }
        """)
        graph_layout = QVBoxLayout(graph_frame)
        graph_layout.setContentsMargins(10, 10, 10, 10)

        graph_title = QLabel("Detection Graph")
        graph_title.setStyleSheet("color: #888; font-weight: bold; border: none;")
        graph_layout.addWidget(graph_title)

        # Matplotlib figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.figure.patch.set_facecolor('#222222')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._setup_graph_style()
        graph_layout.addWidget(self.canvas, stretch=1)

        left_layout.addWidget(graph_frame, 1, 1)

        # Row stretches
        left_layout.setRowStretch(0, 2)
        left_layout.setRowStretch(1, 1)
        left_layout.setColumnStretch(0, 1)
        left_layout.setColumnStretch(1, 1)

        # RIGHT PANEL - Minimal Controls
        self.right_panel = QFrame()
        self.right_panel.setFixedWidth(280) # Slightly wider for tablet
        self.right_panel.setStyleSheet("""
            QFrame {
                border: none;
                border-left: 3px solid #555;
            }
        """)
        self.right_panel.setFixedWidth(350)

        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(20, 40, 20, 40)
        right_layout.setSpacing(20)

        # Title
        title = QLabel("Recording")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #f44336; border: none;")
        right_layout.addWidget(title)

        # Status
        self.status_label = QLabel("Initializing...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #aaa; font-size: 16px; border: none;")
        right_layout.addWidget(self.status_label)

        # LED indicator
        self.led_status_label = QLabel("LED: OFF")
        self.led_status_label.setStyleSheet("color: #888; font-size: 16px; border: none;")
        right_layout.addWidget(self.led_status_label)

        # Timer info
        self.timer_label = QLabel("Next capture in: --")
        self.timer_label.setStyleSheet("color: #888; font-size: 16px; border: none;")
        right_layout.addWidget(self.timer_label)

        # Capture count
        self.capture_count_label = QLabel("Captures: 0")
        self.capture_count_label.setStyleSheet("color: #aaa; font-size: 20px; font-weight: bold; border: none;")
        right_layout.addWidget(self.capture_count_label)

        right_layout.addStretch()

        # Stop Button - MASSIVE for safety
        self.stop_btn = QPushButton("⏹  Stop Recording")
        self.stop_btn.setFixedHeight(80)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #c62828;
                color: white;
                font-size: 20px;
                font-weight: bold;
                border-radius: 10px;
                border: none;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        self.stop_btn.clicked.connect(self._on_stop)
        right_layout.addWidget(self.stop_btn)

        # Return Button
        self.return_btn = QPushButton("← Return to Setup")
        self.return_btn.setFixedHeight(60)
        self.return_btn.clicked.connect(self._on_return)
        right_layout.addWidget(self.return_btn)

        # Add panels to main layout
        main_layout.addWidget(self.left_panel, stretch=1)
        main_layout.addWidget(self.right_panel, stretch=0)

        # Countdown timer for UI updates
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)  # Update every second
        self._countdown_timer.timeout.connect(self._update_countdown)
        self._seconds_until_capture = 10

    def _setup_graph_style(self):
        """Configure graph appearance."""
        self.ax.set_facecolor('#2a2a2a')
        self.ax.set_title("Detections over Time", color='white', fontsize=10)
        self.ax.set_xlabel("Capture #", color='#888')
        self.ax.set_ylabel("Count", color='#888')
        self.ax.tick_params(colors='#888')
        self.ax.spines['bottom'].set_color('#444')
        self.ax.spines['top'].set_color('#444')
        self.ax.spines['left'].set_color('#444')
        self.ax.spines['right'].set_color('#444')
        self.ax.grid(True, linestyle='--', alpha=0.3, color='#444')

    # ========== LED Control ==========

    def _init_led(self):
        """Initialize LED GPIO."""
        if self._led_initialized:
            return
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.led_pin, GPIO.OUT, initial=GPIO.LOW)
            self._led_initialized = True
        except Exception as e:
            print(f"[RecordRunning] Failed to init LED GPIO: {e}")

    def _set_led(self, on: bool):
        self._init_led()
        try:
            GPIO.output(self.led_pin, GPIO.HIGH if on else GPIO.LOW)
            state = "ON" if on else "OFF"
            self.led_status_label.setText(f"LED: {state}")
            self.led_status_label.setStyleSheet(
                f"color: {'#4caf50' if on else '#888'}; border: none;"
            )
        except Exception as e:
            print(f"[RecordRunning] Failed to set LED: {e}")

    # ========== Recording Control ==========

    def set_recording_config(self, settings: dict, recipe_steps: list):
        """Set recording configuration from setup screen."""
        self._recording_settings = settings.copy() if settings else {}
        # Recipe steps are stored but not used for timing - we capture every 10s
        self._recipe_steps = list(recipe_steps) if recipe_steps else []

    def on_enter(self):
        """Called when entering this screen - start recording."""
        self._reset_recording_state()
        self._start_recording()

    def on_leave(self):
        """Called when leaving this screen - stop recording and release camera."""
        self._stop_recording()

    def _reset_recording_state(self):
        """Reset all recording state."""
        self._capture_count = 0
        self._graph_data_time = []
        self._graph_data_counts = []
        self._is_recording = False
        self._seconds_until_capture = 10

        self.capture_count_label.setText("Captures: 0")
        self.timer_label.setText("Next capture in: --")
        self.model_preview.setText("Waiting for capture...")
        self.model_preview.setPixmap(QPixmap())

        self._update_graph()

    def _start_recording(self):
        """Start the recording process."""
        print("[RecordRunning] Starting recording...")

        try:
            # 1. Turn LED ON
            self._set_led(True)

            # 2. Start camera
            self.status_label.setText("Starting camera...")
            self.camera_preview.start_preview(self._recording_settings)

            # 3. Mark as recording
            self._is_recording = True
            self.status_label.setText("Recording - capturing every 10s")

            # 4. Do first capture immediately
            print("[RecordRunning] Doing initial capture...")
            self._do_capture()

            # 5. Start the 10-second capture timer
            self._seconds_until_capture = 10
            self._capture_timer.start()
            self._countdown_timer.start()

            print("[RecordRunning] Recording started successfully")

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self._is_recording = False
            self._set_led(False)
            print(f"[RecordRunning] Failed to start - {e}")
            import traceback
            traceback.print_exc()

    def _stop_recording(self):
        """Stop the recording process and release camera."""

        # Stop timers
        self._capture_timer.stop()
        self._countdown_timer.stop()

        self._is_recording = False

        # Turn LED OFF
        self._set_led(False)

        # Stop camera
        try:
            self.camera_preview.stop_preview()
        except Exception as e:
            print(f"[RecordRunning] Error stopping camera: {e}")

        self.status_label.setText("Recording stopped")
        self.timer_label.setText("Next capture in: --")

    def _on_capture_timer(self):
        """Called every 10 seconds to capture an image."""
        if not self._is_recording:
            return

        self._do_capture()
        self._seconds_until_capture = 10

    def _update_countdown(self):
        """Update the countdown display every second."""
        if not self._is_recording:
            return

        self._seconds_until_capture -= 1
        if self._seconds_until_capture < 0:
            self._seconds_until_capture = 10

        self.timer_label.setText(f"Next capture in: {self._seconds_until_capture}s")

    def _do_capture(self):
        """Capture image and run model analysis."""
        if not self._is_recording:
            return

        try:
            # Check if camera is running
            if not self.camera_preview.is_running():
                print("[RecordRunning] Camera not running!")
                self.status_label.setText("Error: Camera not running")
                return

            # Build output path
            settings = self._recording_settings
            base_dir = settings.get('output_directory', '/tmp')
            current_date = datetime.now().strftime("%Y.%m.%d")
            current_time = datetime.now().strftime("%H.%M.%S")
            date_folder = os.path.join(base_dir, current_date)
            os.makedirs(date_folder, exist_ok=True)

            image_name = settings.get('image_name', 'rec')
            filename = f"{current_date}_{current_time}_{image_name}_{self._capture_count:04d}.jpg"
            output_path = os.path.join(date_folder, filename)

            # Capture to array
            image_array = self.camera_preview.capture_to_array(settings)

            if image_array is None:
                print("[RecordRunning] Capture returned None!")
                self.status_label.setText("Error: Capture failed")
                return

            print(f"[RecordRunning] Captured image shape={image_array.shape}")

            # Save to disk
            from PIL import Image
            img = Image.fromarray(image_array)
            img.save(output_path, quality=settings.get('quality', 95))
            print(f"[RecordRunning] Saved to {output_path}")

            # Run placeholder model
            processed_img, detection_count = self._placeholder_model_inference(image_array)

            # Update model preview
            self._update_model_preview(processed_img)

            # Update graph
            self._capture_count += 1
            self._graph_data_time.append(self._capture_count)
            self._graph_data_counts.append(detection_count)
            self._update_graph()

            # Update UI
            self.capture_count_label.setText(f"Captures: {self._capture_count}")
            self.status_label.setText(f"Captured: {filename}")

        except Exception as e:
            print(f"[RecordRunning] Capture error - {e}")
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"Capture error: {e}")

    def _placeholder_model_inference(self, image_array):
        """Placeholder for AI Model - returns processed image and count."""
        count = random.randint(0, 25)
        processed_array = np.ascontiguousarray(image_array)
        return processed_array, count

    def _update_model_preview(self, image_array):
        """Update the model preview label with processed image."""
        if image_array is None:
            return

        try:
            h, w = image_array.shape[:2]
            ch = image_array.shape[2] if len(image_array.shape) > 2 else 1

            image_array = np.ascontiguousarray(image_array)

            if ch == 3:
                bytes_per_line = 3 * w
                q_img = QImage(image_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                q_img = QImage(image_array.data, w, h, w, QImage.Format.Format_Grayscale8)

            # Must copy the QImage because the numpy array might be recycled
            q_img = q_img.copy()

            pix = QPixmap.fromImage(q_img)
            if not pix.isNull():
                scaled = pix.scaled(
                    self.model_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )
                self.model_preview.setPixmap(scaled)
        except Exception as e:
            print(f"[RecordRunning] Error updating preview - {e}")

    def _update_graph(self):
        """Update the matplotlib graph."""
        try:
            self.ax.clear()
            self._setup_graph_style()

            if self._graph_data_time:
                self.ax.plot(
                    self._graph_data_time,
                    self._graph_data_counts,
                    '-o',
                    linewidth=2,
                    markersize=6,
                    color='#f44336'
                )
                self.ax.fill_between(
                    self._graph_data_time,
                    self._graph_data_counts,
                    alpha=0.2,
                    color='#f44336'
                )

            self.canvas.draw()
        except Exception as e:
            print(f"[RecordRunning] Error updating graph - {e}")

    def _on_stop(self):
        """Handle stop button click."""
        self._stop_recording()
        self.stop_requested.emit()

    def _on_return(self):
        """Handle return button click."""
        if self._is_recording:
            reply = QMessageBox.question(
                self, "Confirm",
                "Recording is still active. Stop and return to setup?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._stop_recording()
        self.back_requested.emit()

    def close(self):
        """Clean up resources."""
        self._stop_recording()

        # Clean up LED GPIO
        try:
            GPIO.output(self.led_pin, GPIO.LOW)
        except Exception:
            pass

        if hasattr(self, 'camera_preview'):
            self.camera_preview.close()
        super().close()