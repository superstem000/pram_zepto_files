"""Camera Controls Widget — settings form + capture buttons."""

from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QFormLayout, QGridLayout,
    QSpinBox, QLineEdit, QSizePolicy, QCheckBox, QComboBox, QMessageBox, QSpacerItem
)
from PyQt6.QtCore import pyqtSignal
from widgets.widget_collapsible_box import CollapsibleBox
import os
import RPi.GPIO as GPIO
from config import save_config_atomic
from datetime import datetime

class CameraControlsWidget(QWidget):
    live_toggled = pyqtSignal(bool)
    capture_requested = pyqtSignal()
    settings_changed = pyqtSignal(dict)

    # Settings whose UI widget is a QLineEdit. For each key K,
    # the widget attribute is self.camera_<K> and the config key is default_<K>.
    _TEXT_SETTINGS_KEYS = (
        "shutter", "contrast", "ev", "prev_time", "gain", "digital_gain",
        "saturation", "awbgains", "sharpness", "brightness",
    )

    def __init__(self, parent_layout, config, mode='free', parent=None):
        super().__init__(parent)
        self.parent_layout = parent_layout
        self.config = config
        self.config_path = getattr(config, "cfg_path", None)
        self.mode = mode
        self.CAMERA = getattr(self.config, "camera", {})
        self._live_running = False

        self.led_pin = 20
        self.led_on = False
        self.LED_btn = None
        if self.mode == 'free':
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.led_pin, GPIO.OUT, initial=GPIO.LOW)
            except: pass

        self.default_folder = os.path.expanduser("~/Desktop/Users/")
        os.makedirs(self.default_folder, exist_ok=True)
        self.selected_directory = self.default_folder

        self._build_ui()

    def _build_ui(self):
        box = CollapsibleBox("Camera Control")

        panel = QFrame()
        panel_layout = QHBoxLayout(panel)
        panel_layout.setContentsMargins(30, 30, 30, 120)
        panel_layout.setSpacing(40)

        # ── Left column: settings list ──
        left_widget = QWidget()
        left_grid = QGridLayout(left_widget)
        left_grid.setContentsMargins(0, 0, 0, 0)
        left_grid.setVerticalSpacing(18)
        left_grid.setHorizontalSpacing(20)

        self.status_label = QLabel("Idle")
        self.camera_width = QSpinBox(); self.camera_width.setMaximum(10000)
        self.camera_height = QSpinBox(); self.camera_height.setMaximum(10000)

        def config_input(w):
            w.setMinimumHeight(50)
            return w

        self.camera_shutter = config_input(QLineEdit()); self.camera_shutter.setPlaceholderText("e.g. 10000")
        self.camera_contrast = config_input(QLineEdit()); self.camera_contrast.setPlaceholderText("e.g. 1.0")
        self.camera_ev = config_input(QLineEdit()); self.camera_ev.setPlaceholderText("e.g. 0")
        self.camera_prev_time = config_input(QLineEdit()); self.camera_prev_time.setPlaceholderText("e.g. 10000")
        self.camera_gain = config_input(QLineEdit()); self.camera_gain.setPlaceholderText("e.g. 1.5")
        self.camera_digital_gain = config_input(QLineEdit()); self.camera_digital_gain.setPlaceholderText("e.g. 1.0")
        self.camera_saturation = config_input(QLineEdit()); self.camera_saturation.setPlaceholderText("e.g. 1.0")
        self.camera_awbgains = config_input(QLineEdit()); self.camera_awbgains.setPlaceholderText("red,blue")
        self.camera_sharpness = config_input(QLineEdit()); self.camera_sharpness.setPlaceholderText("e.g. 1.0")
        self.camera_brightness = config_input(QLineEdit()); self.camera_brightness.setPlaceholderText("e.g. 0.5")

        self.camera_width.setMinimumHeight(50)
        self.camera_height.setMinimumHeight(50)

        # Load defaults from config.
        self.camera_width.setValue(int(self._cam_get("default_width", 4056)))
        self.camera_height.setValue(int(self._cam_get("default_height", 3040)))
        for key in self._TEXT_SETTINGS_KEYS:
            getattr(self, f"camera_{key}").setText(str(self._cam_get(f"default_{key}", "")))

        row = 0
        def add_row(label_text, widget):
            nonlocal row
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-size: 14px;")
            left_grid.addWidget(lbl, row, 0)
            left_grid.addWidget(widget, row, 1)
            row += 1

        add_row("Status:", self.status_label)
        add_row("Width:", self.camera_width)
        add_row("Height:", self.camera_height)
        add_row("Exposure (µs):", self.camera_shutter)
        add_row("Contrast:", self.camera_contrast)
        add_row("EV:", self.camera_ev)
        add_row("Prev. Time:", self.camera_prev_time)
        add_row("AnalogueGain:", self.camera_gain)
        add_row("DigitalGain:", self.camera_digital_gain)
        add_row("AWB Gains (r,b):", self.camera_awbgains)
        add_row("Saturation:", self.camera_saturation)
        add_row("Sharpness:", self.camera_sharpness)
        add_row("Brightness:", self.camera_brightness)

        left_grid.setRowStretch(row, 1)
        left_grid.addItem(QSpacerItem(20, 60), row + 1, 0)

        # ── Right column: controls ──
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(25)

        # Denoise + quality
        row1 = QHBoxLayout()
        row1.setSpacing(10)

        self.camera_denoise = QComboBox(); self.camera_denoise.addItems(["off", "fast", "high_quality"]); self.camera_denoise.setMinimumHeight(50)
        saved_denoise = str(self._cam_get("default_denoise", "off")).strip()
        idx = self.camera_denoise.findText(saved_denoise)
        if idx >= 0:
            self.camera_denoise.setCurrentIndex(idx)
        self.camera_quality = QSpinBox(); self.camera_quality.setRange(0, 100); self.camera_quality.setValue(int(self._cam_get("default_quality", 95))); self.camera_quality.setMinimumHeight(50)

        lbl_denoise = QLabel("Denoise:"); lbl_denoise.setFixedWidth(70)
        lbl_qual = QLabel("Quality:"); lbl_qual.setFixedWidth(60)

        row1.addWidget(lbl_denoise)
        row1.addWidget(self.camera_denoise, 1)
        row1.addWidget(lbl_qual)
        row1.addWidget(self.camera_quality, 1)

        right_layout.addLayout(row1)

        # Channel
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        self.camera_channel = QComboBox(); self.camera_channel.addItems(["RGB", "Red", "Green", "Blue", "Grayscale"]); self.camera_channel.setMinimumHeight(50)

        lbl_chan = QLabel("Channel:"); lbl_chan.setFixedWidth(70)
        row2.addWidget(lbl_chan)
        row2.addWidget(self.camera_channel, 1)

        right_layout.addLayout(row2)

        # Checkboxes
        check_layout = QVBoxLayout()
        check_layout.setSpacing(12)
        self.camera_raw_preview = QCheckBox("Raw Preview")
        self.camera_fullraw = QCheckBox("Save Raw (DNG)")
        self.camera_no_prev = QCheckBox("No Preview")
        self.camera_hflip = QCheckBox("Horizontal Flip")
        self.camera_vflip = QCheckBox("Vertical Flip")

        for chk in (self.camera_raw_preview, self.camera_fullraw, self.camera_no_prev, self.camera_hflip, self.camera_vflip):
            chk.setMinimumHeight(35)
            check_layout.addWidget(chk)

        right_layout.addLayout(check_layout)

        right_layout.addSpacing(15)

        # File saving
        self.folder_search_btn = QPushButton("Select Folder to Save")
        self.folder_search_btn.setMinimumHeight(55)
        self.folder_search_btn.clicked.connect(self._select_save_directory)

        self.selected_directory_label = QLabel(self.default_folder)
        self.selected_directory_label.setStyleSheet("color: gray; font-size: 13px; margin-bottom: 5px;")
        self.selected_directory_label.setWordWrap(True)

        image_name_layout = QHBoxLayout()
        image_name_layout.addWidget(QLabel("Image Name:"))
        self.image_name = QLineEdit()
        self.image_name.setMinimumHeight(50)
        image_name_layout.addWidget(self.image_name)

        right_layout.addWidget(self.folder_search_btn)
        right_layout.addWidget(self.selected_directory_label)
        right_layout.addLayout(image_name_layout)

        right_layout.addSpacing(15)

        # Action buttons
        self.set_default_btn = QPushButton("Set Default")
        self.set_default_btn.setMinimumHeight(55)
        self.set_default_btn.clicked.connect(self._set_default_settings)

        if self.mode == 'free':
            self.show_live_btn = QPushButton("Show Live")
            self.show_live_btn.setMinimumHeight(55)
            self.show_live_btn.clicked.connect(self._on_live_toggle)

            self.apply_btn = QPushButton("⟳  Apply Settings")
            self.apply_btn.setMinimumHeight(55)
            self.apply_btn.setToolTip("Push current settings to the live camera")
            self.apply_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1565C0;
                    color: white;
                    font-weight: bold;
                    border-radius: 6px;
                    border: 2px solid #0D47A1;
                }
                QPushButton:hover { background-color: #1976D2; }
                QPushButton:pressed { background-color: #0D47A1; }
                QPushButton:disabled { background-color: #555; color: #999; border-color: #444; }
            """)
            self.apply_btn.clicked.connect(self._on_apply_settings)

            self.capture_btn = QPushButton("Capture && Save")
            self.capture_btn.setMinimumHeight(55)
            self.capture_btn.clicked.connect(lambda: self.capture_requested.emit())

            self.LED_btn = QPushButton("Turn LED On")
            self.LED_btn.setMinimumHeight(55)
            self.LED_btn.clicked.connect(self._toggle_LED)

            btn_stack = QVBoxLayout()
            btn_stack.setSpacing(30)

            btn_stack.addWidget(self.set_default_btn)
            btn_stack.addWidget(self.LED_btn)
            btn_stack.addWidget(self.show_live_btn)
            btn_stack.addWidget(self.apply_btn)
            btn_stack.addWidget(self.capture_btn)

            right_layout.addLayout(btn_stack)
        else:
            right_layout.addWidget(self.set_default_btn)

        right_layout.addSpacing(50)
        right_layout.addStretch()

        panel_layout.addWidget(left_widget, stretch=1)
        panel_layout.addWidget(right_widget, stretch=1)

        content_layout = QVBoxLayout()
        content_layout.addWidget(panel)
        box.setContentLayout(content_layout)
        self.parent_layout.addWidget(box)
        self.camera_group = box

    # --- Helpers ---
    def _cam_get(self, key, default):
        cam = self.CAMERA
        return cam.get(key, default) if isinstance(cam, dict) else getattr(cam, key, default)

    def _set_cam(self, key, value):
        if isinstance(self.CAMERA, dict): self.CAMERA[key] = value
        else: setattr(self.CAMERA, key, value)

    def _safe_text(self, w: QLineEdit) -> str:
        return w.text().strip()

    def get_camera_settings(self) -> dict:
        d = {
            'width':  self.camera_width.value(),
            'height': self.camera_height.value(),
        }
        for key in self._TEXT_SETTINGS_KEYS:
            if key == 'prev_time':
                continue  # UI-only, not sent to the camera
            d[key] = self._safe_text(getattr(self, f"camera_{key}"))
        d.update({
            'denoise': self.camera_denoise.currentText(),
            'channel': self.camera_channel.currentText(),
            'quality': self.camera_quality.value(),
            'hflip': self.camera_hflip.isChecked(),
            'vflip': self.camera_vflip.isChecked(),
            'raw_preview': self.camera_raw_preview.isChecked(),
            'save_raw': self.camera_fullraw.isChecked(),
            'no_preview': self.camera_no_prev.isChecked(),
            'output_directory': self.selected_directory_label.text(),
            'image_name': self.image_name.text() or "image",
        })
        return d

    def get_output_path(self) -> str:
        base_dir = self.selected_directory_label.text()
        if not base_dir or base_dir == "No directory selected":
            QMessageBox.critical(self, "Error", "No directory selected")
            return None
        current_date = datetime.now().strftime("%Y.%m.%d")
        current_time = datetime.now().strftime("%H.%M.%S")
        date_folder = os.path.join(base_dir, current_date)
        os.makedirs(date_folder, exist_ok=True)
        image_name = self.image_name.text() or "image"
        return os.path.join(date_folder, f"{current_date}_{current_time}_{image_name}.jpg")

    def set_status(self, text: str):
        self.status_label.setText(text)

    def set_live_state(self, running: bool):
        self._live_running = running
        self.show_live_btn.setText("Stop Live" if running else "Show Live")
        self.status_label.setText("Live" if running else "Idle")

    def _on_live_toggle(self):
        self.live_toggled.emit(not self._live_running)

    def _on_apply_settings(self):
        """Emit settings_changed so the parent screen can push them to the camera."""
        if not self._live_running:
            QMessageBox.warning(self, "Not Live", "Start live preview first, then apply settings.")
            return
        self.settings_changed.emit(self.get_camera_settings())
        self.status_label.setText("Settings applied")

    def _select_save_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder", self.selected_directory)
        if folder:
            self.selected_directory = folder
            self.selected_directory_label.setText(folder)

    def _set_default_settings(self):
        """Persist the current UI values as defaults in config, then save to disk."""
        self._set_cam("default_width", int(self.camera_width.value()))
        self._set_cam("default_height", int(self.camera_height.value()))
        for key in self._TEXT_SETTINGS_KEYS:
            self._set_cam(f"default_{key}", self._safe_text(getattr(self, f"camera_{key}")))
        self._set_cam("default_denoise", self.camera_denoise.currentText().strip())
        self._set_cam("default_quality", int(self.camera_quality.value()))

        try:
            if self.config_path:
                save_config_atomic(self.config, self.config_path)
            QMessageBox.information(self, "Saved", "Default camera settings saved.")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Could not save config:\n{e}")

    def _toggle_LED(self):
        if self.mode != 'free': return
        try:
            self.led_on = not self.led_on
            GPIO.output(self.led_pin, GPIO.HIGH if self.led_on else GPIO.LOW)
            self.LED_btn.setText("Turn LED Off" if self.led_on else "Turn LED On")
        except Exception as e:
            QMessageBox.critical(self, "GPIO Error", f"Failed to toggle LED:\\n{e}")
            self.led_on = False

    def close(self):
        if self.mode == 'free':
            try: GPIO.output(self.led_pin, GPIO.LOW); GPIO.cleanup(self.led_pin)
            except: pass
        super().close()