"""
Main Application Window with screen management via QStackedWidget.
"""

from PyQt6.QtWidgets import QMainWindow, QStackedWidget, QMessageBox, QWidget, QVBoxLayout
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtCore import Qt

from motors.stepper_motors_hardware import StepperMotorHardware
from widgets.stepper_motors_widget import StepperMotorsWidget
from screens.home_screen import HomeScreen
from screens.free_mode_screen import FreeModeScreen
from screens.record_setup_screen import RecordSetupScreen
from screens.record_running_screen import RecordRunningScreen


class AppMainWindow(QMainWindow):
    """Main window that manages all application screens."""

    # Screen indices
    SCREEN_HOME = 0
    SCREEN_FREE_MODE = 1
    SCREEN_RECORD_SETUP = 2
    SCREEN_RECORD_RUNNING = 3

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config_path = config.cfg_path

        self.THEME = getattr(self.config, "theme", {})
        self.is_dark_mode = True

        if not self.THEME:
            self.THEME = {
                'bg_color': '#2b2b2b',
                'panel_bg': '#333333',
                'font_color': '#ffffff',
                'button_bg': '#3c3c3c',
                'button_hover': '#4a4a4a',
                'border_color': '#555',
                'groupbox_title_bg': '#3c3c3c'
            }

        self.MOTOR = getattr(self.config, "motor", {})
        self.CAMERA = getattr(self.config, "camera", {})

        # Shared hardware controller; lives for the window's lifetime.
        self.mc = StepperMotorHardware(self.config)

        self._init_window()
        self._init_screens()

    def _init_window(self):
        """Configure main window properties."""
        self.setWindowTitle("PramZepto")

        screen_geometry = QGuiApplication.primaryScreen().geometry()
        self.window_width = screen_geometry.width()
        self.window_height = screen_geometry.height() - 70

        self.setWindowState(Qt.WindowState.WindowMaximized)
        self._apply_global_styles()

    def toggle_theme(self):
        """Switch between Light and Dark modes."""
        self.is_dark_mode = not self.is_dark_mode

        if self.is_dark_mode:
            self.THEME.update({
                'bg_color': '#2b2b2b',
                'panel_bg': '#333333',
                'font_color': '#ffffff',
                'button_bg': '#3c3c3c',
                'button_hover': '#4a4a4a',
                'button_pressed': '#2a2a2a',
                'border_color': '#555',
                'groupbox_title_bg': '#3c3c3c',
                'input_bg': '#404040',
                'input_border': '#666',
                'table_head_bg': '#444',
                'table_grid': '#555'
            })
        else:
            self.THEME.update({
                'bg_color': '#f0f2f5',
                'panel_bg': '#ffffff',
                'font_color': '#000000',
                'button_bg': '#e0e0e0',
                'button_hover': '#d0d0d0',
                'button_pressed': '#c0c0c0',
                'border_color': '#cccccc',
                'groupbox_title_bg': '#e0e0e0',
                'input_bg': '#ffffff',
                'input_border': '#999',
                'table_head_bg': '#d0d0d0',
                'table_grid': '#ccc'
            })

        self._apply_global_styles()

        if hasattr(self, 'home_screen'):
            self.home_screen._update_logo_color(self.is_dark_mode)

    def _apply_global_styles(self):
        """Apply global stylesheet."""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.THEME.get('bg_color', '#2b2b2b')};
                font-family: {self.THEME.get('font_family', 'Segoe UI')};
                font-size: 14pt;
                color: {self.THEME.get('font_color', '#ffffff')};
            }}

            /* --- BUTTONS --- */
            QPushButton {{
                background-color: {self.THEME.get('button_bg', '#3c3c3c')};
                color: {self.THEME.get('font_color', '#ffffff')};
                border: 1px solid {self.THEME.get('border_color', '#555')};
                border-radius: 4px;
                padding: 10px 20px;
                min-height: 50px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.THEME.get('button_hover', '#4a4a4a')};
            }}
            QPushButton:pressed {{
                background-color: {self.THEME.get('button_pressed', '#2a2a2a')};
            }}
            QPushButton:disabled {{
                background-color: {self.THEME.get('button_bg', '#3c3c3c')};
                opacity: 0.6;
                color: #888888;
                border: 1px solid #444;
            }}

            /* --- INPUTS --- */
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                min-height: 45px;
                padding: 5px;
                border: 2px solid {self.THEME.get('input_border', '#666')};
                border-radius: 4px;
                background-color: {self.THEME.get('input_bg', '#404040')};
                color: {self.THEME.get('font_color', '#ffffff')};
            }}
            QComboBox::drop-down {{
                width: 40px;
                border-left: 1px solid {self.THEME.get('input_border', '#666')};
            }}

            /* --- TABLES & HEADERS --- */
            QTableWidget {{
                background-color: {self.THEME.get('input_bg', '#404040')};
                gridline-color: {self.THEME.get('table_grid', '#555')};
                border: 1px solid {self.THEME.get('border_color', '#555')};
                color: {self.THEME.get('font_color', '#ffffff')};
            }}
            QHeaderView::section {{
                background-color: {self.THEME.get('table_head_bg', '#444')};
                color: {self.THEME.get('font_color', '#ffffff')};
                padding: 4px;
                border: 1px solid {self.THEME.get('border_color', '#555')};
                font-weight: bold;
            }}
            QTableCornerButton::section {{
                background-color: {self.THEME.get('table_head_bg', '#444')};
                border: 1px solid {self.THEME.get('border_color', '#555')};
            }}
            QTableWidget::item:selected {{
                background-color: #2196F3;
                color: white;
            }}

            /* --- FRAMES & PANELS --- */
            QFrame {{
                border: none;
            }}

            /* --- GROUPBOXES --- */
            QGroupBox {{
                background-color: {self.THEME.get('panel_bg', '#333333')};
                border: 1px solid {self.THEME.get('border_color', '#555')};
                border-radius: 5px;
                margin-top: 25px;
                padding: 20px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
                background-color: {self.THEME.get('groupbox_title_bg', '#3c3c3c')};
                font-weight: bold;
                color: {self.THEME.get('font_color', '#ffffff')};
            }}

            /* --- SCROLLBARS --- */
            QScrollBar:vertical {{
                border: none;
                background: {self.THEME.get('bg_color', '#2b2b2b')};
                width: 40px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: #666;
                min-height: 40px;
                border-radius: 4px;
            }}
        """)

    def _init_screens(self):
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Shared stepper widget used by both Free Mode and Record Setup.
        self._stepper_widget_container = QWidget()
        self._stepper_widget_layout = QVBoxLayout(self._stepper_widget_container)
        self._stepper_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.shared_stepper_widget = StepperMotorsWidget(
            self._stepper_widget_layout, self.config, self.mc
        )

        self.home_screen = HomeScreen(self.config, parent=self)
        self.free_mode_screen = FreeModeScreen(
            self.config, self.mc, self.shared_stepper_widget, parent=self
        )
        self.record_setup_screen = RecordSetupScreen(
            self.config, self.mc, self.shared_stepper_widget, parent=self
        )
        self.record_running_screen = RecordRunningScreen(self.config, self.mc, parent=self)

        self.stack.addWidget(self.home_screen)
        self.stack.addWidget(self.free_mode_screen)
        self.stack.addWidget(self.record_setup_screen)
        self.stack.addWidget(self.record_running_screen)

        self.home_screen.free_mode_requested.connect(self._go_to_free_mode)
        self.home_screen.record_mode_requested.connect(self._go_to_record_setup)
        self.free_mode_screen.back_requested.connect(self._go_to_home)
        self.record_setup_screen.back_requested.connect(self._go_to_home)
        self.record_setup_screen.start_recording_requested.connect(self._go_to_record_running)
        self.record_running_screen.stop_requested.connect(self._go_to_record_setup)
        self.record_running_screen.back_requested.connect(self._go_to_record_setup)

        self.stack.setCurrentIndex(self.SCREEN_HOME)

    # ==================== Navigation ====================

    def _go_to_home(self):
        self._cleanup_current_screen()
        self.stack.setCurrentIndex(self.SCREEN_HOME)

    def _go_to_free_mode(self):
        self._cleanup_current_screen()
        self.free_mode_screen.on_enter()
        self.stack.setCurrentIndex(self.SCREEN_FREE_MODE)

    def _go_to_record_setup(self):
        self._cleanup_current_screen()
        self.record_setup_screen.on_enter()
        self.stack.setCurrentIndex(self.SCREEN_RECORD_SETUP)

    def _go_to_record_running(self):
        self._cleanup_current_screen()
        settings = self.record_setup_screen.get_recording_settings()
        recipe_steps = self.record_setup_screen.get_recipe_steps()
        self.record_running_screen.set_recording_config(settings, recipe_steps)
        self.record_running_screen.on_enter()
        self.stack.setCurrentIndex(self.SCREEN_RECORD_RUNNING)

    def _cleanup_current_screen(self):
        current = self.stack.currentWidget()
        if hasattr(current, 'on_leave'):
            try:
                current.on_leave()
            except Exception as e:
                print(f"[AppMainWindow] Error in on_leave: {e}")

    # ==================== Cleanup ====================

    def closeEvent(self, event):
        for i in range(self.stack.count()):
            widget = self.stack.widget(i)
            if hasattr(widget, 'on_leave'):
                try: widget.on_leave()
                except Exception: pass
            if hasattr(widget, 'close'):
                try: widget.close()
                except Exception: pass
        try:
            self.mc.close()
        except Exception: pass
        super().closeEvent(event)