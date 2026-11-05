"""
Home Screen - "Launcher" style dashboard.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage, QColor

class HomeScreen(QWidget):
    """Home screen with hierarchy-based layout and industry styling."""

    free_mode_requested = pyqtSignal()
    record_mode_requested = pyqtSignal()

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.THEME = getattr(config, "theme", {})
        self._build_ui()

    def _build_ui(self):
        """Build the dashboard UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- 1. HEADER BAR ---
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet("background-color: #222; border-bottom: 1px solid #333;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 0, 30, 0)
        header_layout.setSpacing(20)

        brand_small = QLabel("PramZepto Control")
        brand_small.setStyleSheet("color: #666; font-size: 14px; font-weight: bold;")
        header_layout.addWidget(brand_small)

        header_layout.addStretch()

        status_label = QLabel("v1.0  |  System Ready ●")
        status_label.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 14px;")
        header_layout.addWidget(status_label)

        # Theme Toggle Button
        self.theme_btn = QPushButton("🌗")
        self.theme_btn.setFixedSize(40, 40)
        self.theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_btn.setToolTip("Toggle Light/Dark Mode")
        self.theme_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.theme_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 20px;
                color: white;
                font-size: 18px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #444;
                border-color: #777;
            }
        """)
        self.theme_btn.clicked.connect(self._on_toggle_theme)
        header_layout.addWidget(self.theme_btn, alignment=Qt.AlignmentFlag.AlignVCenter)

        main_layout.addWidget(header)

        # --- 2. CONTENT AREA ---
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 30, 0, 0)
        content_layout.setSpacing(0)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # [A] LOGO SECTION
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._load_logo()

        content_layout.addWidget(self.logo_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # [B] Top Spring
        content_layout.addStretch(1)

        # [C] HERO CARD: Record Mode
        hero_container = QFrame()
        hero_container.setFixedSize(600, 300)
        hero_container.setCursor(Qt.CursorShape.PointingHandCursor)
        hero_container.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 12px;
            }
            QFrame:hover {
                background-color: #333;
                border: 1px solid #666;
            }
        """)
        hero_container.mousePressEvent = lambda e: self.record_mode_requested.emit()

        hero_layout = QVBoxLayout(hero_container)
        hero_layout.setContentsMargins(40, 40, 40, 40)

        hero_title = QLabel("New Recording Session")
        hero_title.setStyleSheet("color: #fff; font-size: 32px; font-weight: bold; border: none; background: transparent;")
        hero_layout.addWidget(hero_title)

        hero_desc = QLabel("Configure move recipes, set camera parameters, and begin automated model analysis.")
        hero_desc.setWordWrap(True)
        hero_desc.setStyleSheet("color: #aaa; font-size: 16px; border: none; background: transparent;")
        hero_layout.addWidget(hero_desc)

        hero_layout.addStretch()

        cta_btn = QPushButton("Start Setup  →")
        cta_btn.setFixedSize(200, 50)
        cta_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                border: none;
                font-size: 16px;
            }
            QPushButton:hover { background-color: #d32f2f; }
        """)
        cta_btn.clicked.connect(self.record_mode_requested.emit)
        hero_layout.addWidget(cta_btn, alignment=Qt.AlignmentFlag.AlignRight)

        content_layout.addWidget(hero_container, alignment=Qt.AlignmentFlag.AlignCenter)

        # [D] FIXED SPACER
        content_layout.addSpacing(20)

        # [E] UTILITY BAR: Free Mode
        utility_frame = QFrame()
        utility_frame.setFixedSize(600, 80)
        utility_frame.setCursor(Qt.CursorShape.PointingHandCursor)
        utility_frame.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border: 1px solid #555;
                border-radius: 8px;
            }
            QFrame:hover {
                background-color: #4a4a4a;
                border: 1px solid #888;
            }
        """)
        utility_frame.mousePressEvent = lambda e: self.free_mode_requested.emit()

        util_layout = QHBoxLayout(utility_frame)
        util_layout.setContentsMargins(20, 0, 20, 0)

        util_icon = QLabel("🛠")
        util_icon.setStyleSheet("font-size: 24px; border: none; background: transparent;")
        util_layout.addWidget(util_icon)

        util_text = QLabel("Manual Control / Free Mode")
        util_text.setStyleSheet("color: #ccc; font-size: 18px; font-weight: 500; border: none; background: transparent;")
        util_layout.addWidget(util_text)

        util_layout.addStretch()

        util_arrow = QLabel("›")
        util_arrow.setStyleSheet("color: #666; font-size: 28px; border: none; background: transparent;")
        util_layout.addWidget(util_arrow)

        content_layout.addWidget(utility_frame, alignment=Qt.AlignmentFlag.AlignCenter)

        # [F] Bottom Spring
        content_layout.addStretch(3)

        main_layout.addWidget(content_widget)

    def _create_transparent_pixmap(self, image_path):
        """
        Loads an image and programmatically turns white pixels transparent.
        Useful for logos with solid white backgrounds in dark mode.
        """
        image = QImage(image_path)
        if image.isNull():
            return QPixmap()

        # Ensure format supports alpha channel (transparency)
        image = image.convertToFormat(QImage.Format.Format_ARGB32)

        # Define transparency color
        transparent = QColor(0, 0, 0, 0)

        # Threshold for "whiteness" (0-255).
        # Pixels brighter than this in all channels will become transparent.
        # 230 allows for slight compression artifacts/off-white.
        threshold = 230

        for y in range(image.height()):
            for x in range(image.width()):
                color = image.pixelColor(x, y)
                # Check if the pixel is very close to white
                if color.red() > threshold and color.green() > threshold and color.blue() > threshold:
                    image.setPixelColor(x, y, transparent)

        return QPixmap.fromImage(image)

    def _load_logo(self):
        logo_name = "PramZepto.png"
        logo_path = os.path.join(os.path.dirname(__file__), logo_name)
        if not os.path.exists(logo_path):
            logo_path = logo_name

        if os.path.exists(logo_path):
            pix = self._create_transparent_pixmap(logo_path)

            if not pix.isNull():
                scaled = pix.scaledToHeight(200, Qt.TransformationMode.SmoothTransformation)
                self.logo_label.setPixmap(scaled)
                self.logo_label.setFixedSize(scaled.width(), scaled.height())
            else:
                self._set_fallback_text(is_dark=True)
        else:
            self._set_fallback_text(is_dark=True)

    def _set_fallback_text(self, is_dark):
        self.logo_label.setText("PramZepto")
        font = QFont(); font.setPointSize(48); font.setBold(True)
        self.logo_label.setFont(font)
        color = "#00bcd4" if is_dark else "#00796b"
        self.logo_label.setStyleSheet(f"color: {color};")

    def _update_logo_color(self, is_dark):
        if not self.logo_label.pixmap():
            self._set_fallback_text(is_dark)

    def _on_toggle_theme(self):
        top_window = self.window()
        if hasattr(top_window, 'toggle_theme'):
            top_window.toggle_theme()

    def on_enter(self):
        pass

    def on_leave(self):
        pass