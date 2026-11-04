"""
NanoCounter Application Entry Point

Multi-screen application with:
- Home Screen: Mode selection
- Free Mode: Manual control with single capture
- Record Mode Setup: Configure recipes and camera settings
- Record Mode Running: Automated recording with analysis
"""

import sys
from PyQt6.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer

from app_main_window import AppMainWindow
from config import load_config, AppConfig
from pathlib import Path

_HERE = Path(__file__).resolve().parent
CONFIG_PATH = _HERE / "config.json"
SPLASH_IMAGE_PATH = "/home/pramzepto/Desktop/pramZepto/GUI_final/splash.png"

# Keep a module-level reference so the window isn't GC'd
main_window = None


def show_main_window(splash: QSplashScreen, config: AppConfig):
    """Initialize and show the main application window."""
    global main_window
    main_window = AppMainWindow(config)
    main_window.show()
    splash.finish(main_window)


def main():
    app = QApplication(sys.argv)

    # Splash screen
    pix = QPixmap(SPLASH_IMAGE_PATH)
    if pix.isNull():
        pix = QPixmap(500, 350)
        pix.fill(Qt.GlobalColor.black)

    splash = QSplashScreen(
        pix.scaled(
            500, 350,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ),
        Qt.WindowType.SplashScreen | Qt.WindowType.WindowStaysOnTopHint
    )
    splash.setWindowFlag(Qt.WindowType.FramelessWindowHint)
    splash.showMessage(
        "Loading NanoCounter...",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        Qt.GlobalColor.white
    )
    splash.show()
    app.processEvents()

    # Load configuration
    try:
        config: AppConfig = load_config(CONFIG_PATH)
    except Exception as e:
        splash.close()
        QMessageBox.critical(None, "Config Error", f"Failed to load config:\n{e}")
        sys.exit(1)

    # Show main window after splash delay
    QTimer.singleShot(1200, lambda: show_main_window(splash, config))

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
