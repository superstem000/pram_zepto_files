import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


def apply_channel_filter(frame: np.ndarray, channel: str) -> np.ndarray:
    """Apply a named channel filter to an RGB frame.

    'rgb' (default) returns the frame unchanged; 'red' / 'green' / 'blue'
    zero out the other channels; 'grayscale' returns a 3-channel gray.
    Unknown values return the frame unchanged.
    """
    channel = channel.lower() if isinstance(channel, str) else "rgb"
    if channel in ("rgb", ""):
        return frame
    if channel in ("red", "green", "blue"):
        idx = {"red": 0, "green": 1, "blue": 2}[channel]
        filtered = np.zeros_like(frame)
        filtered[:, :, idx] = frame[:, :, idx]
        return filtered
    if channel == "grayscale":
        gray = np.mean(frame, axis=2).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=2)
    return frame


def _raw_to_qimage_gray(raw: np.ndarray) -> QImage:
    # raw may be uint16 with 10/12-bit data
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    raw = np.ascontiguousarray(raw)

    # Normalize robustly for display (percentile stretch)
    r = raw.astype(np.float32)
    lo = np.percentile(r, 1.0)
    hi = np.percentile(r, 99.0)
    if hi <= lo:
        hi = lo + 1.0
    r = (r - lo) * (255.0 / (hi - lo))
    r = np.clip(r, 0, 255).astype(np.uint8)

    h, w = r.shape
    return QImage(r.data, w, h, w, QImage.Format.Format_Grayscale8).copy()


class CameraPreviewThread(QThread):
    frame_ready = pyqtSignal(QImage)
    error = pyqtSignal(str)

    def __init__(self, picam2, fps=30, stream="main", parent=None):
        super().__init__(parent)
        self.picam2 = picam2
        self.fps = max(1, int(fps))
        self._running = False
        self.stream = stream
        self.channel = "rgb"

    def set_stream(self, stream: str):
        self.stream = stream

    def set_channel(self, channel: str):
        self.channel = channel

    def start_stream(self):
        self._running = True
        if not self.isRunning():
            self.start()

    def stop_stream(self):
        self._running = False
        self.wait(1000)

    def run(self):
        dt = 1.0 / float(self.fps)
        try:
            while self._running:
                if self.stream == "raw":
                    raw = self.picam2.capture_array("raw")
                    qimg = _raw_to_qimage_gray(raw)
                    self.frame_ready.emit(qimg)
                else:
                    frame = self.picam2.capture_array(self.stream)
                    if frame is None:
                        time.sleep(dt)
                        continue

                    # BGR -> RGB swap, then apply channel filter.
                    frame = frame[:, :, ::-1].copy()
                    frame = apply_channel_filter(frame, self.channel)

                    h, w, ch = frame.shape
                    qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                    self.frame_ready.emit(qimg)

                time.sleep(dt)
        except Exception as e:
            self.error.emit(str(e))