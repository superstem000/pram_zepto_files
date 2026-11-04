"""
Camera Preview Widget — wraps Picamera2 for live preview, single
capture, and autofocus-grade flushed capture.

Only one Picamera2 instance can exist at a time. This widget fully
releases the camera when stop_preview() is called so other widgets
or screens can use it.
"""

import time
import numpy as np

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

from picamera2 import Picamera2
from libcamera import Transform

from camera.camera_thread import CameraPreviewThread, apply_channel_filter


class CameraPreviewWidget(QWidget):

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.CAMERA = getattr(config, "camera", {})

        self.picam2 = None
        self.preview_thread = None
        self._is_running = False
        self._current_settings = {}

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background-color: black;")
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.preview_label.setMinimumSize(1, 1)

        layout.addWidget(self.preview_label)

    def _release_camera(self):
        """Fully release all camera resources. Must be called before
        another widget/screen can use the camera."""
        if self.preview_thread is not None:
            try:
                self.preview_thread.stop_stream()
                if self.preview_thread.isRunning():
                    self.preview_thread.wait(3000)
            except Exception as e:
                print(f"[CameraPreview] Error stopping preview thread: {e}")
            finally:
                try:
                    self.preview_thread.frame_ready.disconnect()
                    self.preview_thread.error.disconnect()
                except Exception:
                    pass
                self.preview_thread = None

        if self.picam2 is not None:
            if self._is_running:
                try:
                    self.picam2.stop()
                except Exception as e:
                    print(f"[CameraPreview] Error stopping camera: {e}")
            try:
                self.picam2.close()
            except Exception as e:
                print(f"[CameraPreview] Error closing camera: {e}")
            finally:
                self.picam2 = None

        self._is_running = False
        time.sleep(0.3)  # let hardware release before next init

    def _init_camera(self, settings: dict):
        self._release_camera()

        try:
            self.picam2 = Picamera2()

            preview_w, preview_h = (1280, 720)
            capture_w = settings.get('width', 4056)
            capture_h = settings.get('height', 3040)

            transform = Transform(
                hflip=1 if settings.get('hflip', False) else 0,
                vflip=1 if settings.get('vflip', False) else 0
            )

            denoise_mode = self._get_denoise_mode(settings.get('denoise', 'off'))
            preview_controls = {}
            if denoise_mode is not None:
                preview_controls["NoiseReductionMode"] = denoise_mode

            # Dual stream: main = full-res for capture, lores = low-res for preview.
            self.preview_config = self.picam2.create_preview_configuration(
                main={"size": (capture_w, capture_h), "format": "RGB888"},
                lores={"size": (preview_w, preview_h), "format": "RGB888"},
                raw={},
                transform=transform,
                controls=preview_controls
            )
            self.picam2.configure(self.preview_config)

            stream = "raw" if settings.get('raw_preview', False) else "lores"
            self.preview_thread = CameraPreviewThread(
                self.picam2, fps=30, stream=stream, parent=self
            )
            self.preview_thread.frame_ready.connect(self._on_preview_frame)
            self.preview_thread.error.connect(self._on_camera_error)

            self._current_settings = settings.copy()
            return True

        except Exception as e:
            print(f"[CameraPreview] Camera initialization FAILED: {e}")
            import traceback
            traceback.print_exc()
            self._release_camera()
            raise RuntimeError(f"Camera initialization failed: {e}")

    def _get_denoise_mode(self, denoise_str: str):
        if not denoise_str:
            return None
        den = str(denoise_str).strip().lower()
        if not den or den == "off":
            return None

        try:
            import libcamera
            nre = libcamera.controls.draft.NoiseReductionModeEnum
            return {
                "off": nre.Off,
                "fast": nre.Fast,
                "high_quality": nre.HighQuality,
                "highquality": nre.HighQuality,
            }.get(den)
        except Exception:
            return {"off": 0, "fast": 1, "high_quality": 2, "highquality": 2}.get(den)

    # (settings-key, libcamera-control-name, caster)
    _SIMPLE_CONTROLS = (
        ('shutter',      'ExposureTime',  lambda v: int(float(v))),
        ('ev',           'ExposureValue', float),
        ('gain',         'AnalogueGain',  float),
        ('digital_gain', 'DigitalGain',   float),
        ('contrast',     'Contrast',      float),
        ('brightness',   'Brightness',    float),
        ('saturation',   'Saturation',    float),
        ('sharpness',    'Sharpness',     float),
    )

    def _apply_controls(self, settings: dict, skip_denoise: bool = False):
        if not self.picam2:
            return

        try:
            supported = set(getattr(self.picam2, "camera_controls", {}).keys())
        except Exception:
            supported = set()

        controls = {}

        def put(name: str, value):
            if supported and name not in supported:
                return
            controls[name] = value

        for key, control_name, cast in self._SIMPLE_CONTROLS:
            raw = str(settings.get(key, '')).strip()
            if not raw:
                continue
            try:
                put(control_name, cast(raw))
            except ValueError:
                pass

        awb = str(settings.get('awbgains', '')).strip()
        if awb:
            try:
                r_str, b_str = [x.strip() for x in awb.split(",")]
                r, bl = float(r_str), float(b_str)
                put("AwbEnable", False)
                put("ColourGains", (r, bl))
            except Exception:
                pass
        else:
            put("AwbEnable", True)

        if not skip_denoise:
            den = str(settings.get('denoise', '')).strip().lower()
            if den:
                nrm_val = self._get_denoise_mode(den)
                if nrm_val is not None:
                    put("NoiseReductionMode", nrm_val)

        if controls:
            try:
                self.picam2.set_controls(controls)
            except Exception as e:
                print(f"[CameraPreview] Control apply failed: {e}")

    def _on_preview_frame(self, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
        self.preview_label.setPixmap(scaled)

    def _on_camera_error(self, msg: str):
        print(f"[CameraPreview] Camera error: {msg}")

    # ========== Public Interface ==========

    def start_preview(self, settings: dict):
        self._init_camera(settings)
        try:
            self.picam2.start()
            time.sleep(0.1)
            self._apply_controls(settings)
            self.preview_thread.start_stream()
            self._is_running = True
        except Exception as e:
            print(f"[CameraPreview] Failed to start preview: {e}")
            import traceback
            traceback.print_exc()
            self._release_camera()
            raise RuntimeError(f"Failed to start camera preview: {e}")

    def stop_preview(self):
        """Stop live preview and fully release the camera."""
        self._release_camera()
        self.preview_label.clear()
        self.preview_label.setStyleSheet("background-color: black;")

    def apply_settings(self, settings: dict):
        """Apply updated camera controls to the running camera."""
        if not self.picam2 or not self._is_running:
            raise RuntimeError("Camera not running — start live preview first")

        self._apply_controls(settings)
        self._current_settings.update(settings)

        if self.preview_thread:
            channel = str(settings.get('channel', 'RGB')).lower()
            self.preview_thread.set_channel(channel)

            stream = "raw" if settings.get('raw_preview', False) else "lores"
            self.preview_thread.set_stream(stream)

    def capture_image(self, output_path: str, settings: dict):
        if not self.picam2 or not self._is_running:
            raise RuntimeError("Camera not running")

        self._apply_controls(settings, skip_denoise=True)
        self.picam2.options["quality"] = settings.get('quality', 95)

        request = self.picam2.capture_request()

        try:
            channel = str(settings.get('channel', 'RGB')).lower()
            if channel == "rgb":
                request.save("main", output_path)
            else:
                frame = request.make_array("main")
                frame = self._ensure_rgb(frame)
                frame = self._apply_channel_filter(frame, channel)

                from PIL import Image
                img = Image.fromarray(frame)
                img.save(output_path, quality=settings.get('quality', 95))

            if settings.get('save_raw', False):
                raw_path = output_path.replace('.jpg', '.dng')
                request.save_dng(raw_path)
        finally:
            request.release()

    def capture_to_array(self, settings: dict) -> np.ndarray:
        """Capture one frame and return it as an RGB numpy array."""
        if not self.picam2 or not self._is_running:
            return None

        try:
            self._apply_controls(settings, skip_denoise=True)

            request = self.picam2.capture_request()
            try:
                frame = request.make_array("main")
            finally:
                request.release()

            frame = self._ensure_rgb(frame)

            channel = str(settings.get('channel', 'RGB')).lower()
            if channel != 'rgb':
                frame = self._apply_channel_filter(frame, channel)

            return np.ascontiguousarray(frame)

        except Exception as e:
            print(f"[CameraPreview] capture_to_array failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def flush_and_capture(self, settings: dict, discard: int = 3) -> np.ndarray:
        """Discard stale buffered frames, then capture a fresh one.

        Required for autofocus: after a Z move the buffer still holds
        frames from the previous position; discarding ensures the
        returned image matches the current Z.
        """
        if not self.picam2 or not self._is_running:
            return None

        try:
            for i in range(discard):
                req = self.picam2.capture_request()
                req.release()

            self._apply_controls(settings, skip_denoise=True)
            request = self.picam2.capture_request()
            try:
                frame = request.make_array("main")
            finally:
                request.release()

            frame = self._ensure_rgb(frame)

            channel = str(settings.get('channel', 'RGB')).lower()
            if channel != 'rgb':
                frame = self._apply_channel_filter(frame, channel)

            return np.ascontiguousarray(frame)

        except Exception as e:
            print(f"[CameraPreview] flush_and_capture failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _ensure_rgb(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return None
        if len(frame.shape) == 2:
            return np.stack([frame, frame, frame], axis=2)
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:
                # XBGR -> RGB (drop alpha)
                return frame[:, :, 2::-1].copy()
            if frame.shape[2] == 3:
                return frame[:, :, ::-1].copy()
        return frame

    def _apply_channel_filter(self, frame: np.ndarray, channel: str) -> np.ndarray:
        return apply_channel_filter(frame, channel)

    def is_running(self) -> bool:
        return self._is_running

    def close(self):
        self._release_camera()
        super().close()
