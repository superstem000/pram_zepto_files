from PyQt6 import QtCore
from gpiozero import DigitalInputDevice


class StepperMotorsPhotoswitch(QtCore.QObject):
    entered_home = QtCore.pyqtSignal(str)  # axis
    left_home    = QtCore.pyqtSignal(str)  # axis

    def __init__(self, axis_gpio: dict, pull_up: bool = True,
                 active_state: str = "low", debounce_s: float = 0.002, parent=None):
        super().__init__(parent)
        self._devs = {}
        self._active_low = (str(active_state).lower() == "low")

        bounce_time = None if (debounce_s is None or debounce_s <= 0) else float(debounce_s)

        for axis, gpio in axis_gpio.items():
            dev = DigitalInputDevice(int(gpio), pull_up=pull_up, bounce_time=bounce_time)
            self._devs[axis] = dev

            # Map gpiozero's activated/deactivated edges to home enter/leave
            # based on whether HOME is defined as active-low or active-high.
            if self._active_low:
                dev.when_deactivated = lambda ax=axis: self.entered_home.emit(ax)
                dev.when_activated   = lambda ax=axis: self.left_home.emit(ax)
            else:
                dev.when_activated   = lambda ax=axis: self.entered_home.emit(ax)
                dev.when_deactivated = lambda ax=axis: self.left_home.emit(ax)

    def is_home(self, axis: str) -> bool:
        dev = self._devs.get(axis)
        if not dev:
            return False
        v = 1 if dev.value > 0.5 else 0
        return (v == 0) if self._active_low else (v == 1)

    def close(self):
        for d in self._devs.values():
            try:
                d.close()
            except Exception:
                pass
