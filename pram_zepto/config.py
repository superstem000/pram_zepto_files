from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any
import json, os, tempfile

@dataclass
class UARTCfg:
    port: str
    baud: int = 115200

@dataclass
class AxisCfg:
    enable: int
    step: int
    dir: int
    driver_address: int
    microstepping: int
    photoelectric_switch: int
    current_ma: int = 800
    invert_dir: bool = False
    spreadcycle: bool = False

@dataclass
class MotionCfg:
    accel_fullstep: int = 100
    max_speed_fullstep: int = 500

@dataclass
class MotorLimits:
    x_home: int = 0
    y_home: int = 0
    z_home: int = 0
    x_full_step_min: float = 0.0
    x_full_step_max: float = 6000.0
    y_full_step_min: float = 0.0
    y_full_step_max: float = 5200.0
    z_full_step_min: float = 0.0
    z_full_step_max: float = 17000.0
    x_current_position_full_step: float = 0.0
    y_current_position_full_step: float = 0.0
    z_current_position_full_step: float = 0.0

@dataclass
class EndstopsCfg:
    chip: int = 0
    debounce_ms: int = 8
    pull_up: bool = True
    active_state: str = "low"   # "low" or "high"

@dataclass
class ForbiddenZone:
    x_forbidden_start: float = 2900.0
    x_forbidden_end: float = 8000.0
    y_forbidden_start: float = 1500.0
    y_forbidden_end: float = 5000.0
    z_forbidden_start: float = 1000.0
    z_forbidden_end: float = 17000.0

@dataclass
class SpeedControl:
    """Variable speed control via a multiplier on a base speed.
    Multiplier x1.0 = base_speed_fullstep (the 'normal' speed).
    The multiplier is clamped so actual speed stays in [min_speed, max_speed].
    Homing always uses max_speed internally.
    """
    base_speed_fullstep: int = 300    # x1.0 reference speed
    base_accel_fullstep: int = 200    # x1.0 reference accel
    max_speed_fullstep: int = 800     # absolute ceiling
    max_accel_fullstep: int = 400     # accel at max speed
    min_speed_fullstep: int = 3       # absolute floor

@dataclass
class ZSafeRect:
    """Rectangle in XY where Z is allowed to go higher."""
    x_min: float = 4050.0
    x_max: float = 4450.0
    y_min: float = 260.0
    y_max: float = 670.0
    z_max: float = 300.0

@dataclass
class SafetyBounds:
    """Safety bounds for the stage.

    Rules:
      1. z_global_max: Z must stay below this everywhere, EXCEPT inside z_safe_rect
      2. xy_forbidden: cannot have x > xy_forbidden_x AND y > xy_forbidden_y simultaneously
      3. z_safe_rect: inside this XY rectangle, Z can go up to z_safe_rect.z_max
    """
    z_global_max: float = 80.0
    xy_forbidden_x: float = 3800.0
    xy_forbidden_y: float = 850.0
    z_safe_rect: ZSafeRect = field(default_factory=ZSafeRect)

@dataclass
class AppConfig:
    cfg_path: str
    uart: UARTCfg
    axes: Dict[str, AxisCfg]
    motion: MotionCfg = field(default_factory=MotionCfg)
    motor: MotorLimits = field(default_factory=MotorLimits)
    endstops: EndstopsCfg = field(default_factory=EndstopsCfg)
    forbidden_zone: ForbiddenZone = field(default_factory=ForbiddenZone)
    speed_control: SpeedControl = field(default_factory=SpeedControl)
    safety_bounds: SafetyBounds = field(default_factory=SafetyBounds)
    theme: Dict[str, Any] = field(default_factory=dict)
    camera: Dict[str, Any] = field(default_factory=dict)
    default_position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

def _require(d: Dict[str, Any], key: str, ctx: str):
    if key not in d:
        raise ValueError(f"Missing '{key}' in {ctx}")
    return d[key]

def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    data = json.loads(p.read_text())

    # Required top-level keys
    for k in ("uart", "axes"):
        _require(data, k, "config")

    # UART
    uart_dict = _require(data, "uart", "config")
    uart = UARTCfg(
        port=_require(uart_dict, "port", "uart"),
        baud=uart_dict.get("baud", 115200)
    )

    # Axes
    axes_dict = _require(data, "axes", "config")
    axes: Dict[str, AxisCfg] = {}
    for name in ("x", "y", "z"):
        if name not in axes_dict:
            raise ValueError(f"Missing axis '{name}' in axes")
        ax = axes_dict[name]
        for need in ("enable", "step", "dir", "driver_address"):
            _require(ax, need, f"axes.{name}")
        axes[name] = AxisCfg(
            enable=ax["enable"],
            step=ax["step"],
            dir=ax["dir"],
            driver_address=ax["driver_address"],
            current_ma=ax.get("current_ma", 800),
            microstepping=ax.get("microstepping", 1),
            invert_dir=ax.get("invert_dir", False),
            spreadcycle=ax.get("spreadcycle", False),
            photoelectric_switch=ax["photoelectric_switch"]
        )

    # Motion / Motor / Theme / Camera / Endstops / Default position
    motion = MotionCfg(**data.get("motion", {}))
    motor  = MotorLimits(**data.get("motor", {}))
    theme  = data.get("theme", {})
    camera = data.get("camera", {})
    endstops = EndstopsCfg(**data.get("endstops", {}))
    forbidden_zone = ForbiddenZone(**data.get("forbidden_zone", {}))
    default_position = data.get("default_position", {"x": 0, "y": 0, "z": 0})

    # Speed control
    sc_raw = data.get("speed_control", {})
    speed_control = SpeedControl(
        base_speed_fullstep=sc_raw.get("base_speed_fullstep", 300),
        base_accel_fullstep=sc_raw.get("base_accel_fullstep", 200),
        max_speed_fullstep=sc_raw.get("max_speed_fullstep", 800),
        max_accel_fullstep=sc_raw.get("max_accel_fullstep", 400),
        min_speed_fullstep=sc_raw.get("min_speed_fullstep", 3),
    )

    # Safety bounds
    sb_raw = data.get("safety_bounds", {})
    zsr_raw = sb_raw.get("z_safe_rect", {})
    xyf_raw = sb_raw.get("xy_forbidden", {})
    safety_bounds = SafetyBounds(
        z_global_max=sb_raw.get("z_global_max", 80.0),
        xy_forbidden_x=xyf_raw.get("x_min", 3800.0),
        xy_forbidden_y=xyf_raw.get("y_min", 850.0),
        z_safe_rect=ZSafeRect(
            x_min=zsr_raw.get("x_min", 4050.0),
            x_max=zsr_raw.get("x_max", 4450.0),
            y_min=zsr_raw.get("y_min", 260.0),
            y_max=zsr_raw.get("y_max", 670.0),
            z_max=zsr_raw.get("z_max", 300.0),
        ),
    )

    return AppConfig(
        cfg_path=str(path),
        uart=uart,
        axes=axes,
        motion=motion,
        motor=motor,
        endstops=endstops,
        forbidden_zone=forbidden_zone,
        speed_control=speed_control,
        safety_bounds=safety_bounds,
        theme=theme,
        camera=camera,
        default_position=default_position
    )

def save_config_atomic(cfg, path: str):
    data = asdict(cfg)
    # Convert any Path objects to strings for JSON serialization
    if isinstance(data.get("cfg_path"), Path):
        data["cfg_path"] = str(data["cfg_path"])
    dname = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", dir=dname, delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)