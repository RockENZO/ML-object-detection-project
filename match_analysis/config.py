import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    sample_hz: float = 10.0
    imgsz: int = 1280
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    calibration_hz: float = 2.0
    max_reprojection_px: float = 8.0
    prediction_seconds: float = 1.0
    possession_persistence: float = 0.5
    possession_radius_m: float = 2.5
    ambiguity_margin_m: float = 1.0
    max_ball_speed_mps: float = 40.0
    team_margin: float = 0.15
    cut_threshold: float = 0.55
    grass_fraction: float = 0.2
    min_players: int = 6
    pressure_window: float = 30.0
    replay_distance: int = 3
    replay_confirm_samples: int = 10

    @classmethod
    def load(cls, path=None):
        values = json.loads(Path(path).read_text()) if path else {}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                "Unknown configuration keys: " + ", ".join(sorted(unknown))
            )
        instance = cls(**values)
        for key, value in asdict(instance).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    "Configuration values must be finite and positive: " + key
                )
        for key in (
            "imgsz",
            "min_players",
            "replay_distance",
            "replay_confirm_samples",
        ):
            if not isinstance(getattr(instance, key), int):
                raise ValueError("Integer required: " + key)
        if (
            instance.prediction_seconds > 1
            or instance.sample_hz > 30
            or instance.calibration_hz > instance.sample_hz
        ):
            raise ValueError(
                "Prediction cap is one second; calibration rate must not exceed sample rate (max 30 Hz)"
            )
        return instance

    def as_dict(self):
        return asdict(self)
