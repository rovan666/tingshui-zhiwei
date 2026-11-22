# -*- coding: utf-8 -*-
"""16 通道线阵的统一配置；水中和空气参数始终分开。"""
from dataclasses import asdict, dataclass, field as dc_field
import json
from pathlib import Path

import numpy as np


@dataclass
class ArrayConfig:
    n_channels: int = 16
    spacing_m: float = 0.05
    reference_channel: int = 7
    fs: int = 100_000
    sound_speed: float = 1500.0

    def __post_init__(self):
        if self.n_channels != 16:
            raise ValueError("本方案固定为 16 通道")
        if not 0 <= self.reference_channel < self.n_channels:
            raise ValueError("参考通道越界")
        if self.spacing_m <= 0 or self.fs <= 0 or self.sound_speed <= 0:
            raise ValueError("间距、采样率和声速必须为正")

    def positions(self) -> np.ndarray:
        x = (np.arange(self.n_channels) - 7.5) * self.spacing_m
        return np.column_stack([x, np.zeros(self.n_channels)])


@dataclass
class FieldConfig:
    length_m: float = 4.0
    width_m: float = 3.0
    angle_min_deg: float = -60.0
    angle_max_deg: float = 60.0
    angle_step_deg: float = 1.0
    range_min_m: float = 0.5
    range_step_m: float = 0.25


@dataclass
class FeatureConfig:
    n_fft: int = 2048
    hop: int = 512
    n_mels: int = 128
    fmin: float = 1000.0
    fmax: float = 6000.0
    window_samples: int = 100_000
    update_samples: int = 25_000
    n_doa_bands: int = 8


@dataclass
class SystemConfig:
    config_id: str
    medium: str
    array: ArrayConfig = dc_field(default_factory=ArrayConfig)
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    features: FeatureConfig = dc_field(default_factory=FeatureConfig)


def default_water_config() -> SystemConfig:
    return SystemConfig(config_id="water_ula16_v1", medium="water")


def doa_fmax(config: SystemConfig) -> float:
    a = config.array
    return float(min(config.features.fmax, a.sound_speed / (2 * a.spacing_m), a.fs / 2))


def save_config(config: SystemConfig, path) -> None:
    Path(path).write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path) -> SystemConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return SystemConfig(
        config_id=raw["config_id"], medium=raw["medium"],
        array=ArrayConfig(**raw["array"]), field=FieldConfig(**raw["field"]),
        features=FeatureConfig(**raw["features"]),
    )
