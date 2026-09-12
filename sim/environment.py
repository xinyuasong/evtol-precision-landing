"""Disturbances and sensor imperfections. Everything here is seeded."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WindParams:
    steady_ms: tuple[float, float] = (0.0, 0.0)
    gust_sigma_ms: float = 0.0
    gust_tau_s: float = 2.0  # Dryden-style first-order coloured noise


@dataclass
class AltSensorParams:
    noise_m: float = 0.01
    vario_noise_ms: float = 0.03
    bias_m: float = 0.0
    drift_ms: float = 0.0  # bias grows at this rate (m per s)
    dropout_start_s: float | None = None
    dropout_end_s: float | None = None
    dropout_prob: float = 0.0  # per-sample random dropouts


@dataclass
class CameraParams:
    fps: float = 30.0
    latency_s: float = 0.04  # capture -> pose delivered to control
    frame_drop_prob: float = 0.0
    noise_sigma: float = 4.0  # sensor noise, grey levels
    gain: float = 1.0
    offset: float = 0.0
    exposure_s: float = 0.004  # sets motion-blur length
    rolling_shutter_shear_px: float = 0.0  # peak row shear
    vibration_hz: float = 0.0
    occlusion_start_s: float | None = None
    occlusion_end_s: float | None = None
    occlusion_below_alt_m: float | None = None  # once the vehicle descends below this, the tag is gone for good
    pad_moving_ms: tuple[float, float] = (0.0, 0.0)


@dataclass
class BatteryParams:
    v0: float = 24.6
    sag_v_per_s: float = 0.0
    sag_start_s: float | None = None


@dataclass
class EnvParams:
    wind: WindParams = field(default_factory=WindParams)
    alt: AltSensorParams = field(default_factory=AltSensorParams)
    camera: CameraParams = field(default_factory=CameraParams)
    battery: BatteryParams = field(default_factory=BatteryParams)


class Environment:
    def __init__(self, p: EnvParams, rng: np.random.Generator) -> None:
        self.p = p
        self.rng = rng
        self.gust = np.zeros(2)
        self.alt_bias = float(p.alt.bias_m)

    def wind(self, t: float, dt: float) -> np.ndarray:
        w = self.p.wind
        if w.gust_sigma_ms > 0:
            a = dt / w.gust_tau_s
            self.gust += -a * self.gust + w.gust_sigma_ms * np.sqrt(2 * a) * self.rng.normal(size=2)
        return np.array([w.steady_ms[0] + self.gust[0], w.steady_ms[1] + self.gust[1], 0.0])

    def alt_measurement(self, t: float, dt: float, true_alt: float, true_vz: float) -> tuple[float | None, float | None]:
        a = self.p.alt
        self.alt_bias += a.drift_ms * dt
        if a.dropout_start_s is not None and a.dropout_start_s <= t <= (a.dropout_end_s or 1e9):
            return None, None
        if a.dropout_prob > 0 and self.rng.random() < a.dropout_prob:
            return None, None
        return true_alt + self.alt_bias + self.rng.normal(0, a.noise_m), true_vz + self.rng.normal(0, a.vario_noise_ms)

    def vbat(self, t: float) -> float:
        b = self.p.battery
        if b.sag_start_s is not None and t > b.sag_start_s:
            return b.v0 - b.sag_v_per_s * (t - b.sag_start_s)
        return b.v0

    def occluded(self, t: float, alt_m: float | None = None) -> bool:
        c = self.p.camera
        if c.occlusion_start_s is not None and c.occlusion_start_s <= t <= (c.occlusion_end_s or 1e9):
            return True
        if c.occlusion_below_alt_m is not None and alt_m is not None:
            if alt_m < c.occlusion_below_alt_m:
                self._latched_occlusion = True
            return getattr(self, "_latched_occlusion", False)
        return False

    def frame_dropped(self) -> bool:
        return self.p.camera.frame_drop_prob > 0 and self.rng.random() < self.p.camera.frame_drop_prob

    def pad_position(self, t: float) -> np.ndarray:
        v = self.p.camera.pad_moving_ms
        return np.array([v[0] * t, v[1] * t, 0.0])
