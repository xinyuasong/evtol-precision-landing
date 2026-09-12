"""Altitude loop. The Pi owns throttle; nothing below this file holds altitude.

ASSUMES rangefinder-grade altitude. A bare barometer drifts metres in ground
effect and would make this loop unflyable near touchdown.

The altitude source is an interface so a second sensor (and fusion) can be added
without touching the controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

from control.filter import LowPass
from control.guidance import PID


@dataclass(frozen=True)
class AltitudeSample:
    alt_m: float  # positive up, above the pad/ground
    vario_ms: float  # positive up
    t_mono: float


class AltitudeSource(Protocol):
    def read(self) -> AltitudeSample | None: ...


class TelemetryAltitudeSource:
    """Single implementation for now: the FC's MSP_ALTITUDE estimate via the telemetry thread."""

    def __init__(self, telemetry) -> None:
        self._telem = telemetry

    def read(self) -> AltitudeSample | None:
        s = self._telem.altitude()
        if s is None:
            return None
        (alt, vario), t = s
        return AltitudeSample(alt, vario, t)


class VerticalMode(Enum):
    MIN_THROTTLE = auto()  # IDLE / DISARM: stick at minimum
    HOLD = auto()  # hold the setpoint captured on entry
    DESCEND = auto()  # staged, gated on horizontal error
    FINAL = auto()  # fixed open-loop descent rate
    ABORT_CLIMB = auto()  # climb to the abort altitude, then hold


def descent_rate(dcfg: dict, horiz_err_m: float | None, alt_m: float) -> float:
    """Commanded descent rate, m/s, positive down. Negative means climb."""
    if horiz_err_m is None or horiz_err_m > dcfg["climb_err_m"]:
        return -float(dcfg["climb_rate_ms"])
    if horiz_err_m > dcfg["hold_err_m"]:
        return 0.0
    rate = dcfg["rate_min_ms"] + dcfg["rate_per_m"] * max(alt_m, 0.0)
    return float(min(dcfg["rate_max_ms"], rate))


def ground_effect_feedforward(hover_us: float, alt_m: float, gecfg: dict, mid_us: float, min_us: float) -> float:
    """Ground effect increases thrust near the surface; pull the hover feedforward
    down proportionally so the loop doesn't bounce on the cushion."""
    h = float(gecfg["height_m"])
    if alt_m >= h or h <= 0:
        return hover_us
    frac = 1.0 - max(alt_m, 0.0) / h
    return hover_us - gecfg["ff_reduction"] * (hover_us - min_us) * frac


class TouchdownDetector:
    def __init__(self, tcfg: dict) -> None:
        self.alt_max = float(tcfg["alt_m"])
        self.vario_max = float(tcfg["vario_ms"])
        self.thr_max = int(tcfg["max_throttle_us"])
        self.sustain = float(tcfg["sustain_s"])
        self.stall_sustain = float(tcfg["stall_sustain_s"])
        self.reset()

    def reset(self) -> None:
        self._t_ok = 0.0
        self._t_stall = 0.0
        self.detected = False
        self.reason = ""

    def step(self, alt_m: float, vario_ms: float, throttle_us: int, descending: bool, dt: float) -> bool:
        """Two ways to be on the ground:
        - low, still, and throttle well below hover (the altimeter agrees), or
        - a descent is commanded, the vehicle is not moving, and the throttle has been
          driven well below hover for a while (the altimeter is biased and never reads
          low: the ground arrived before the number did)."""
        still_and_low_thr = abs(vario_ms) <= self.vario_max and throttle_us <= self.thr_max
        self._t_ok = self._t_ok + dt if (still_and_low_thr and alt_m <= self.alt_max) else 0.0
        self._t_stall = self._t_stall + dt if (still_and_low_thr and descending) else 0.0
        if self._t_ok >= self.sustain:
            self.detected, self.reason = True, "low, still, low throttle"
        elif self._t_stall >= self.stall_sustain:
            self.detected, self.reason = True, "descent stalled at low throttle (altimeter bias?)"
        else:
            self.detected = False
        return self.detected


@dataclass
class VerticalCommand:
    throttle_us: int
    alt_sp_m: float
    rate_cmd_ms: float  # positive down
    pid_terms: tuple[float, float, float]
    alt_filt_m: float
    vario_filt_ms: float


class AltitudeController:
    def __init__(self, acfg: dict, rc_cfg: dict) -> None:
        self.cfg = acfg
        self.pid = PID.from_cfg(acfg["pid"])
        self.hover_us = float(acfg["hover_throttle_us"])
        self.authority = float(acfg["throttle_authority"])
        self.range_us = float(rc_cfg["range_us"])
        self.mid_us = float(rc_cfg["mid_us"])
        self.min_us = float(rc_cfg["min_us"])
        self.max_us = float(rc_cfg["max_us"])
        self.lp_alt = LowPass(acfg["lowpass_tau_s"])
        self.lp_vario = LowPass(acfg["lowpass_tau_s"])
        self.touchdown = TouchdownDetector(acfg["touchdown"])
        self.alt_sp: float | None = None
        self._mode: VerticalMode | None = None
        self._last_throttle = int(self.min_us)

    def reset(self) -> None:
        self.pid.reset()
        self.lp_alt.reset()
        self.lp_vario.reset()
        self.touchdown.reset()
        self.alt_sp = None
        self._mode = None
        self._last_throttle = int(self.min_us)

    def step(self, sample: AltitudeSample | None, mode: VerticalMode, horiz_err_m: float | None, dt: float) -> VerticalCommand | None:
        """Returns None when there is no altitude to act on and the mode needs one.
        MIN_THROTTLE never needs a sample."""
        if mode is VerticalMode.MIN_THROTTLE:
            self.alt_sp = None
            self._mode = mode
            self.pid.reset()
            self._last_throttle = int(self.min_us)
            return VerticalCommand(int(self.min_us), 0.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0)

        if sample is None:
            return None

        alt = self.lp_alt.step(sample.alt_m, dt)
        vario = self.lp_vario.step(sample.vario_ms, dt)

        entering = mode is not self._mode
        self._mode = mode
        if self.alt_sp is None or (entering and mode in (VerticalMode.HOLD, VerticalMode.DESCEND)):
            self.alt_sp = alt  # capture current altitude, no step input

        if mode is VerticalMode.HOLD:
            rate = 0.0
        elif mode is VerticalMode.DESCEND:
            rate = descent_rate(self.cfg["descent"], horiz_err_m, alt)
        elif mode is VerticalMode.FINAL:
            rate = float(self.cfg["final"]["descent_rate_ms"])
        elif mode is VerticalMode.ABORT_CLIMB:
            target = float(self.cfg["abort"]["climb_to_m"])
            rate = -float(self.cfg["abort"]["climb_rate_ms"]) if self.alt_sp < target else 0.0
        else:
            raise ValueError(mode)

        floor = float(self.cfg["final"]["setpoint_floor_m"]) if mode is VerticalMode.FINAL else 0.0
        self.alt_sp = max(floor, self.alt_sp - rate * dt)
        err = self.alt_sp - alt
        out = self.pid.step(err, dt, d_err=-vario)

        ff = ground_effect_feedforward(self.hover_us, alt, self.cfg["ground_effect"], self.mid_us, self.min_us)
        thr = ff + out * self.range_us * self.authority
        thr_i = int(max(self.min_us, min(self.max_us, round(thr))))
        self._last_throttle = thr_i
        return VerticalCommand(thr_i, self.alt_sp, rate, self.pid.last_terms, alt, vario)

    def touchdown_step(self, dt: float) -> bool:
        if self.lp_alt.y is None:
            return False
        descending = self._mode in (VerticalMode.DESCEND, VerticalMode.FINAL)
        return self.touchdown.step(self.lp_alt.y, self.lp_vario.y or 0.0, self._last_throttle, descending, dt)
