"""Outer-loop guidance: horizontal PIDs and RC channel synthesis.

The FC runs attitude stabilisation in ANGLE mode. Stick deflection commands a
lean angle. This module turns a position error into a stick position. That's it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PID:
    def __init__(self, kp: float, ki: float, kd: float, i_limit: float, out_limit: float, d_tau_s: float) -> None:
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.i_limit, self.out_limit = float(i_limit), float(out_limit)
        self.d_tau = float(d_tau_s)
        self.reset()

    @classmethod
    def from_cfg(cls, c: dict) -> PID:
        return cls(c["kp"], c["ki"], c["kd"], c["i_limit"], c["out_limit"], c["d_tau_s"])

    def reset(self) -> None:
        self.i = 0.0
        self.d_filt = 0.0
        self.prev_err: float | None = None
        self.last_terms = (0.0, 0.0, 0.0)

    def step(self, err: float, dt: float, d_err: float | None = None) -> float:
        """d_err: measured derivative of the error, if a filter provides one.
        Preferred over differencing, which amplifies noise and reacts to setpoint steps."""
        if dt <= 0:
            return 0.0
        p = self.kp * err

        self.i += err * dt
        self.i = float(np.clip(self.i, -self.i_limit, self.i_limit))
        i = self.ki * self.i

        if d_err is not None:
            raw_d = d_err
        elif self.prev_err is not None:
            raw_d = (err - self.prev_err) / dt
        else:
            raw_d = 0.0
        alpha = dt / (self.d_tau + dt)
        self.d_filt += alpha * (raw_d - self.d_filt)
        d = self.kd * self.d_filt
        self.prev_err = err

        out = p + i + d
        if abs(out) > self.out_limit:
            # anti-windup: undo this tick's integration when saturated
            self.i -= err * dt
            self.i = float(np.clip(self.i, -self.i_limit, self.i_limit))
            i = self.ki * self.i
            out = float(np.clip(p + i + d, -self.out_limit, self.out_limit))
        self.last_terms = (p, i, d)
        return out


@dataclass
class HorizontalCommand:
    out_x: float  # normalised -1..1, +forward
    out_y: float  # normalised -1..1, +right
    out_yaw: float


class HorizontalGuidance:
    def __init__(self, cfg: dict) -> None:
        self.pid_x = PID.from_cfg(cfg["x"])
        self.pid_y = PID.from_cfg(cfg["y"])
        self.yaw_enabled = bool(cfg["yaw"]["enabled"])
        self.pid_yaw = PID.from_cfg(cfg["yaw"]["pid"])

    def reset(self) -> None:
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_yaw.reset()

    def step(self, err_x: float, err_y: float, dt: float, derr_x=None, derr_y=None, yaw_err=0.0) -> HorizontalCommand:
        ox = self.pid_x.step(err_x, dt, derr_x)
        oy = self.pid_y.step(err_y, dt, derr_y)
        oyaw = self.pid_yaw.step(yaw_err, dt) if self.yaw_enabled else 0.0
        return HorizontalCommand(ox, oy, oyaw)


class RCSynth:
    """Maps normalised controller outputs to µs stick values in the FC's channel order."""

    def __init__(self, rc_cfg: dict, authority: float, stick_scale: float | None = None) -> None:
        self.mid = int(rc_cfg["mid_us"])
        self.rng = int(rc_cfg["range_us"])
        self.min = int(rc_cfg["min_us"])
        self.max = int(rc_cfg["max_us"])
        self.map = str(rc_cfg["map"]).upper()
        self.roll_sign = int(rc_cfg["roll_sign"])
        self.pitch_sign = int(rc_cfg["pitch_sign"])
        self.yaw_sign = int(rc_cfg["yaw_sign"])
        self.aux = [int(a) for a in rc_cfg["aux_channels"]]
        self.authority = float(authority)
        self.scale = float(stick_scale) if stick_scale is not None else self.authority

    def stick(self, norm: float, authority: float | None = None) -> int:
        """norm * stick_scale sets the deflection (loop gain lives here); authority only clips it."""
        a = self.authority if authority is None else authority
        norm = max(-1.0, min(1.0, float(norm)))
        us = norm * self.rng * self.scale
        lim = self.rng * a
        return int(round(self.mid + max(-lim, min(lim, us))))

    def clamp(self, us: float) -> int:
        return int(max(self.min, min(self.max, round(us))))

    def neutral(self) -> list[int]:
        return self.channels(0.0, 0.0, self.min, 0.0)

    def channels(self, out_x: float, out_y: float, throttle_us: float, out_yaw: float = 0.0) -> list[int]:
        """Assemble the first 4 channels per map, then aux."""
        vals = {
            "A": self.stick(self.roll_sign * out_y),
            "E": self.stick(self.pitch_sign * out_x),
            "T": self.clamp(throttle_us),
            "R": self.stick(self.yaw_sign * out_yaw),
        }
        return [vals[c] for c in self.map] + list(self.aux)
