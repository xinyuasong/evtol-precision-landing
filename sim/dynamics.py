"""Translational multirotor dynamics with first-order attitude response.

World frame: X forward, Y right, Z up. Pad centre at the origin, on the ground.
Vehicle yaw is held at zero, so the vehicle's level frame is the world frame
with Z flipped. Attitude convention matches control/frames.py: pitch positive =
nose up, roll positive = right wing down.

Stick -> commanded lean angle -> first-order lag -> horizontal acceleration.
Throttle -> thrust (the mock FC maps the channel; this takes newtons).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 9.81


@dataclass
class VehicleParams:
    mass_kg: float = 2.0
    att_tau_s: float = 0.12  # first-order lean-angle response
    # quadratic drag, N/(m/s)^2: 0.5 * rho * A * Cd with A ~0.05 m^2, Cd ~1 -> ~0.03-0.06.
    # (0.30 was a 10x overestimate: 5 m/s of wind became 3.75 m/s^2, beyond any authority.)
    drag_xy: float = 0.05
    drag_z: float = 0.10
    rotor_radius_m: float = 0.15
    ground_effect_gain: float = 1.0  # multiplier on the Cheeseman-Bennett term; 0 disables
    ground_effect_max: float = 1.25  # cap on thrust augmentation
    touchdown_vz_max_ms: float = 1.0  # harder than this counts as a crash


@dataclass
class VehicleState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world, m
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world, m/s
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_rate_dps: float = 0.0
    pitch_rate_dps: float = 0.0
    on_ground: bool = True
    t: float = 0.0

    @property
    def alt(self) -> float:
        return float(self.pos[2])


class Dynamics:
    def __init__(self, params: VehicleParams) -> None:
        self.p = params
        self.state = VehicleState()
        self.touchdown_pos: np.ndarray | None = None
        self.touchdown_vz: float | None = None
        self.touchdown_t: float | None = None
        self.max_alt = 0.0

    def reset(self, pos, vel=(0, 0, 0), t=0.0) -> None:
        self.state = VehicleState(pos=np.array(pos, float), vel=np.array(vel, float), t=t)
        self.state.on_ground = self.state.pos[2] <= 0.0
        self.touchdown_pos = self.touchdown_vz = self.touchdown_t = None
        self.max_alt = self.state.alt

    def ground_effect_factor(self, alt_m: float) -> float:
        if self.p.ground_effect_gain <= 0:
            return 1.0
        h = max(alt_m, 0.02)
        r = self.p.rotor_radius_m
        term = (r / (4 * h)) ** 2 * self.p.ground_effect_gain
        return float(min(self.p.ground_effect_max, 1.0 / max(1e-3, 1.0 - term)))

    def step(self, dt: float, roll_cmd_deg: float, pitch_cmd_deg: float, thrust_n: float, wind_ms) -> VehicleState:
        s, p = self.state, self.p
        # attitude lag
        a = dt / (p.att_tau_s + dt)
        new_roll = s.roll_deg + a * (roll_cmd_deg - s.roll_deg)
        new_pitch = s.pitch_deg + a * (pitch_cmd_deg - s.pitch_deg)
        s.roll_rate_dps = (new_roll - s.roll_deg) / dt
        s.pitch_rate_dps = (new_pitch - s.pitch_deg) / dt
        s.roll_deg, s.pitch_deg = new_roll, new_pitch

        phi, theta = math.radians(s.roll_deg), math.radians(s.pitch_deg)
        T = thrust_n * self.ground_effect_factor(s.alt)
        # thrust direction in world frame (yaw = 0): body -Z rotated by R_y(pitch) R_x(roll)
        ax = -T / p.mass_kg * math.sin(theta) * math.cos(phi)  # nose up -> thrust tilts back
        ay = T / p.mass_kg * math.sin(phi)  # right wing down -> thrust tilts right
        az = T / p.mass_kg * math.cos(theta) * math.cos(phi) - G

        rel = s.vel - np.asarray(wind_ms, float)
        drag = np.array([-p.drag_xy * rel[0] * abs(rel[0]), -p.drag_xy * rel[1] * abs(rel[1]), -p.drag_z * rel[2] * abs(rel[2])]) / p.mass_kg
        acc = np.array([ax, ay, az]) + drag

        if s.on_ground and acc[2] <= 0:
            s.vel[:] = 0.0
            s.pos[2] = 0.0
        else:
            s.on_ground = False
            s.vel = s.vel + acc * dt
            s.pos = s.pos + s.vel * dt
            if s.pos[2] <= 0.0:
                if self.touchdown_pos is None:
                    self.touchdown_pos = s.pos[:2].copy()
                    self.touchdown_vz = float(s.vel[2])
                    self.touchdown_t = s.t + dt
                s.pos[2] = 0.0
                s.vel[:] = 0.0
                s.on_ground = True
        s.t += dt
        self.max_alt = max(self.max_alt, s.alt)
        return s
