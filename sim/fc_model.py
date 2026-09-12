"""Mock flight controller. Speaks real MSP over any byte channel.

- Parses MSP_SET_RAW_RC; maps roll/pitch sticks to commanded lean angle and the
  throttle stick to thrust (the Pi owns altitude, so throttle is direct).
- Serves MSP_ATTITUDE / MSP_ALTITUDE / MSP_STATUS / MSP_ANALOG from the sim state
  through the FC's own sensor models.
- Implements the RC-timeout failsafe: if the RC stream stops for failsafe_delay_s
  while armed, the FC ignores MSP RC, levels the aircraft, descends, and disarms
  on touchdown. This is tested behaviour, not decoration.

The same `handle(bytes) -> bytes` serves a pty (sim/harness.py) and the in-process
fast path; only the wire differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control import msp
from control.msp import DIR_ERROR, DIR_FROM_FC, Parser
from sim.dynamics import G, VehicleState

# STICK SIGN CONVENTION — the single place this is defined in the simulator.
#
# Betaflight / INAV (ANGLE mode): rcCommand[axis] = rcData[axis] - 1500, and a positive
# rcCommand targets a positive angle in the IMU convention, where positive pitch is nose
# UP and positive roll is right wing DOWN. So:
#     pitch channel > 1500  ->  nose up   ->  the vehicle accelerates -X (backward)
#     roll  channel > 1500  ->  roll right ->  the vehicle accelerates +Y (right)
# This is a reading of the firmware source, NOT a bench-verified fact. It is asserted by
# tests/test_rc_convention.py so that a 60-second props-off test on real hardware can
# overturn it: flip PITCH_STICK_SIGN here and rc.pitch_sign in config/vehicle.yaml.
PITCH_STICK_SIGN = +1.0  # +1: channel above mid = positive (nose-up) pitch angle
ROLL_STICK_SIGN = +1.0  # +1: channel above mid = positive (right-wing-down) roll angle


@dataclass
class FCParams:
    lean_max_deg: float = 25.0  # full stick in ANGLE mode
    rc_map: str = "AETR"
    rc_min: int = 1000
    rc_mid: int = 1500
    rc_max: int = 2000
    hover_us: int = 1500  # thrust == weight at this throttle (matches config unless a scenario says otherwise)
    thrust_max_n_per_kg: float = 20.0  # thrust/weight ~2 at full stick
    mass_kg: float = 2.0
    failsafe_delay_s: float = 0.5
    failsafe_descent_ms: float = 0.4
    failsafe_kp: float = 8.0
    failsafe_disarm_alt_m: float = 0.03  # on the ground, not on the cushion
    att_noise_deg: float = 0.2
    cycle_us: int = 250
    override_channels_mask: int = 0b1111


@dataclass
class AltSensorModel:
    """What the FC believes about altitude. Set by the environment each tick."""

    alt_m: float | None = None
    vario_ms: float | None = None


class MockFC:
    def __init__(self, params: FCParams, rng: np.random.Generator) -> None:
        self.p = params
        self.rng = rng
        self.parser = Parser()
        self.rc = [params.rc_mid, params.rc_mid, params.rc_min, params.rc_mid] + [params.rc_mid] * 4
        self.t_last_rc: float | None = None
        self.t = 0.0
        self.armed = False
        self.override = False
        self.failsafe = False
        self.failsafe_t: float | None = None
        self.vbat_v = 24.6
        self.alt_sensor = AltSensorModel()
        self.state: VehicleState = VehicleState()
        self.rc_frames = 0
        self.requests: dict[int, int] = {}
        self.landed_by_failsafe = False

    # ---- operator actions ----
    def arm(self) -> None:
        """Pilot arms and hovers on the real radio: neutral sticks, hover throttle."""
        self.armed = True
        self.override = False
        self.failsafe = False
        self.rc = [self.p.rc_mid, self.p.rc_mid, self.p.hover_us, self.p.rc_mid] + [self.p.rc_mid] * 4

    def engage_override(self) -> None:
        """Pilot flips the MSP-override AUX switch. The stream is expected from now."""
        self.override = True
        self.t_last_rc = self.t

    def release_override(self) -> None:
        self.override = False
        self.failsafe = False
        self.rc = [self.p.rc_mid, self.p.rc_mid, self.p.hover_us, self.p.rc_mid] + [self.p.rc_mid] * 4

    def disarm(self) -> None:
        self.armed = False
        self.override = False

    # ---- sim tick ----
    def update(self, t: float, state: VehicleState) -> None:
        self.t = t
        self.state = state
        if self.armed and self.override and not self.failsafe and self.t_last_rc is not None:
            if t - self.t_last_rc > self.p.failsafe_delay_s:
                self.failsafe = True
                self.failsafe_t = t
        if self.failsafe and self.armed and state.alt <= self.p.failsafe_disarm_alt_m and abs(state.vel[2]) < 0.3:
            self.disarm()
            self.landed_by_failsafe = True

    def commands(self) -> tuple[float, float, float]:
        """(roll_cmd_deg, pitch_cmd_deg, thrust_n) for the dynamics."""
        p = self.p
        if not self.armed:
            return 0.0, 0.0, 0.0
        if self.failsafe:
            vz_err = -p.failsafe_descent_ms - float(self.state.vel[2])
            thrust = p.mass_kg * G * (1.0 + p.failsafe_kp * vz_err / G)
            return 0.0, 0.0, max(0.0, thrust)
        m = p.rc_map
        roll_us, pitch_us, thr_us = self.rc[m.index("A")], self.rc[m.index("E")], self.rc[m.index("T")]
        rng_us = p.rc_max - p.rc_mid
        roll_cmd = ROLL_STICK_SIGN * (roll_us - p.rc_mid) / rng_us * p.lean_max_deg
        pitch_cmd = PITCH_STICK_SIGN * (pitch_us - p.rc_mid) / rng_us * p.lean_max_deg
        # throttle: linear, thrust == weight at hover_us, thrust_max at rc_max
        weight = p.mass_kg * G
        if thr_us <= p.hover_us:
            thrust = weight * (thr_us - p.rc_min) / max(1, p.hover_us - p.rc_min)
        else:
            tmax = p.thrust_max_n_per_kg * p.mass_kg
            thrust = weight + (tmax - weight) * (thr_us - p.hover_us) / max(1, p.rc_max - p.hover_us)
        return roll_cmd, pitch_cmd, max(0.0, thrust)

    # ---- MSP ----
    def handle(self, data: bytes) -> bytes:
        out = b""
        for f in self.parser.feed(data):
            self.requests[f.cmd] = self.requests.get(f.cmd, 0) + 1
            if f.cmd == msp.MSP_SET_RAW_RC:
                ch = msp.unpack_rc(f.payload)
                if self.override and not self.failsafe:
                    for i, v in enumerate(ch[: len(self.rc)]):
                        if i >= 4 or (self.p.override_channels_mask >> i) & 1:
                            self.rc[i] = v
                self.t_last_rc = self.t
                self.rc_frames += 1
            elif f.cmd == msp.MSP_ATTITUDE:
                n = self.p.att_noise_deg
                out += msp.encode_v1(
                    f.cmd,
                    msp.pack_attitude(self.state.roll_deg + self.rng.normal(0, n), self.state.pitch_deg + self.rng.normal(0, n), self.state.yaw_deg),
                    DIR_FROM_FC,
                )
            elif f.cmd == msp.MSP_ALTITUDE:
                a = self.alt_sensor
                if a.alt_m is None:
                    out += msp.encode_v1(f.cmd, b"", DIR_ERROR)
                else:
                    out += msp.encode_v1(f.cmd, msp.pack_altitude(a.alt_m, a.vario_ms or 0.0), DIR_FROM_FC)
            elif f.cmd == msp.MSP_STATUS:
                flags = (1 if self.armed else 0) | ((1 if self.override else 0) << 1) | ((1 if self.failsafe else 0) << 2)
                out += msp.encode_v1(f.cmd, msp.pack_status(self.p.cycle_us, 0, 0b111, flags, 0), DIR_FROM_FC)
            elif f.cmd == msp.MSP_ANALOG:
                out += msp.encode_v1(f.cmd, msp.pack_analog(self.vbat_v, 0, 99, 10.0), DIR_FROM_FC)
            elif f.cmd == msp.MSP_RC:
                out += msp.encode_v1(f.cmd, msp.pack_rc(self.rc), DIR_FROM_FC)
            else:
                out += msp.encode_v1(f.cmd, b"", DIR_ERROR)
        return out
