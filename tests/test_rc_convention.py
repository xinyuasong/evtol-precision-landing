"""Falsifiable stick-sign convention. The whole stack (control/ and the mock FC) shares
this assumption, so a shared error would be invisible to every scenario. This test pins
the mock FC's half of it to a physical statement, and pins control/ to the opposite sign
so the two agree. A props-off bench test on real hardware overturns it in two lines:
PITCH_STICK_SIGN in sim/fc_model.py and rc.pitch_sign in config/vehicle.yaml."""

import numpy as np

from control import msp
from control.guidance import RCSynth
from sim.dynamics import Dynamics, VehicleParams
from sim.fc_model import PITCH_STICK_SIGN, ROLL_STICK_SIGN, FCParams, MockFC


def fly(channels, seconds=1.0):
    fc = MockFC(FCParams(), np.random.default_rng(0))
    fc.arm()
    fc.engage_override()
    dyn = Dynamics(VehicleParams(drag_xy=0.0))
    dyn.reset((0, 0, 5.0))
    dt = 1 / 400
    t = 0.0
    for i in range(int(seconds / dt)):
        if i % 8 == 0:
            fc.handle(msp.encode_v1(msp.MSP_SET_RAW_RC, msp.pack_rc(channels)))
        fc.update(t, dyn.state)
        dyn.step(dt, *fc.commands(), wind_ms=(0, 0, 0))
        t += dt
    return dyn.state


def test_pitch_channel_1700_accelerates_minus_x():
    """Betaflight/INAV: channel above mid -> positive (nose-up) pitch -> backward."""
    s = fly([1500, 1700, 1500, 1500])  # AETR
    assert s.pitch_deg > 5.0
    assert s.vel[0] * PITCH_STICK_SIGN < -0.5 and abs(s.vel[1]) < 0.05


def test_roll_channel_1700_accelerates_plus_y():
    s = fly([1700, 1500, 1500, 1500])
    assert s.roll_deg > 5.0
    assert s.vel[1] * ROLL_STICK_SIGN > 0.5 and abs(s.vel[0]) < 0.05


def test_control_and_mock_fc_agree(vcfg):
    """control/ wants to move +X: the stick it sends must make the mock FC accelerate +X."""
    rc = RCSynth(vcfg["rc"], authority=1.0)
    fwd = rc.channels(out_x=0.4, out_y=0.0, throttle_us=1500)
    s = fly(fwd)
    assert s.vel[0] > 0.5, "control's 'forward' stick flies backward on the mock FC — signs disagree"
    right = rc.channels(out_x=0.0, out_y=0.4, throttle_us=1500)
    s = fly(right)
    assert s.vel[1] > 0.5
