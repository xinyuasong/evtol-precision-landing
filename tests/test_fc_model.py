"""The mock FC's failsafe is a tested behaviour: when the RC stream stops, it
levels, descends, and disarms on the ground."""

import numpy as np

from control import msp
from control.msp import Parser
from sim.dynamics import Dynamics, VehicleParams, VehicleState
from sim.fc_model import FCParams, MockFC


def rc_frame(ch):
    return msp.encode_v1(msp.MSP_SET_RAW_RC, msp.pack_rc(ch))


def test_stick_mapping():
    fc = MockFC(FCParams(), np.random.default_rng(0))
    fc.arm()
    fc.engage_override()
    fc.handle(rc_frame([1750, 1250, 1500, 1500]))
    roll, pitch, thrust = fc.commands()
    assert roll == 12.5 and pitch == -12.5  # signs per PITCH_STICK_SIGN / ROLL_STICK_SIGN
    assert thrust == FCParams().mass_kg * 9.81


def test_override_off_ignores_msp_rc():
    fc = MockFC(FCParams(), np.random.default_rng(0))
    fc.arm()
    fc.handle(rc_frame([2000, 2000, 2000, 2000]))
    roll, pitch, thrust = fc.commands()
    assert roll == 0 and pitch == 0  # pilot's neutral sticks still in force


def test_telemetry_replies_parse():
    fc = MockFC(FCParams(att_noise_deg=0.0), np.random.default_rng(0))
    fc.arm()
    fc.state = VehicleState(pos=np.array([0, 0, 1.5]), roll_deg=2.0, pitch_deg=-3.0)
    fc.alt_sensor.alt_m, fc.alt_sensor.vario_ms = 1.5, -0.2
    out = fc.handle(msp.encode_v1(msp.MSP_ATTITUDE) + msp.encode_v1(msp.MSP_ALTITUDE) + msp.encode_v1(msp.MSP_STATUS))
    frames = Parser().feed(out)
    assert msp.unpack_attitude(frames[0].payload) == (2.0, -3.0, 0.0)
    assert msp.unpack_altitude(frames[1].payload) == (1.5, -0.2)
    assert msp.unpack_status(frames[2].payload)["mode_flags"] & 1


def test_altitude_dropout_returns_error_frame():
    fc = MockFC(FCParams(), np.random.default_rng(0))
    fc.alt_sensor.alt_m = None
    frames = Parser().feed(fc.handle(msp.encode_v1(msp.MSP_ALTITUDE)))
    assert frames[0].direction == msp.DIR_ERROR


def test_failsafe_lands_when_stream_stops():
    fc = MockFC(FCParams(), np.random.default_rng(0))
    dyn = Dynamics(VehicleParams())
    dyn.reset((0, 0, 2.0))
    fc.arm()
    fc.engage_override()
    dt = 1 / 400
    t = 0.0
    # 1 s of healthy stream at 50 Hz with a slight roll command
    for i in range(400):
        if i % 8 == 0:
            fc.handle(rc_frame([1550, 1500, 1500, 1500]))
        fc.update(t, dyn.state)
        dyn.step(dt, *fc.commands(), wind_ms=(0, 0, 0))
        t += dt
    assert not fc.failsafe and dyn.state.roll_deg > 1.0
    # stream stops
    for _ in range(400 * 30):
        fc.update(t, dyn.state)
        dyn.step(dt, *fc.commands(), wind_ms=(0, 0, 0))
        t += dt
        if not fc.armed and dyn.state.on_ground:
            break
    assert fc.failsafe
    assert fc.failsafe_t is not None and fc.failsafe_t - 1.0 <= FCParams().failsafe_delay_s + 0.05
    assert fc.landed_by_failsafe and not fc.armed
    assert dyn.touchdown_pos is not None and abs(dyn.touchdown_vz) < 1.0
    assert abs(dyn.state.roll_deg) < 0.5  # levelled before landing


def test_failsafe_ignores_late_rc():
    fc = MockFC(FCParams(), np.random.default_rng(0))
    fc.arm()
    fc.engage_override()
    fc.update(1.0, VehicleState(pos=np.array([0, 0, 2.0])))
    assert fc.failsafe
    fc.handle(rc_frame([2000, 2000, 2000, 2000]))
    roll, pitch, _ = fc.commands()
    assert roll == 0 and pitch == 0
