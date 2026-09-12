import pytest

from control.altitude import (
    AltitudeController,
    AltitudeSample,
    TelemetryAltitudeSource,
    TouchdownDetector,
    VerticalMode,
    descent_rate,
    ground_effect_feedforward,
)


@pytest.fixture
def dcfg(vcfg):
    return vcfg["altitude"]["descent"]


def test_descent_rate_gating(dcfg):
    assert descent_rate(dcfg, 1.0, 3.0) == -dcfg["climb_rate_ms"]  # far off: climb
    assert descent_rate(dcfg, None, 3.0) == -dcfg["climb_rate_ms"]  # no error known: climb
    assert descent_rate(dcfg, 0.4, 3.0) == 0.0  # off-centre: hold
    r_high = descent_rate(dcfg, 0.05, 3.0)
    r_low = descent_rate(dcfg, 0.05, 0.5)
    assert 0 < r_low < r_high <= dcfg["rate_max_ms"]
    assert descent_rate(dcfg, 0.05, 50.0) == dcfg["rate_max_ms"]


def test_ground_effect_feedforward(vcfg):
    ge = vcfg["altitude"]["ground_effect"]
    assert ground_effect_feedforward(1500, 1.0, ge, 1500, 1000) == 1500
    at_zero = ground_effect_feedforward(1500, 0.0, ge, 1500, 1000)
    assert at_zero == pytest.approx(1500 - ge["ff_reduction"] * 500)
    mid = ground_effect_feedforward(1500, ge["height_m"] / 2, ge, 1500, 1000)
    assert at_zero < mid < 1500


def test_touchdown_requires_sustain_low_still_and_low_throttle(vcfg):
    c = vcfg["altitude"]["touchdown"]
    td = TouchdownDetector(c)
    low_thr = c["max_throttle_us"] - 10
    for _ in range(20):
        assert not td.step(0.02, 0.0, low_thr, False, 0.02)  # 0.4 s: not yet
    assert td.step(0.02, 0.0, low_thr, False, 0.11)  # crosses 0.5 s
    td.reset()
    for _ in range(100):
        assert not td.step(0.02, 0.3, low_thr, True, 0.02)  # low but still moving
    for _ in range(100):
        assert not td.step(0.5, 0.0, low_thr, False, 0.02)  # still, but not low, not descending
    for _ in range(100):
        assert not td.step(0.02, 0.0, vcfg["altitude"]["hover_throttle_us"], True, 0.02)  # low and still, but hovering on the cushion


def test_touchdown_stalled_descent_despite_biased_altimeter(vcfg):
    c = vcfg["altitude"]["touchdown"]
    td = TouchdownDetector(c)
    low_thr = c["max_throttle_us"] - 10
    n = int(c["stall_sustain_s"] / 0.02)
    for _ in range(n - 2):
        assert not td.step(0.30, 0.0, low_thr, True, 0.02)  # altimeter says 30 cm, but nothing moves
    assert td.step(0.30, 0.0, low_thr, True, 0.02 * 4)
    assert "stall" in td.reason


def test_min_throttle_needs_no_sample(vcfg):
    ac = AltitudeController(vcfg["altitude"], vcfg["rc"])
    cmd = ac.step(None, VerticalMode.MIN_THROTTLE, None, 0.02)
    assert cmd.throttle_us == vcfg["rc"]["min_us"]


def test_no_sample_no_command(vcfg):
    ac = AltitudeController(vcfg["altitude"], vcfg["rc"])
    assert ac.step(None, VerticalMode.HOLD, None, 0.02) is None


def test_hold_captures_current_altitude_without_step(vcfg):
    ac = AltitudeController(vcfg["altitude"], vcfg["rc"])
    cmd = ac.step(AltitudeSample(2.5, 0.0, 0.0), VerticalMode.HOLD, None, 0.02)
    assert cmd.alt_sp_m == pytest.approx(2.5)
    assert cmd.throttle_us == vcfg["altitude"]["hover_throttle_us"]  # zero error: hover feedforward


def test_hold_reacts_to_altitude_error(vcfg):
    ac = AltitudeController(vcfg["altitude"], vcfg["rc"])
    ac.step(AltitudeSample(2.5, 0.0, 0.0), VerticalMode.HOLD, None, 0.02)
    low = ac.step(AltitudeSample(2.0, 0.0, 0.02), VerticalMode.HOLD, None, 0.02)
    assert low.throttle_us > vcfg["altitude"]["hover_throttle_us"]
    ac2 = AltitudeController(vcfg["altitude"], vcfg["rc"])
    ac2.step(AltitudeSample(2.5, 0.0, 0.0), VerticalMode.HOLD, None, 0.02)
    high = ac2.step(AltitudeSample(3.0, 0.0, 0.02), VerticalMode.HOLD, None, 0.02)
    assert high.throttle_us < vcfg["altitude"]["hover_throttle_us"]


def test_descend_lowers_setpoint_only_when_centred(vcfg):
    ac = AltitudeController(vcfg["altitude"], vcfg["rc"])
    ac.step(AltitudeSample(3.0, 0.0, 0.0), VerticalMode.DESCEND, 0.05, 0.02)
    sp0 = ac.alt_sp
    for _ in range(50):
        ac.step(AltitudeSample(3.0, 0.0, 0.0), VerticalMode.DESCEND, 0.05, 0.02)
    assert ac.alt_sp < sp0 - 0.2
    sp1 = ac.alt_sp
    for _ in range(50):
        ac.step(AltitudeSample(3.0, 0.0, 0.0), VerticalMode.DESCEND, 0.4, 0.02)  # off-centre: hold
    assert ac.alt_sp == pytest.approx(sp1)
    for _ in range(50):
        ac.step(AltitudeSample(3.0, 0.0, 0.0), VerticalMode.DESCEND, 1.0, 0.02)  # lost it: climb
    assert ac.alt_sp > sp1 + 0.1


def test_final_descends_regardless_of_error(vcfg):
    ac = AltitudeController(vcfg["altitude"], vcfg["rc"])
    ac.step(AltitudeSample(0.3, 0.0, 0.0), VerticalMode.HOLD, None, 0.02)
    sp0 = ac.alt_sp
    for _ in range(50):
        cmd = ac.step(AltitudeSample(0.3, 0.0, 0.0), VerticalMode.FINAL, 5.0, 0.02)
    assert cmd.alt_sp_m < sp0
    assert cmd.rate_cmd_ms == vcfg["altitude"]["final"]["descent_rate_ms"]


def test_abort_climbs_to_target_then_holds(vcfg):
    ac = AltitudeController(vcfg["altitude"], vcfg["rc"])
    target = vcfg["altitude"]["abort"]["climb_to_m"]
    ac.step(AltitudeSample(0.8, 0.0, 0.0), VerticalMode.HOLD, None, 0.02)
    for _ in range(50):
        cmd = ac.step(AltitudeSample(0.8, 0.0, 0.0), VerticalMode.ABORT_CLIMB, None, 0.02)
    assert cmd.rate_cmd_ms < 0
    for _ in range(2000):
        cmd = ac.step(AltitudeSample(cmd.alt_sp_m, 0.0, 0.0), VerticalMode.ABORT_CLIMB, None, 0.02)
    assert cmd.alt_sp_m == pytest.approx(target, abs=0.02)
    assert cmd.rate_cmd_ms == 0.0


def test_setpoint_floor(vcfg):
    ac = AltitudeController(vcfg["altitude"], vcfg["rc"])
    for _ in range(500):
        cmd = ac.step(AltitudeSample(0.05, 0.0, 0.0), VerticalMode.FINAL, 0.0, 0.02)
    assert cmd.alt_sp_m == pytest.approx(vcfg["altitude"]["final"]["setpoint_floor_m"])
    ac2 = AltitudeController(vcfg["altitude"], vcfg["rc"])
    for _ in range(500):
        cmd = ac2.step(AltitudeSample(0.05, 0.0, 0.0), VerticalMode.DESCEND, 0.0, 0.02)
    assert cmd.alt_sp_m == 0.0  # outside FINAL the floor is the ground


def test_throttle_clamped_to_rc_range(vcfg):
    ac = AltitudeController(vcfg["altitude"], vcfg["rc"])
    ac.step(AltitudeSample(5.0, 0.0, 0.0), VerticalMode.HOLD, None, 0.02)
    cmd = ac.step(AltitudeSample(0.0, -3.0, 0.0), VerticalMode.HOLD, None, 0.02)
    assert vcfg["rc"]["min_us"] <= cmd.throttle_us <= vcfg["rc"]["max_us"]


def test_telemetry_source_wraps_latest():
    class T:
        def altitude(self):
            return ((1.25, -0.1), 42.0)

    s = TelemetryAltitudeSource(T()).read()
    assert s == AltitudeSample(1.25, -0.1, 42.0)

    class Empty:
        def altitude(self):
            return None

    assert TelemetryAltitudeSource(Empty()).read() is None
