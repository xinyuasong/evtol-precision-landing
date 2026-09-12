import pytest

from control.guidance import PID, HorizontalGuidance, RCSynth


def test_pid_proportional_only():
    pid = PID(kp=2.0, ki=0.0, kd=0.0, i_limit=1, out_limit=10, d_tau_s=0.05)
    assert pid.step(0.5, 0.02) == pytest.approx(1.0)


def test_pid_anti_windup_bounds_integrator_under_saturation():
    pid = PID(kp=1.0, ki=1.0, kd=0.0, i_limit=0.3, out_limit=1.0, d_tau_s=0.05)
    for _ in range(500):
        out = pid.step(5.0, 0.02)  # huge persistent error
    assert out == 1.0
    assert abs(pid.i) <= 0.3 + 1e-9
    # after the error flips sign, the integrator must not hold the output saturated
    outs = [pid.step(-5.0, 0.02) for _ in range(5)]
    assert outs[-1] == -1.0


def test_pid_uses_measured_derivative_when_given():
    pid = PID(kp=0.0, ki=0.0, kd=1.0, i_limit=1, out_limit=10, d_tau_s=0.0)
    out = pid.step(0.0, 0.02, d_err=-0.7)
    assert out == pytest.approx(-0.7)


def test_pid_derivative_filter_smooths():
    pid = PID(kp=0.0, ki=0.0, kd=1.0, i_limit=1, out_limit=10, d_tau_s=0.2)
    out = pid.step(0.0, 0.02, d_err=1.0)
    assert 0 < out < 0.2  # first sample heavily filtered
    for _ in range(200):
        out = pid.step(0.0, 0.02, d_err=1.0)
    assert out == pytest.approx(1.0, abs=1e-3)


def test_rc_synth_aetr_and_signs(vcfg):
    rc = RCSynth(vcfg["rc"], authority=0.2)
    ch = rc.channels(out_x=1.0, out_y=-1.0, throttle_us=1234, out_yaw=0.0)
    # AETR: A=roll(from y), E=pitch(from x), T, R
    assert ch[0] == 1500 - 100 * vcfg["rc"]["roll_sign"]  # move left
    assert ch[1] == 1500 + 100 * vcfg["rc"]["pitch_sign"]  # move forward
    assert ch[2] == 1234
    assert ch[3] == 1500
    assert ch[4:] == vcfg["rc"]["aux_channels"]


def test_rc_synth_taer_reorders(vcfg):
    vcfg["rc"]["map"] = "TAER"
    rc = RCSynth(vcfg["rc"], authority=0.5)
    ch = rc.channels(0.5, 0.0, 1100)
    assert ch[0] == 1100 and ch[2] == 1500 + 125 * vcfg["rc"]["pitch_sign"]


def test_rc_synth_sign_config(vcfg):
    vcfg["rc"]["pitch_sign"] = -1
    rc = RCSynth(vcfg["rc"], authority=1.0)
    assert rc.channels(1.0, 0.0, 1500)[1] == 1000


def test_rc_authority_clamps_and_saturates(vcfg):
    rc = RCSynth(vcfg["rc"], authority=0.35)
    assert rc.stick(5.0) == 1500 + 175
    assert rc.stick(-5.0) == 1500 - 175
    assert rc.clamp(3000) == 2000 and rc.clamp(0) == 1000


def test_authority_clips_but_does_not_scale(vcfg):
    lo = RCSynth(vcfg["rc"], authority=0.2, stick_scale=0.2)
    hi = RCSynth(vcfg["rc"], authority=0.5, stick_scale=0.2)
    assert lo.stick(0.5) == hi.stick(0.5) == 1550  # same loop gain
    assert lo.stick(2.0) == 1600 and hi.stick(2.0) == 1600  # norm clips at 1.0 -> scale
    assert hi.stick(1.0) == 1600  # authority is headroom, not gain


def test_horizontal_guidance_yaw_disabled(vcfg):
    g = HorizontalGuidance(vcfg["guidance"])
    cmd = g.step(0.1, -0.1, 0.02, 0.0, 0.0, yaw_err=1.0)
    assert cmd.out_yaw == 0.0
    assert cmd.out_x > 0 and cmd.out_y < 0
