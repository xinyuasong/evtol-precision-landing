import numpy as np
import pytest

from control.filter import CVKalman, LowPass


def make():
    return CVKalman(q=0.5, r0=0.01, z_ref_m=2.0, coast_max_s=0.5)


def test_converges_on_constant():
    kf = make()
    rng = np.random.default_rng(0)
    for _ in range(200):
        kf.predict(0.02)
        kf.update(1.0 + rng.normal(0, 0.03), 2.0)
    assert kf.pos == pytest.approx(1.0, abs=0.03)
    assert abs(kf.vel) < 0.15


def test_estimates_velocity_of_ramp():
    kf = make()
    t = 0.0
    for _ in range(200):
        kf.predict(0.02)
        t += 0.02
        kf.update(0.5 * t, 2.0)
    assert kf.vel == pytest.approx(0.5, abs=0.05)


def test_coasts_then_invalidates():
    kf = make()
    t = 0.0
    for _ in range(100):
        kf.predict(0.02)
        t += 0.02
        kf.update(0.5 * t, 2.0)
    p0 = kf.pos
    for _ in range(20):  # 0.4 s no measurement: still valid, position advances
        kf.predict(0.02)
    assert kf.valid
    assert kf.pos > p0 + 0.15
    for _ in range(10):  # now past 0.5 s
        kf.predict(0.02)
    assert not kf.valid


def test_measurement_noise_scales_with_range():
    """Same measurement sequence, far range: filter should trust it less (slower)."""
    near, far = make(), make()
    for _ in range(5):
        near.predict(0.02)
        far.predict(0.02)
        near.update(0.0, 2.0)
        far.update(0.0, 2.0)
    near.predict(0.02)
    far.predict(0.02)
    near.update(1.0, 2.0)
    far.update(1.0, 8.0)
    assert far.pos < near.pos


def test_lowpass_derivative():
    lp = LowPass(0.05)
    t = 0.0
    for _ in range(200):
        t += 0.01
        lp.step(2.0 * t, 0.01)
    assert lp.y == pytest.approx(2.0 * t, abs=0.15)
    assert lp.dy == pytest.approx(2.0, abs=0.05)
