import pytest

from control.safety import Safety


@pytest.fixture
def s(vcfg):
    return Safety(vcfg["safety"])


def nominal(t=10.0):
    return dict(t=t, loop_dt=0.02, t_attitude=t - 0.01, t_altitude=t - 0.01, alt_m=2.0, horiz_err_m=0.2, vbat_v=24.0, roll_deg=3.0, pitch_deg=-2.0, armed=True)


def test_nominal_ok(s):
    assert s.check(**nominal())
    assert s.ok


def test_never_received_telemetry_gets_a_startup_grace(s, vcfg):
    g = vcfg["safety"]["startup_grace_s"]
    kw = nominal(t=0.0)
    kw.update(t_attitude=None, t_altitude=None)
    assert s.check(**kw)  # first tick, nothing has arrived yet
    kw["t"] = g / 2
    assert s.check(**kw)
    kw["t"] = g + 0.1
    assert not s.check(**kw)
    assert "attitude" in s.trip_reason


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("loop_dt", 0.2, "loop overrun"),
        ("t_attitude", 9.0, "attitude"),
        ("t_altitude", 9.0, "altitude telemetry"),
        ("alt_m", 20.0, "ceiling"),
        ("horiz_err_m", 6.0, "implausible"),
        ("vbat_v", 20.0, "battery"),
        ("roll_deg", 50.0, "tilt"),
    ],
)
def test_each_trip(s, field, value, reason):
    kw = nominal()
    kw[field] = value
    assert not s.check(**kw)
    assert reason in s.trip_reason


def test_altitude_staleness_only_matters_when_armed(s):
    kw = nominal()
    kw["t_altitude"] = None
    kw["armed"] = False
    assert s.check(**kw)


def test_trip_latches(s):
    kw = nominal()
    kw["vbat_v"] = 19.0
    assert not s.check(**kw)
    assert not s.check(**nominal(t=11.0))
    assert "battery" in s.trip_reason and s.trip_t == 10.0


def test_divergence_trips_before_fov_edge(s, vcfg):
    """A sign-inverted transform grows the error at full authority; it must trip before ~1.9 m."""
    err = 0.5
    t = 10.0
    for _ in range(400):
        t += 0.02
        err += 0.02 * 0.4  # 0.4 m/s runaway
        kw = nominal(t)
        kw.update(horiz_err_m=err, controlling=True)
        if not s.check(**kw):
            break
    assert not s.ok and "diverging" in s.trip_reason
    assert err < 1.9


def test_shrinking_error_never_trips_divergence(s):
    err, t = 2.0, 10.0
    for _ in range(600):
        t += 0.02
        err = max(0.0, err - 0.02 * 0.3)
        kw = nominal(t)
        kw.update(horiz_err_m=err, controlling=True)
        assert s.check(**kw)


def test_gust_excursion_does_not_trip_divergence(s):
    """Error grows 0.7 m then turns around — a gust, not an inversion."""
    import math

    t = 10.0
    for i in range(400):
        t += 0.02
        err = 0.1 + 0.7 * abs(math.sin(i * 0.02 * 2 * math.pi / 5.0))  # 5 s period wobble
        kw = nominal(t)
        kw.update(horiz_err_m=err, controlling=True)
        assert s.check(**kw), s.trip_reason


def test_missing_optional_values_do_not_trip(s):
    kw = nominal()
    kw.update(alt_m=None, horiz_err_m=None, vbat_v=None, roll_deg=None, pitch_deg=None, t_altitude=None, armed=False)
    assert s.check(**kw)
