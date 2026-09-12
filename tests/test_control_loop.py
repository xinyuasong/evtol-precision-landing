"""ControlLoop driven with hand-built Inputs. No threads, no sockets, no serial."""

import math

import pytest

from control.altitude import AltitudeSample
from control.main import ControlLoop, Inputs
from control.pose_sub import TagPose
from control.state_machine import S

DT = 0.02
ARMED_OVERRIDE = {"mode_flags": 0b11}


FACE_ON = (math.pi, 0.0, 0.0)  # tag->camera rotation for a level downward camera


def pose(t, x, y, z, tag_id=0, err=0.5, valid=True):
    return TagPose(t, tag_id, x, y, z, *FACE_ON, err, valid)


class Driver:
    """Feeds a loop with constant telemetry and a caller-chosen pose each tick."""

    def __init__(self, vcfg, tcfg, alt=3.0, flags=ARMED_OVERRIDE, vbat=24.0):
        self.loop = ControlLoop(vcfg, tcfg)
        self.t = 100.0
        self.alt = alt
        self.flags = flags
        self.vbat = vbat
        self.att = (0.0, 0.0, 0.0)

    def tick(self, p_cam=None, tag_id=0, n=1, pose_age=0.0, reproj=0.5):
        """p_cam is the PAD ORIGIN in the camera frame; the tag's own offset is added
        so the loop has to resolve it back (face-on: printed-right = +x, printed-up = -y)."""
        out = None
        for _ in range(n):
            self.t += DT
            p = None
            if p_cam:
                t = self.loop.tags.get(tag_id, {"offset_xy_m": [0, 0]})
                ox, oy = t["offset_xy_m"]
                p = pose(self.t - pose_age, p_cam[0] + ox, p_cam[1] - oy, p_cam[2], tag_id=tag_id, err=reproj)
            inp = Inputs(
                now=self.t,
                poses=[(p, self.t)] if p else [],
                attitude=self.att,
                t_attitude=self.t,
                altitude=AltitudeSample(self.alt, 0.0, self.t),
                status=self.flags,
                vbat_v=self.vbat,
            )
            out = self.loop.step(inp)
        return out

    def rc(self, out):
        m = self.loop.rc.map
        return {c: out.channels[m.index(c)] for c in "AETR"}


def test_idle_streams_min_throttle_neutral_sticks(vcfg, tcfg):
    d = Driver(vcfg, tcfg, flags={"mode_flags": 0})
    out = d.tick()
    assert out.state is S.IDLE
    rc = d.rc(out)
    assert rc["T"] == vcfg["rc"]["min_us"] and rc["A"] == rc["E"] == 1500


def test_tag_forward_pitches_forward_tag_right_rolls_right(vcfg, tcfg):
    d = Driver(vcfg, tcfg)
    d.tick(n=1)  # IDLE -> SEARCH
    # camera: tag toward image top (-Y) is toward the nose
    out = d.tick(p_cam=(0.0, -0.5, 3.0), n=80)
    assert out.state in (S.ACQUIRE, S.TRACK)
    rc = d.rc(out)
    ps, rs = vcfg["rc"]["pitch_sign"], vcfg["rc"]["roll_sign"]
    assert (rc["E"] - 1500) * ps > 0 and abs(rc["A"] - 1500) <= 1
    d2 = Driver(vcfg, tcfg)
    d2.tick(n=1)
    out = d2.tick(p_cam=(0.5, 0.0, 3.0), n=80)  # tag right in image
    rc = d2.rc(out)
    assert (rc["A"] - 1500) * rs > 0 and abs(rc["E"] - 1500) <= 1


def test_tilt_compensation_cancels_phantom_error(vcfg, tcfg):
    """Tag on the optical axis while pitched nose-up 10 deg: the tag really is
    forward, so the loop must command forward, not neutral."""
    d = Driver(vcfg, tcfg)
    d.att = (0.0, 10.0, 0.0)
    d.tick(n=1)
    out = d.tick(p_cam=(0.0, 0.0, 3.0), n=80)
    assert out.log["p_level_x"] == pytest.approx(3 * 0.17365, abs=1e-3)
    assert (d.rc(out)["E"] - 1500) * vcfg["rc"]["pitch_sign"] > 0


def test_search_holds_neutral_without_tag(vcfg, tcfg):
    d = Driver(vcfg, tcfg)
    out = d.tick(n=5)
    assert out.state is S.SEARCH
    rc = d.rc(out)
    assert rc["A"] == rc["E"] == 1500
    assert out.log["alt_sp_m"] == pytest.approx(3.0)


def test_stale_pose_is_ignored(vcfg, tcfg):
    d = Driver(vcfg, tcfg)
    d.tick(n=1)
    out = d.tick(p_cam=(0.0, 0.0, 3.0), n=20, pose_age=1.0)
    assert out.state is S.SEARCH and out.horiz_err is None


def test_reprojection_gate_and_altitude_validity(vcfg, tcfg):
    d = Driver(vcfg, tcfg)
    d.tick(n=1)
    assert d.tick(p_cam=(0, 0, 3.0), n=20, reproj=5.0).state is S.SEARCH  # bad fit rejected
    assert d.tick(p_cam=(0, 0, 3.0), n=20, tag_id=2).state is S.SEARCH  # 2.5 cm tag not valid at 3 m
    assert d.tick(p_cam=(0, 0, 3.0), n=20, tag_id=99).state is S.SEARCH  # unknown id
    assert d.tick(p_cam=(0, 0, 3.0), n=20, tag_id=0).state is not S.SEARCH


def test_tag_handoff_is_continuous(vcfg, tcfg):
    d = Driver(vcfg, tcfg, alt=2.0)  # both the 0.40 m and 0.10 m tags are valid here
    d.tick(n=1)
    d.tick(p_cam=(0.1, 0.0, 2.0), n=60, tag_id=0)
    before = d.loop.kf_x.pos, d.loop.kf_y.pos
    assert before[1] > 0.05
    out = d.tick(p_cam=(0.1, 0.0, 2.0), n=5, tag_id=1)  # same pad origin via a different tag with a different offset
    assert d.loop.kf_x.pos == pytest.approx(before[0], abs=0.01)
    assert d.loop.kf_y.pos == pytest.approx(before[1], abs=0.01)
    assert out.state is not S.SEARCH


def test_safety_trip_stops_transmission(vcfg, tcfg):
    d = Driver(vcfg, tcfg)
    d.tick(n=3)
    d.vbat = 19.0
    out = d.tick(n=1)
    assert out.channels is None
    assert "battery" in d.loop.safety.trip_reason
    d.vbat = 24.0
    assert d.tick(n=1).channels is None  # latched


def test_implausible_horizontal_error_trips(vcfg, tcfg):
    """An error beyond max_horiz_err_m stops the stream (measurement sanity gate)."""
    d = Driver(vcfg, tcfg)
    d.tick(n=1)
    out = d.tick(p_cam=(6.0, 0.0, 3.0), n=3)
    assert out.channels is None and "implausible" in d.loop.safety.trip_reason


def test_diverging_error_trips(vcfg, tcfg):
    """The sign-inversion catch: the tag keeps getting further away while we chase it."""
    d = Driver(vcfg, tcfg)
    d.tick(n=1)
    out = None
    for i in range(400):
        out = d.tick(p_cam=(0.3 + 0.008 * i, 0.0, 3.0), n=1)  # 0.4 m/s runaway
        if out.channels is None:
            break
    assert out.channels is None and "diverging" in d.loop.safety.trip_reason


def test_missing_altitude_yields_no_command_then_trips(vcfg, tcfg):
    d = Driver(vcfg, tcfg)
    d.tick(n=2)
    d.t += DT
    inp = Inputs(d.t, [], d.att, d.t, None, d.flags, 24.0)
    out = d.loop.step(inp)
    assert out.channels is None
    for _ in range(int((vcfg["safety"]["startup_grace_s"] + vcfg["safety"]["altitude_timeout_s"]) / DT) + 5):
        d.t += DT
        out = d.loop.step(Inputs(d.t, [], d.att, d.t, None, d.flags, 24.0))
    assert "altitude" in d.loop.safety.trip_reason


def test_prefers_smallest_usable_tag(vcfg, tcfg):
    d = Driver(vcfg, tcfg, alt=1.5)
    d.tick(n=1)
    d.t += DT
    big = pose(d.t, 0.3 - 0.35, 0.0, 1.5, tag_id=0)  # pad origin at +0.3 via the big tag
    small = pose(d.t, 0.3 + 0.10, 0.0, 1.5, tag_id=1)  # same origin via the 0.10 m tag
    out = d.loop.step(Inputs(d.t, [(big, d.t), (small, d.t)], d.att, d.t, AltitudeSample(1.5, 0.0, d.t), d.flags, 24.0))
    assert out.log["tag_id"] == 1
    assert out.log["p_level_y"] == pytest.approx(0.3, abs=1e-9)


def test_handback_stops_transmission(vcfg, tcfg):
    d = Driver(vcfg, tcfg)
    n = int(vcfg["state_machine"]["search_timeout_s"] / DT) + 5
    out = d.tick(n=n)
    assert out.state is S.HANDBACK and out.channels is None


def test_log_row_has_core_fields(vcfg, tcfg):
    d = Driver(vcfg, tcfg)
    d.tick(n=1)
    out = d.tick(p_cam=(0.2, -0.1, 3.0), n=10)
    for k in ("fsm_state", "rc_throttle", "kf_x", "pid_p_x", "alt_sp_m", "tx", "loop_dt_s"):
        assert k in out.log
