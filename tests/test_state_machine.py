import pytest

from control.state_machine import Mission, S

DT = 0.02


@pytest.fixture
def m(vcfg):
    return Mission(vcfg["state_machine"])


def run(m, n, **kw):
    kw.setdefault("tag_xy", None)
    kw.setdefault("alt_m", 3.0)
    kw.setdefault("horiz_err_m", None)
    kw.setdefault("armed", True)
    kw.setdefault("override", True)
    kw.setdefault("touchdown", False)
    for _ in range(n):
        s = m.update(DT, **kw)
    return s


def to_search(m):
    assert run(m, 1) is S.SEARCH
    return m


def to_track(m, vcfg):
    to_search(m)
    assert run(m, vcfg["state_machine"]["acquire_consecutive_frames"], tag_xy=(0.5, 0.1)) is S.ACQUIRE
    assert run(m, int(vcfg["state_machine"]["acquire_settle_s"] / DT) + 2, tag_xy=(0.5, 0.1), horiz_err_m=0.5) is S.TRACK
    return m


def to_descend(m, vcfg):
    to_track(m, vcfg)
    assert run(m, int(vcfg["state_machine"]["track_stable_s"] / DT) + 2, tag_xy=(0.1, 0.0), horiz_err_m=0.1) is S.DESCEND
    return m


def test_idle_waits_for_arm_and_override(m):
    assert run(m, 5, armed=False, override=False) is S.IDLE
    assert run(m, 5, armed=True, override=False) is S.IDLE
    assert run(m, 1, armed=True, override=True) is S.SEARCH


def test_release_returns_to_idle_from_anywhere(m, vcfg):
    to_descend(m, vcfg)
    assert run(m, 1, tag_xy=(0, 0), horiz_err_m=0.0, override=False) is S.IDLE


def test_search_needs_consecutive_detections(m, vcfg):
    to_search(m)
    c = vcfg["state_machine"]
    n = c["acquire_consecutive_frames"]
    gap = int(c["detection_gap_s"] / DT) + 2
    for _ in range(3):
        run(m, n - 1, tag_xy=(0, 0))
        assert run(m, gap) is S.SEARCH  # a real gap resets the count
    # short gaps (slower camera than control loop) do not reset it
    for _ in range(n):
        run(m, 1, tag_xy=(0, 0))
        run(m, 1)
    assert m.state is S.ACQUIRE


def test_short_detection_gaps_do_not_count_as_loss(m, vcfg):
    to_track(m, vcfg)
    for _ in range(100):  # tag every 3rd tick, as a 30 fps camera under a 50 Hz loop would give
        run(m, 1, tag_xy=(0.1, 0), horiz_err_m=0.1)
        run(m, 2, horiz_err_m=0.1)
    assert m.state in (S.TRACK, S.DESCEND)
    assert not any(b is S.SEARCH for _, a, b, _ in m.transitions if a is S.TRACK)


def test_acquire_falls_back_on_loss(m, vcfg):
    to_search(m)
    run(m, vcfg["state_machine"]["acquire_consecutive_frames"], tag_xy=(0, 0))
    assert run(m, 1) is S.ACQUIRE
    assert run(m, int(vcfg["state_machine"]["detection_gap_s"] / DT) + 2) is S.SEARCH


def test_acquire_rejects_unstable_pose(m, vcfg):
    to_search(m)
    run(m, vcfg["state_machine"]["acquire_consecutive_frames"], tag_xy=(0, 0))
    n = int(vcfg["state_machine"]["acquire_settle_s"] / DT) + 2
    for i in range(n):
        m.update(DT, (0.0 if i % 2 else 1.0, 0.0), 3.0, 0.5, True, True, False)
    assert m.state is S.ACQUIRE  # sigma too high: keeps sampling


def test_track_to_descend_requires_sustained_centring(m, vcfg):
    to_track(m, vcfg)
    half = int(vcfg["state_machine"]["track_stable_s"] / DT) // 2
    run(m, half, tag_xy=(0.1, 0), horiz_err_m=0.1)
    run(m, 1, tag_xy=(0.5, 0), horiz_err_m=0.5)  # excursion resets the timer
    assert run(m, half + 2, tag_xy=(0.1, 0), horiz_err_m=0.1) is S.TRACK
    assert run(m, half + 2, tag_xy=(0.1, 0), horiz_err_m=0.1) is S.DESCEND


def test_track_timeout_returns_to_search(m, vcfg):
    to_track(m, vcfg)
    n = int(vcfg["state_machine"]["track_timeout_s"] / DT) + 2
    run(m, n, tag_xy=(0.5, 0), horiz_err_m=0.5)  # tag visible but never centred
    assert any(a is S.TRACK and b is S.SEARCH and "timeout" in r for _, a, b, r in m.transitions)


def test_track_lost_returns_to_search(m, vcfg):
    to_track(m, vcfg)
    n = int(vcfg["state_machine"]["track_lost_s"] / DT) + 2
    assert run(m, n) is S.SEARCH


def test_descend_lost_aborts_then_searches(m, vcfg):
    to_descend(m, vcfg)
    n = int(vcfg["state_machine"]["descend_lost_s"] / DT) + 2
    assert run(m, n, alt_m=1.5) is S.ABORT
    assert run(m, int(vcfg["state_machine"]["abort_min_hold_s"] / DT) + 2, alt_m=2.0) is S.SEARCH


def test_descend_to_final_needs_low_and_centred(m, vcfg):
    to_descend(m, vcfg)
    c = vcfg["state_machine"]
    assert run(m, 1, tag_xy=(0.05, 0), horiz_err_m=0.05, alt_m=c["final_alt_m"] + 0.1) is S.DESCEND
    low = c["final_alt_m"] - 0.05
    assert run(m, 1, tag_xy=(0.1, 0), horiz_err_m=c["final_err_m"] + 0.02, alt_m=low) is S.DESCEND
    assert run(m, 1, tag_xy=(0.05, 0), horiz_err_m=0.05, alt_m=low) is S.FINAL


def test_final_ignores_vision_and_ends_on_touchdown(m, vcfg):
    to_descend(m, vcfg)
    run(m, 1, tag_xy=(0.05, 0), horiz_err_m=0.05, alt_m=vcfg["state_machine"]["final_alt_m"] - 0.05)
    n = int(vcfg["state_machine"]["final_timeout_s"] / DT) - 5
    assert run(m, n, alt_m=0.1) is S.FINAL  # tag gone, still committed (until the timeout)
    assert run(m, 1, alt_m=0.05, touchdown=True) is S.DISARM
    assert run(m, 10, armed=False, override=False) is S.DISARM  # terminal


def test_search_timeout_hands_back(m, vcfg):
    to_search(m)
    n = int(vcfg["state_machine"]["search_timeout_s"] / DT) + 2
    assert run(m, n) is S.HANDBACK
    assert run(m, 10, armed=False, override=False) is S.HANDBACK  # terminal


def test_touchdown_during_descend_disarms(m, vcfg):
    to_descend(m, vcfg)
    assert run(m, 1, tag_xy=(0.05, 0), horiz_err_m=0.05, alt_m=0.3, touchdown=True) is S.DISARM


def test_final_timeout_disarms_without_touchdown(m, vcfg):
    to_descend(m, vcfg)
    run(m, 1, tag_xy=(0.05, 0), horiz_err_m=0.05, alt_m=vcfg["state_machine"]["final_alt_m"] - 0.05)
    n = int(vcfg["state_machine"]["final_timeout_s"] / DT) + 2
    assert run(m, n, alt_m=0.06) is S.DISARM
    assert "timeout" in m.transitions[-1][3]


def test_transition_log_records_reasons(m, vcfg):
    to_descend(m, vcfg)
    assert [a.name + ">" + b.name for _, a, b, _ in m.transitions] == ["IDLE>SEARCH", "SEARCH>ACQUIRE", "ACQUIRE>TRACK", "TRACK>DESCEND"]
    assert all(r for *_, r in m.transitions)
