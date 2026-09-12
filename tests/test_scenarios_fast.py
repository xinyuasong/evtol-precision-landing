"""CI subset of the scenario suite: seeded, short, Python detector, direct transport."""

import pytest

from sim.runner import SCENARIO_DIR, evaluate, load_scenario, run_one

FAST = ["nominal_calm", "companion_stall", "touchdown_detect_fails"]  # ~40 s total on CI


@pytest.mark.parametrize("name", FAST)
def test_scenario_passes(name):
    spec = load_scenario(SCENARIO_DIR / f"{name}.yaml")
    res = run_one(spec, "direct")
    ok, why = evaluate(res, spec.pass_criteria)
    assert ok, "; ".join(why)


def test_seed_reproduces_exactly():
    spec = load_scenario(SCENARIO_DIR / "nominal_calm.yaml", seed=7)
    spec.duration_s = 8.0
    a = run_one(spec, "direct")
    b = run_one(spec, "direct")
    assert [r.pos for r in a.records] == [r.pos for r in b.records]
    assert a.transitions == b.transitions
