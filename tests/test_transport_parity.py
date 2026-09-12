"""The same world driven through both transports must land the same way. The pty
run is real-time and byte-identical to hardware; the direct run is what Monte
Carlo uses. If these diverge, the Monte Carlo table does not transfer."""

import pytest

from sim.runner import SCENARIO_DIR, load_scenario, run_one


@pytest.mark.slow
def test_pty_and_direct_agree_on_nominal():
    spec = load_scenario(SCENARIO_DIR / "nominal_calm.yaml")
    spec.duration_s = 40.0
    direct = run_one(spec, "direct")
    pty = run_one(spec, "pty")
    assert direct.touchdown_err_m is not None and pty.touchdown_err_m is not None
    assert direct.terminal_state == "DISARM" and pty.terminal_state == "DISARM"
    assert abs(direct.touchdown_err_m - pty.touchdown_err_m) < 0.05
    assert abs(direct.touchdown_t - pty.touchdown_t) < 3.0
    assert pty.tx_ticks > 0.8 * 50 * pty.touchdown_t  # the RC stream really flowed at ~50 Hz
