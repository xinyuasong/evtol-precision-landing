"""control/ must not know a simulator exists. This is the project's central claim;
make it mechanical so it cannot erode at 2 am."""

import re
from pathlib import Path

CONTROL = Path(__file__).resolve().parents[1] / "control"
FORBIDDEN = re.compile(r"\b(sim|sims|simulation|simulate|simulated|simulator|mock|mocked|mocks|SIMULATION|pty|openpty)\b", re.IGNORECASE)


def test_control_has_no_simulation_awareness():
    hits = []
    for f in sorted(CONTROL.rglob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if FORBIDDEN.search(line):
                hits.append(f"{f.relative_to(CONTROL.parent)}:{i}: {line.strip()}")
    assert not hits, "control/ references the simulator:\n" + "\n".join(hits)


def test_control_does_not_import_sim_package():
    for f in CONTROL.rglob("*.py"):
        for line in f.read_text().splitlines():
            assert not re.match(r"\s*(from|import)\s+(sim|tests)\b", line), f"{f}: {line}"
