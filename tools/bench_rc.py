"""Props OFF. Send fixed stick values over MSP and watch the Receiver tab move.

    python tools/bench_rc.py --port /dev/serial0 --roll 1600 --pitch 1500 --throttle 1000
Confirms channel map and sign before any flight. Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control.config import load_vehicle  # noqa: E402
from control.msp import MSPLink  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/vehicle.yaml")
    ap.add_argument("--port", default=None)
    for c in ("roll", "pitch", "throttle", "yaw"):
        ap.add_argument(f"--{c}", type=int, default=1500 if c != "throttle" else 1000)
    a = ap.parse_args(argv)
    cfg = load_vehicle(a.config)
    s, rc = cfg["serial"], cfg["rc"]
    link = MSPLink(a.port or s["port"], s["baud"], s["timeout_s"])
    vals = {"A": a.roll, "E": a.pitch, "T": a.throttle, "R": a.yaw}
    ch = [vals[c] for c in rc["map"]] + list(rc["aux_channels"])
    print("PROPS OFF. Sending", dict(zip(rc["map"], ch, strict=False)), "at 50 Hz. Ctrl-C to stop.")
    try:
        while True:
            link.set_rc(ch, rc["min_us"], rc["max_us"], rc["mid_us"])
            att = link.attitude()
            if att:
                print(f"\rattitude roll={att[0]:+6.1f} pitch={att[1]:+6.1f}   ", end="")
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
