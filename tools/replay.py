"""Re-run the control loop against a logged flight (control/main.py --log CSV) with the
current config, so gains can be changed offline and compared tick for tick.

    python tools/replay.py logs/flight.csv --config config/vehicle.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control.altitude import AltitudeSample  # noqa: E402
from control.config import load_tags, load_vehicle  # noqa: E402
from control.main import ControlLoop, Inputs  # noqa: E402
from control.pose_sub import TagPose  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--config", default="config/vehicle.yaml")
    ap.add_argument("--tags", default="config/tags.yaml")
    a = ap.parse_args(argv)
    loop = ControlLoop(load_vehicle(a.config), load_tags(a.tags))
    flags = {"mode_flags": 0b11}
    n = 0
    for r in csv.DictReader(open(a.log)):
        t = float(r["t_mono"])
        poses = []
        if r.get("tag_id") and r.get("p_cam_x"):
            # the log holds the resolved pad origin; replay it as a face-on tag with zero offset
            poses = [
                (
                    TagPose(
                        t - float(r.get("pose_age_s") or 0), 2, float(r["p_cam_x"]), float(r["p_cam_y"]), float(r["p_cam_z"]), 3.14159265, 0.0, 0.0, 0.0, True
                    ),
                    t,
                )
            ]
        att = (float(r["roll_deg"]), float(r["pitch_deg"]), float(r["yaw_deg"])) if r.get("roll_deg") else None
        alt = AltitudeSample(float(r["alt_m"]), float(r["vario_ms"] or 0), t) if r.get("alt_m") else None
        out = loop.step(Inputs(t, poses, att, t, alt, flags, float(r["vbat"]) if r.get("vbat") else None))
        n += 1
        if out.channels and r.get("rc_pitch"):
            m = loop.rc.map
            print(
                f"{t:10.3f} {out.state.name:8} logged pitch={r['rc_pitch']:>5} replay pitch={out.channels[m.index('E')]:>5}  logged roll={r['rc_roll']:>5} replay roll={out.channels[m.index('A')]:>5}"
            )
    print(f"{n} ticks replayed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
