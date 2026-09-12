"""Cheap linear surrogate for tuning the horizontal loop: 1-axis point mass,
first-order lean-angle lag, camera latency, 30 Hz measurements through the real
CVKalman, the real PID at 50 Hz. No rendering. Prints settling metrics per gain set.

    python tools/tune_horizontal.py
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control.config import load_vehicle  # noqa: E402
from control.filter import CVKalman  # noqa: E402
from control.guidance import PID  # noqa: E402

G = 9.81


def simulate(kp, ki, kd, cfg, x0=0.5, alt=3.0, t_end=30.0, lean_max=25.0, att_tau=0.12, latency=0.05, fps=30.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    fc = cfg["filter"]
    kf = CVKalman(fc["q"], fc["r0"], fc["z_ref_m"], fc["coast_max_s"])
    pid = PID(kp, ki, kd, 0.5, 1.0, cfg["guidance"]["x"]["d_tau_s"])
    authority = cfg["guidance"]["authority"]
    dt = 1 / 400
    x, v, lean = x0, 0.0, 0.0
    lean_cmd = 0.0
    t = 0.0
    next_c, next_f = 0.0, 0.0
    pending = []  # (deliver_t, measurement)
    xs = []
    while t < t_end:
        if t >= next_f:
            next_f += 1 / fps
            pending.append((t + latency, -x + rng.normal(0, noise * alt / 2)))
        if t >= next_c:
            next_c += 1 / 50
            kf.predict(1 / 50)
            while pending and pending[0][0] <= t:
                kf.update(pending.pop(0)[1], alt)
            if kf.valid:
                out = pid.step(kf.pos, 1 / 50, kf.vel)
                lean_cmd = out * authority * lean_max
        lean += dt / (att_tau + dt) * (lean_cmd - lean)
        a = G * math.tan(math.radians(lean)) - 0.15 * v * abs(v)
        v += a * dt
        x += v * dt
        t += dt
        xs.append(abs(x))
    xs = np.array(xs)
    n = len(xs)
    settled = next((i for i in range(n) if (xs[i:] < 0.05).all()), None)
    return {"settle_s": None if settled is None else settled * dt, "overshoot": float(xs[int(2 / dt) :].max()), "final": float(xs[-int(3 / dt) :].max())}


if __name__ == "__main__":
    cfg = load_vehicle(Path(__file__).resolve().parents[1] / "config" / "vehicle.yaml")
    rows = []
    for kp, kd in itertools.product([0.4, 0.6, 0.8, 1.0, 1.3], [0.4, 0.7, 1.0, 1.4, 1.8, 2.4]):
        r = simulate(kp, 0.05, kd, cfg)
        rows.append((kp, kd, r))
        print(
            f"kp={kp:4.1f} kd={kd:4.1f}  settle={r['settle_s'] if r['settle_s'] is None else round(r['settle_s'], 1)!s:>5}  peak_after_2s={r['overshoot']:.3f}  final={r['final']:.3f}"
        )
