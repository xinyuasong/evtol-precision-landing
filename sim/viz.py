"""Replay a scenario result CSV: trajectory, error, altitude, FSM timeline, PID terms.

python -m sim.viz sim/results/nominal_calm_seed0.csv
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import pandas as pd


def main(path: str) -> None:
    df = pd.read_csv(path)
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(df["y"], df["x"])
    ax.plot(0, 0, "r+", ms=14)
    ax.set_xlabel("y right (m)")
    ax.set_ylabel("x fwd (m)")
    ax.set_title("trajectory (top view)")
    ax.axis("equal")
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(df["t"], df["horiz_err"])
    ax.set_title("horizontal error (m)")
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(df["t"], df["z"], label="true")
    ax.plot(df["t"], df["alt_est"], label="estimate")
    ax.legend()
    ax.set_title("altitude (m)")
    ax = fig.add_subplot(2, 3, 4)
    codes, names = pd.factorize(df["state"])
    ax.step(df["t"], codes, where="post")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("FSM")
    ax = fig.add_subplot(2, 3, 5)
    pid = df["pid"].apply(json.loads)
    for i, name in enumerate(("P", "I", "D")):
        ax.plot(df["t"], pid.apply(lambda p, i=i: p[0][i] if p else 0), label=f"x {name}")
    ax.legend()
    ax.set_title("PID terms, x")
    ax = fig.add_subplot(2, 3, 6)
    for c in ("rc_roll", "rc_pitch", "rc_thr"):
        ax.plot(df["t"], df[c], label=c)
    ax.legend()
    ax.set_title("RC (us)")
    fig.suptitle(path)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])
