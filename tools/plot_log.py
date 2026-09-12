"""Plot a control-node CSV log or a sim/results scenario CSV.  python tools/plot_log.py file.csv"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import pandas as pd


def main(path: str) -> None:
    df = pd.read_csv(path)
    tcol = "t" if "t" in df else "t_mono"
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(11, 10))
    if "x" in df:
        ax[0].plot(df[tcol], df["x"], label="x")
        ax[0].plot(df[tcol], df["y"], label="y")
    if "horiz_err" in df:
        ax[0].plot(df[tcol], df["horiz_err"], label="horiz err")
    ax[0].legend()
    ax[0].set_ylabel("m")
    for c in ("z", "alt_m", "alt_est", "alt_sp_m"):
        if c in df:
            ax[1].plot(df[tcol], df[c], label=c)
    ax[1].legend()
    ax[1].set_ylabel("altitude m")
    for c in ("rc_roll", "rc_pitch", "rc_thr", "rc_throttle"):
        if c in df:
            ax[2].plot(df[tcol], df[c], label=c)
    ax[2].legend()
    ax[2].set_ylabel("us")
    st = df["state"] if "state" in df else df["fsm_state"]
    codes, names = pd.factorize(st)
    ax[3].step(df[tcol], codes, where="post")
    ax[3].set_yticks(range(len(names)))
    ax[3].set_yticklabels(names)
    ax[3].set_xlabel("s")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])
