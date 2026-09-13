"""Record a scenario as images: what the camera sees, what the detector found, and how
the flight went. Produces a GIF, a contact sheet, and a dashboard PNG.

    python tools/record_run.py sim/scenarios/nominal_calm.yaml --out media/

The overlay is drawn from the DETECTOR's output, not from ground truth: the green box is
the tag reprojected using the pose solvePnP recovered, and the crosshair is the pad origin
after the per-tag offset is applied. If the pose were wrong, the box would not sit on the tag.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control.frames import rodrigues  # noqa: E402
from sim.harness import DirectRunner  # noqa: E402
from sim.runner import load_scenario  # noqa: E402
from vision.py.detector_py import object_points  # noqa: E402

GREEN = (80, 220, 120)
AMBER = (60, 190, 250)
WHITE = (245, 245, 245)


def annotate(img, K, dist, poses, tags, t, rec):
    """Draw the detector's own solution on the frame, plus a status bar."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for p in poses:
        tag = tags[p.tag_id]
        rvec = np.array([p.rx, p.ry, p.rz])
        tvec = np.array([p.x, p.y, p.z])
        # the tag square, reprojected from the recovered pose
        quad, _ = cv2.projectPoints(object_points(tag["size_m"]), rvec, tvec, K, dist)
        cv2.polylines(vis, [quad.reshape(-1, 2).astype(np.int32)], True, GREEN, 2)
        cv2.putText(
            vis,
            f"id {p.tag_id}  err {p.reproj_err:.2f}px",
            tuple(quad.reshape(-1, 2)[0].astype(int) + np.array([0, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            GREEN,
            1,
            cv2.LINE_AA,
        )
        # the pad origin, after the per-tag offset is removed
        off = np.array([tag["offset_xy_m"][0], tag["offset_xy_m"][1], 0.0])
        origin_cam = tvec - rodrigues(rvec) @ off
        if origin_cam[2] > 0.05:
            o, _ = cv2.projectPoints(origin_cam.reshape(1, 3), np.zeros(3), np.zeros(3), K, dist)
            cx, cy = o.reshape(2).astype(int)
            cv2.drawMarker(vis, (cx, cy), AMBER, cv2.MARKER_CROSS, 22, 2)

    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, h - 34), (w, h), (30, 30, 30), -1)
    state = rec.state_name if rec else "?"
    alt = f"{rec.pos[2]:.2f} m" if rec else "-"
    err = f"{rec.horiz_err * 100:.0f} cm" if (rec and rec.horiz_err is not None) else "-"
    cv2.putText(
        vis,
        f"t {t:5.1f}s   {state:<8}  alt {alt:>7}   horiz err {err:>7}   tags {len(poses)}",
        (8, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    return vis


def dashboard(res, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = res.records
    t = [x.t for x in r]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"{res.name} — touchdown {res.touchdown_err_m * 100:.1f} cm, terminal {res.terminal_state}"
        if res.touchdown_err_m is not None
        else f"{res.name} — no touchdown, terminal {res.terminal_state}"
    )

    a = ax[0][0]
    a.plot([x.pos[1] for x in r], [x.pos[0] for x in r], lw=1.2)
    a.plot(0, 0, "r+", ms=16, mew=2)
    a.plot(r[0].pos[1], r[0].pos[0], "go", ms=6)
    a.set_xlabel("y, right (m)")
    a.set_ylabel("x, forward (m)")
    a.set_title("trajectory over the pad")
    a.axis("equal")
    a.grid(alpha=0.3)

    a = ax[0][1]
    a.plot(t, [x.horiz_err for x in r], lw=1.2)
    a.axhline(0.20, ls="--", c="gray", lw=0.8)
    a.axhline(0.08, ls="--", c="gray", lw=0.8)
    a.set_title("horizontal error (m) — dashed: descend and commit gates")
    a.grid(alpha=0.3)

    a = ax[0][2]
    a.plot(t, [x.pos[2] for x in r], label="true")
    a.plot(t, [x.alt_est for x in r], lw=0.8, label="estimate")
    a.set_title("altitude (m)")
    a.legend()
    a.grid(alpha=0.3)

    a = ax[1][0]
    names = []
    codes = []
    for x in r:
        if x.state_name not in names:
            names.append(x.state_name)
        codes.append(names.index(x.state_name))
    a.step(t, codes, where="post", lw=1.4)
    a.set_yticks(range(len(names)))
    a.set_yticklabels(names)
    a.set_title("mission state")
    a.grid(alpha=0.3)

    a = ax[1][1]
    for i, lbl in enumerate(("P", "I", "D")):
        a.plot(t, [x.pid[0][i] if x.pid else 0 for x in r], lw=1.0, label=f"x {lbl}")
    a.set_title("PID terms, forward axis")
    a.legend()
    a.grid(alpha=0.3)

    a = ax[1][2]
    # roll and pitch track each other closely, so give them different line styles or one
    # hides the other; throttle has a much wider range and gets its own axis.
    a.plot(t, [x.rc[0] if x.rc else None for x in r], lw=1.2, label="roll")
    a.plot(t, [x.rc[1] if x.rc else None for x in r], lw=1.2, ls="--", label="pitch")
    a.axhline(1500, c="gray", lw=0.6)
    a.set_ylabel("stick (us)")
    a.legend(loc="upper left")
    a2 = a.twinx()
    a2.plot(t, [x.rc[2] if x.rc else None for x in r], lw=1.0, c="tab:green", label="throttle")
    a2.set_ylabel("throttle (us)", color="tab:green")
    a2.legend(loc="upper right")
    a.set_title("RC channels sent over MSP")
    a.grid(alpha=0.3)

    for row in ax:
        for a in row:
            a.set_xlabel("t (s)") if a is not ax[0][0] else None
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--out", default="media")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gif-fps", type=int, default=12)
    ap.add_argument("--gif-stride", type=int, default=6, help="keep every Nth camera frame")
    ap.add_argument("--gif-scale", type=float, default=0.5, help="GIF downscale; 1.0 keeps 640x480")
    a = ap.parse_args(argv)

    spec = load_scenario(a.scenario, a.seed)
    runner = DirectRunner(spec)
    world = runner.world
    tags = {int(t["id"]): t for t in spec.tags_cfg["tags"]}
    frames: list[tuple[float, np.ndarray, list]] = []

    original = world.capture

    def capture_and_keep():
        t_before = world.t
        poses = original()
        img = world.cam.render(world.dyn.state, t_before, world.env, world.env.pad_position(t_before))
        if img is not None:
            frames.append((t_before, img, poses))
        return poses

    world.capture = capture_and_keep
    res = runner.run()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    K, dist = world.cam.K, world.cam.dist

    def rec_at(t):
        best = None
        for x in res.records:
            if x.t <= t:
                best = x
            else:
                break
        return best

    vis = [annotate(img, K, dist, poses, tags, t, rec_at(t)) for i, (t, img, poses) in enumerate(frames) if i % a.gif_stride == 0]

    import imageio.v2 as imageio

    small = [cv2.resize(v, None, fx=a.gif_scale, fy=a.gif_scale, interpolation=cv2.INTER_AREA) for v in vis]
    imageio.mimsave(out / f"{spec.name}_camera.gif", [cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for v in small], fps=a.gif_fps, loop=0, subrectangles=True)

    # contact sheet: six frames spread across the descent
    picks = [cv2.resize(vis[int(i * (len(vis) - 1) / 5)], (426, 320), interpolation=cv2.INTER_AREA) for i in range(6)]
    rows = [np.hstack(picks[:3]), np.hstack(picks[3:])]
    cv2.imwrite(str(out / f"{spec.name}_frames.png"), np.vstack(rows))

    dashboard(res, out / f"{spec.name}_dashboard.png")

    td = f"{res.touchdown_err_m * 100:.1f} cm" if res.touchdown_err_m is not None else "none"
    print(f"{spec.name}: touchdown {td}, {len(frames)} frames captured -> {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
