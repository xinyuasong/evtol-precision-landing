"""CSV logger fed from a queue. The control loop never blocks on disk."""

from __future__ import annotations

import csv
import queue
import threading
from pathlib import Path

FIELDS = [
    "t_mono",
    "t_wall",
    "fsm_state",
    "tag_id",
    "tag_valid",
    "reproj_err",
    "pose_age_s",
    "p_cam_x",
    "p_cam_y",
    "p_cam_z",
    "p_level_x",
    "p_level_y",
    "p_level_z",
    "kf_x",
    "kf_y",
    "kf_vx",
    "kf_vy",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "alt_m",
    "vario_ms",
    "alt_sp_m",
    "rate_cmd_ms",
    "err_x",
    "err_y",
    "horiz_err",
    "pid_p_x",
    "pid_i_x",
    "pid_d_x",
    "out_x",
    "pid_p_y",
    "pid_i_y",
    "pid_d_y",
    "out_y",
    "pid_p_z",
    "pid_i_z",
    "pid_d_z",
    "rc_roll",
    "rc_pitch",
    "rc_throttle",
    "rc_yaw",
    "vbat",
    "loop_dt_s",
    "safety_trip",
    "tx",
]


class CSVLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue = queue.Queue(maxsize=10000)
        self._running = False
        self._thread = threading.Thread(target=self._run, name="logger", daemon=True)
        self.dropped = 0

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)

    def put(self, row: dict) -> None:
        try:
            self._q.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            while self._running or not self._q.empty():
                try:
                    row = self._q.get(timeout=0.2)
                except queue.Empty:
                    continue
                w.writerow({k: row.get(k, "") for k in FIELDS})
