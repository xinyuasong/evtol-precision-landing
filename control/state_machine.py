"""Mission state machine. Explicit states, explicit transitions, timed everything.

    IDLE -> SEARCH -> ACQUIRE -> TRACK -> DESCEND -> FINAL -> DISARM
                 ^                                     |
                 +--------------- ABORT <--------------+  (tag lost / not converging)

Safety trips are handled *outside* this machine: when Safety trips, the loop
stops transmitting and the FC's own failsafe lands the aircraft. The ABORT state
here is a software recovery (climb, hold, re-search), not a failsafe.
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np


class S(Enum):
    IDLE = auto()
    SEARCH = auto()
    ACQUIRE = auto()
    TRACK = auto()
    DESCEND = auto()
    FINAL = auto()
    DISARM = auto()
    ABORT = auto()
    HANDBACK = auto()  # gave up: stop transmitting, the FC's own failsafe lands it


class Mission:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.state = S.IDLE
        self.t = 0.0
        self.t_enter = 0.0
        self.consec = 0
        self.t_last_tag: float | None = None
        self._stable_since: float | None = None
        self._acq_samples: list[tuple[float, float]] = []
        self.transitions: list[tuple[float, S, S, str]] = []
        self.last_reason = ""

    def _go(self, s: S, reason: str) -> None:
        if s is not self.state:
            self.transitions.append((self.t, self.state, s, reason))
            self.state = s
            self.t_enter = self.t
            self._stable_since = None
            self._acq_samples = []
            self.last_reason = reason

    def elapsed(self) -> float:
        return self.t - self.t_enter

    def update(
        self,
        dt: float,
        tag_xy: tuple[float, float] | None,
        alt_m: float | None,
        horiz_err_m: float | None,
        armed: bool,
        override: bool,
        touchdown: bool,
    ) -> S:
        c = self.cfg
        self.t += dt

        # Detections arrive at the camera rate, slower than this loop runs; a tick
        # without a new pose is not a lost tag. Continuity is judged by time.
        if tag_xy is not None:
            self.consec += 1
            self.t_last_tag = self.t
        since_tag = None if self.t_last_tag is None else self.t - self.t_last_tag
        seen = since_tag is not None and since_tag <= c["detection_gap_s"]
        if not seen:
            self.consec = 0

        # Global: losing arm or override returns to IDLE from any non-terminal state.
        if self.state not in (S.IDLE, S.DISARM, S.HANDBACK) and not (armed and override):
            self._go(S.IDLE, "arm/override released")
            return self.state

        st = self.state
        if st is S.IDLE:
            if armed and override:
                self._go(S.SEARCH, "armed + override engaged")

        elif st is S.SEARCH:
            if self.consec >= c["acquire_consecutive_frames"]:
                self._go(S.ACQUIRE, f"{self.consec} consecutive detections")
            elif self.elapsed() > c["search_timeout_s"]:
                self._go(S.HANDBACK, "search timeout, nothing to land on")

        elif st is S.ACQUIRE:
            if tag_xy is not None:
                self._acq_samples.append(tag_xy)
            if not seen:
                self._go(S.SEARCH, "tag lost during acquire")
            elif self.elapsed() >= c["acquire_settle_s"]:
                sig = float(np.max(np.std(np.asarray(self._acq_samples), axis=0))) if len(self._acq_samples) > 1 else float("inf")
                if sig <= c["acquire_sigma_max_m"]:
                    self._go(S.TRACK, f"pose stable sigma={sig:.3f} m")
                else:
                    self._acq_samples = []
                    self.t_enter = self.t  # stay, re-sample

        elif st is S.TRACK:
            if self.elapsed() > c["track_timeout_s"]:
                self._go(S.SEARCH, "track timeout, not converging")
            elif since_tag is not None and since_tag > c["track_lost_s"]:
                self._go(S.SEARCH, "tag lost in track")
            elif horiz_err_m is not None and seen and horiz_err_m < c["track_ok_err_m"]:
                if self._stable_since is None:
                    self._stable_since = self.t
                elif self.t - self._stable_since >= c["track_stable_s"]:
                    self._go(S.DESCEND, f"centred < {c['track_ok_err_m']} m for {c['track_stable_s']} s")
            else:
                self._stable_since = None

        elif st is S.DESCEND:
            if touchdown:
                self._go(S.DISARM, "touchdown detected during descent")
            elif since_tag is not None and since_tag > c["descend_lost_s"]:
                self._go(S.ABORT, "tag lost in descent")
            elif alt_m is not None and alt_m < c["final_alt_m"] and horiz_err_m is not None and horiz_err_m < c["final_err_m"]:
                self._go(S.FINAL, "committed: low and centred")

        elif st is S.FINAL:
            # Vision is ignored here by design. Touchdown ends it; if the detector never
            # fires, the timeout does — the vehicle is within centimetres of the ground
            # either way and the setpoint floor must not push on forever.
            if touchdown:
                self._go(S.DISARM, "touchdown detected")
            elif self.elapsed() > c["final_timeout_s"]:
                self._go(S.DISARM, "FINAL timeout, touchdown never detected")

        elif st is S.ABORT:
            if self.elapsed() >= c["abort_min_hold_s"]:
                self._go(S.SEARCH, "abort hold complete")

        elif st in (S.DISARM, S.HANDBACK):
            pass

        return self.state
