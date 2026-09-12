"""Envelope limits and the watchdog. When this trips, the loop stops transmitting
and the FC's own failsafe lands the aircraft. Trips latch."""

from __future__ import annotations


class Safety:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.trip_reason: str | None = None
        self.trip_t: float | None = None
        self.t_start: float | None = None
        self._err_hist: list[tuple[float, float]] = []

    @property
    def ok(self) -> bool:
        return self.trip_reason is None

    def _trip(self, reason: str, t: float) -> bool:
        if self.trip_reason is None:
            self.trip_reason = reason
            self.trip_t = t
        return False

    def check(
        self,
        t: float,
        loop_dt: float,
        t_attitude: float | None,
        t_altitude: float | None,
        alt_m: float | None,
        horiz_err_m: float | None,
        vbat_v: float | None,
        roll_deg: float | None,
        pitch_deg: float | None,
        armed: bool,
        controlling: bool = False,
    ) -> bool:
        if not self.ok:
            return False
        c = self.cfg
        if self.t_start is None:
            self.t_start = t
        # A sample that has never arrived is "stale" only once the startup grace has passed;
        # the telemetry thread needs a poll or two before the first tick can see anything.
        past_grace = t - self.t_start > c["startup_grace_s"]
        if loop_dt > c["loop_overrun_s"]:
            return self._trip(f"loop overrun {loop_dt * 1000:.0f} ms", t)
        if (t_attitude is None and past_grace) or (t_attitude is not None and t - t_attitude > c["attitude_timeout_s"]):
            return self._trip("attitude telemetry stale", t)
        if armed and ((t_altitude is None and past_grace) or (t_altitude is not None and t - t_altitude > c["altitude_timeout_s"])):
            return self._trip("altitude telemetry stale", t)
        if alt_m is not None and alt_m > c["max_alt_m"]:
            return self._trip(f"altitude {alt_m:.1f} m over ceiling", t)
        if horiz_err_m is not None and horiz_err_m > c["max_horiz_err_m"]:
            return self._trip(f"horizontal error {horiz_err_m:.1f} m implausible", t)
        # Divergence: while the controller is acting on vision, the error should not keep
        # growing. A transform sign error makes it grow at full authority; this trips well
        # before the tag leaves the field of view.
        if controlling and horiz_err_m is not None:
            self._err_hist.append((t, horiz_err_m))
            self._err_hist = [(tt, e) for tt, e in self._err_hist if t - tt <= c["divergence_window_s"]]
            if len(self._err_hist) > 5 and t - self._err_hist[0][0] >= 0.9 * c["divergence_window_s"]:
                lo = min(e for _, e in self._err_hist)
                errs = [e for _, e in self._err_hist]
                # A sign error grows the error monotonically at full authority. A gust
                # excursion also grows it, but not monotonically: it turns around. Require
                # near-monotonic growth so the trip is specific to the inversion signature.
                ups = sum(1 for a, b in zip(errs, errs[1:], strict=False) if b >= a)
                monotonic = ups >= c["divergence_monotonic_frac"] * (len(errs) - 1)
                if horiz_err_m - lo > c["divergence_m"] and monotonic:
                    return self._trip(f"horizontal error diverging (+{horiz_err_m - lo:.2f} m in {c['divergence_window_s']:.0f} s) — transform sign?", t)
        else:
            self._err_hist = []
        if vbat_v is not None and vbat_v < c["min_vbat_v"]:
            return self._trip(f"battery {vbat_v:.1f} V", t)
        if roll_deg is not None and pitch_deg is not None and max(abs(roll_deg), abs(pitch_deg)) > c["max_tilt_deg"]:
            return self._trip("tilt over limit", t)
        return True
