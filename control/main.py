"""Control node entry point. Fixed-rate loop; opens a serial port and speaks MSP.

python -m control.main --port /dev/serial0 --config config/vehicle.yaml --tags config/tags.yaml
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass

import numpy as np

from control.altitude import AltitudeController, AltitudeSample, TelemetryAltitudeSource, VerticalMode
from control.config import load_tags, load_vehicle
from control.filter import CVKalman
from control.frames import FrameTransformer, apply_tag_offset
from control.guidance import HorizontalGuidance, RCSynth
from control.logger import CSVLogger
from control.msp import MSPLink
from control.pose_sub import PoseSubscriber, TagPose
from control.safety import Safety
from control.state_machine import Mission, S
from control.telemetry import Telemetry

_VERTICAL_MODE = {
    S.IDLE: VerticalMode.MIN_THROTTLE,
    S.DISARM: VerticalMode.MIN_THROTTLE,
    S.SEARCH: VerticalMode.HOLD,
    S.ACQUIRE: VerticalMode.HOLD,
    S.TRACK: VerticalMode.HOLD,
    S.DESCEND: VerticalMode.DESCEND,
    S.FINAL: VerticalMode.FINAL,
    S.ABORT: VerticalMode.ABORT_CLIMB,
    S.HANDBACK: VerticalMode.MIN_THROTTLE,  # never transmitted; see below
}
_VISION_STATES = (S.ACQUIRE, S.TRACK, S.DESCEND)


@dataclass
class Inputs:
    """Everything the loop reads in one tick. Built by ControlLoop.gather() from the
    live subsystems, or by a test directly."""

    now: float
    poses: list[tuple[TagPose, float]]  # newest per tag id, with local receive time
    attitude: tuple[float, float, float] | None
    t_attitude: float | None
    altitude: AltitudeSample | None
    status: dict | None
    vbat_v: float | None


@dataclass
class Outputs:
    channels: list[int] | None  # None: do not transmit
    state: S
    horiz_err: float | None
    log: dict


class ControlLoop:
    """Pure computation. No threads, no sockets; those live in the subsystems it is handed."""

    def __init__(self, vcfg: dict, tcfg: dict) -> None:
        self.cfg = vcfg
        self.period = 1.0 / float(vcfg["loop"]["rate_hz"])
        self.pose_max_age = float(vcfg["loop"]["pose_max_age_s"])
        self.max_reproj = float(vcfg["pose"]["max_reproj_err_px"])
        self.tags = {int(t["id"]): t for t in tcfg["tags"]}
        self.frames = FrameTransformer(vcfg["camera"]["R_bc"])
        fc = vcfg["filter"]
        self.kf_x = CVKalman(fc["q"], fc["r0"], fc["z_ref_m"], fc["coast_max_s"])
        self.kf_y = CVKalman(fc["q"], fc["r0"], fc["z_ref_m"], fc["coast_max_s"])
        self.guidance = HorizontalGuidance(vcfg["guidance"])
        self.rc = RCSynth(vcfg["rc"], vcfg["guidance"]["authority"], vcfg["guidance"]["stick_scale"])
        self.alt = AltitudeController(vcfg["altitude"], vcfg["rc"])
        self.fsm = Mission(vcfg["state_machine"])
        self.safety = Safety(vcfg["safety"])
        self.armed_bit = int(vcfg["telemetry"]["armed_flag_bit"])
        self.override_bit = int(vcfg["telemetry"]["override_flag_bit"])
        self._seen: set[tuple[int, float]] = set()
        self._last_cmd = (0.0, 0.0, 0.0)
        self._last_tick: float | None = None
        self._prev_state = S.IDLE

    # ---- helpers ----

    def _tag_usable(self, pose: TagPose, alt_m: float | None) -> bool:
        if not pose.valid or pose.reproj_err > self.max_reproj:
            return False
        t = self.tags.get(pose.tag_id)
        if t is None:
            return False
        if alt_m is not None:
            lo, hi = t["valid_alt_m"]
            if not (lo <= alt_m <= hi):
                return False
        return True

    def _pose_age(self, now: float, pose: TagPose, t_recv: float) -> float:
        age = now - pose.t_capture
        if age < 0 or age > 10.0:
            age = now - t_recv  # publisher clock not comparable; use receive time
        return max(age, 0.0)

    def _pick_pose(self, inp: Inputs, alt_m: float | None) -> tuple[TagPose | None, float | None]:
        """Freshest usable pose per tag; among those, the smallest tag (highest precision)."""
        best, best_age = None, None
        for pose, t_recv in inp.poses:
            age = self._pose_age(inp.now, pose, t_recv)
            if age > self.pose_max_age or not self._tag_usable(pose, alt_m):
                continue
            key = (pose.tag_id, pose.t_capture)
            if key in self._seen:
                continue
            if best is None or self.tags[pose.tag_id]["size_m"] < self.tags[best.tag_id]["size_m"]:
                best, best_age = pose, age
        return best, best_age

    # ---- the tick ----

    def step(self, inp: Inputs) -> Outputs:
        now = inp.now
        dt = self.period if self._last_tick is None else now - self._last_tick
        self._last_tick = now
        log: dict = {"t_mono": now, "t_wall": time.time(), "loop_dt_s": dt}

        # attitude / status
        roll = pitch = yaw = None
        if inp.attitude is not None:
            roll, pitch, yaw = inp.attitude
        flags = inp.status["mode_flags"] if inp.status else 0
        armed = bool(flags >> self.armed_bit & 1)
        override = bool(flags >> self.override_bit & 1)
        alt_m = inp.altitude.alt_m if inp.altitude else None
        vario = inp.altitude.vario_ms if inp.altitude else None
        log.update(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw, alt_m=alt_m, vario_ms=vario, vbat=inp.vbat_v)

        # vision: new, fresh, usable pose -> level frame -> KF
        self.kf_x.predict(dt)
        self.kf_y.predict(dt)
        tag_xy = None
        pose, age = self._pick_pose(inp, alt_m)
        if pose is not None:
            log.update(tag_id=pose.tag_id, tag_valid=int(pose.valid), reproj_err=pose.reproj_err, pose_age_s=age)
        if pose is not None and roll is not None:
            self._seen.add((pose.tag_id, pose.t_capture))
            if len(self._seen) > 64:
                self._seen = set(sorted(self._seen, key=lambda k: k[1])[-32:])
            t = self.tags[pose.tag_id]
            p_cam = apply_tag_offset((pose.x, pose.y, pose.z), (pose.rx, pose.ry, pose.rz), t["offset_xy_m"])
            p_level = self.frames.camera_to_level(p_cam, roll, pitch)
            rng = float(np.linalg.norm(p_level))
            self.kf_x.update(float(p_level[0]), rng)
            self.kf_y.update(float(p_level[1]), rng)
            tag_xy = (float(p_level[0]), float(p_level[1]))
            log.update(p_cam_x=p_cam[0], p_cam_y=p_cam[1], p_cam_z=p_cam[2], p_level_x=p_level[0], p_level_y=p_level[1], p_level_z=p_level[2])

        kf_ok = self.kf_x.valid and self.kf_y.valid
        err_x = self.kf_x.pos if kf_ok else None
        err_y = self.kf_y.pos if kf_ok else None
        horiz_err = math.hypot(err_x, err_y) if kf_ok else None
        log.update(
            kf_x=err_x,
            kf_y=err_y,
            kf_vx=self.kf_x.vel if kf_ok else None,
            kf_vy=self.kf_y.vel if kf_ok else None,
            err_x=err_x,
            err_y=err_y,
            horiz_err=horiz_err,
        )

        # touchdown detector runs on filtered altitude every tick
        touchdown = self.alt.touchdown_step(dt)

        # mission
        state = self.fsm.update(dt, tag_xy, alt_m, horiz_err, armed, override, touchdown)
        if state is not self._prev_state:
            if state in (S.SEARCH, S.IDLE, S.ABORT):
                self.guidance.reset()
                self._last_cmd = (0.0, 0.0, 0.0)
                self.kf_x.reset()
                self.kf_y.reset()
            if state is S.IDLE:
                self.alt.reset()
            self._prev_state = state
        log["fsm_state"] = state.name

        # safety
        ok = self.safety.check(
            now,
            dt,
            inp.t_attitude,
            inp.altitude.t_mono if inp.altitude else None,
            alt_m,
            horiz_err,
            inp.vbat_v,
            roll,
            pitch,
            armed,
            controlling=state in _VISION_STATES and kf_ok,
        )
        log["safety_trip"] = self.safety.trip_reason or ""

        # horizontal
        out_x = out_y = out_yaw = 0.0
        if state in _VISION_STATES and kf_ok:
            cmd = self.guidance.step(err_x, err_y, dt, self.kf_x.vel, self.kf_y.vel)
            out_x, out_y, out_yaw = cmd.out_x, cmd.out_y, cmd.out_yaw
            self._last_cmd = (out_x, out_y, out_yaw)
        elif state is S.FINAL:
            # Vision is ignored by design, but the last command carries the integrator's
            # estimate of the wind. Neutral sticks here drift the vehicle downwind during the
            # last 20 cm; holding the last command does not.
            out_x, out_y, out_yaw = self._last_cmd
        px, ix, dx = self.guidance.pid_x.last_terms
        py, iy, dy = self.guidance.pid_y.last_terms
        log.update(out_x=out_x, out_y=out_y, pid_p_x=px, pid_i_x=ix, pid_d_x=dx, pid_p_y=py, pid_i_y=iy, pid_d_y=dy)

        # vertical
        vcmd = self.alt.step(inp.altitude, _VERTICAL_MODE[state], horiz_err, dt)
        channels: list[int] | None
        if vcmd is None:
            channels = None  # nothing safe to command without altitude
        else:
            channels = self.rc.channels(out_x, out_y, vcmd.throttle_us, out_yaw)
            log.update(alt_sp_m=vcmd.alt_sp_m, rate_cmd_ms=vcmd.rate_cmd_ms, pid_p_z=vcmd.pid_terms[0], pid_i_z=vcmd.pid_terms[1], pid_d_z=vcmd.pid_terms[2])

        if not ok or state is S.HANDBACK:
            channels = None  # stop transmitting: FC failsafe takes over
        if channels is not None:
            m = self.rc.map
            log.update(rc_roll=channels[m.index("A")], rc_pitch=channels[m.index("E")], rc_throttle=channels[m.index("T")], rc_yaw=channels[m.index("R")])
        log["tx"] = int(channels is not None)
        return Outputs(channels, state, horiz_err, log)


class Node:
    """Wires the loop to the real subsystems: serial link, telemetry thread, pose subscriber, logger."""

    def __init__(self, vcfg: dict, tcfg: dict, port: str, log_path: str | None) -> None:
        s = vcfg["serial"]
        self.link = MSPLink(port, s["baud"], s["timeout_s"])
        self.telem = Telemetry(self.link, vcfg["telemetry"]["rate_hz"])
        self.alt_src = TelemetryAltitudeSource(self.telem)
        self.pose_sub = PoseSubscriber(vcfg["pose"]["udp_host"], vcfg["pose"]["udp_port"])
        self.logger = CSVLogger(log_path) if log_path else None
        self.loop = ControlLoop(vcfg, tcfg)
        self.rc_cfg = vcfg["rc"]
        self.running = False

    def gather(self, now: float) -> Inputs:
        a = self.telem.attitude()
        st = self.telem.status()
        an = self.telem.analog()
        return Inputs(
            now=now,
            poses=self.pose_sub.latest(),
            attitude=a[0] if a else None,
            t_attitude=a[1] if a else None,
            altitude=self.alt_src.read(),
            status=st[0] if st else None,
            vbat_v=an[0]["vbat_v"] if an else None,
        )

    def run(self) -> int:
        self.running = True
        self.telem.start()
        self.pose_sub.start()
        if self.logger:
            self.logger.start()
        period = self.loop.period
        next_t = time.monotonic()
        rc = self.rc_cfg
        try:
            while self.running:
                now = time.monotonic()
                out = self.loop.step(self.gather(now))
                if out.channels is not None:
                    self.link.set_rc(out.channels, rc["min_us"], rc["max_us"], rc["mid_us"])
                if self.logger:
                    self.logger.put(out.log)
                if out.state is S.DISARM:
                    print("[control] landed and disarmed", file=sys.stderr)
                    break
                if out.state is S.HANDBACK:
                    print("[control] HANDBACK: search timed out — transmission stopped, FC failsafe lands", file=sys.stderr)
                    break
                if not self.loop.safety.ok:
                    print(f"[control] SAFETY TRIP: {self.loop.safety.trip_reason} — transmission stopped", file=sys.stderr)
                    break
                next_t += period
                sleep = next_t - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.monotonic()  # overran; resync, don't spiral
        finally:
            self.telem.stop()
            self.pose_sub.stop()
            if self.logger:
                self.logger.stop()
            self.link.close()
        return 0 if self.loop.safety.ok else 2

    def stop(self, *_) -> None:
        self.running = False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/vehicle.yaml")
    ap.add_argument("--tags", default="config/tags.yaml")
    ap.add_argument("--port", default=None, help="serial port; defaults to serial.port in config")
    ap.add_argument("--log", default=None, help="CSV log path")
    args = ap.parse_args(argv)
    vcfg = load_vehicle(args.config)
    tcfg = load_tags(args.tags)
    node = Node(vcfg, tcfg, args.port or vcfg["serial"]["port"], args.log)
    signal.signal(signal.SIGINT, node.stop)
    signal.signal(signal.SIGTERM, node.stop)
    return node.run()


if __name__ == "__main__":
    sys.exit(main())
