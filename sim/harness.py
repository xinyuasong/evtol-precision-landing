"""Wires the world (dynamics, environment, camera) to the mock FC and to the
control node, and steps everything on one simulated clock.

Two transports for the control node:

  direct  — in-process. ControlLoop.step() is called at its own rate with the
            simulated time; RC and telemetry still go through the MSP codec and
            the mock FC's handle(), only the serial bytes-on-a-wire step is
            skipped. Fast, deterministic; used for Monte Carlo.
  pty     — control/main.py runs as a separate process on a real pseudo-terminal
            and a UDP pose socket. Byte-identical to hardware. Real-time only.

The world is the same in both. tests/test_transport_parity.py checks they agree.
"""

from __future__ import annotations

import copy
import heapq
import math
import os
import pty
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from control import msp
from control.altitude import AltitudeSample
from control.main import ControlLoop, Inputs
from control.msp import Parser
from control.pose_sub import TagPose, pack_pose
from control.state_machine import S
from sim.camera_sim import CameraSim
from sim.dynamics import Dynamics, VehicleParams
from sim.environment import Environment, EnvParams
from sim.fc_model import FCParams, MockFC

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ScenarioSpec:
    name: str
    duration_s: float
    seed: int
    initial_pos: tuple[float, float, float]
    initial_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    arm_at_s: float = 0.5  # override switch flipped; the pilot hovers before this
    physics_hz: float = 400.0
    telemetry_hz: float = 20.0
    control_hz: float = 50.0
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    fc: FCParams = field(default_factory=FCParams)
    env: EnvParams = field(default_factory=EnvParams)
    vehicle_cfg: dict = field(default_factory=dict)  # fully-loaded vehicle.yaml (with overrides) — what control/ sees
    true_R_bc: list | None = None  # how the camera is REALLY mounted; the config may be wrong about it
    tags_cfg: dict = field(default_factory=dict)
    calib_path: str = str(ROOT / "calib" / "camera_640x480.yaml")
    companion_stall_at_s: float | None = None  # control node stops transmitting
    pass_criteria: dict = field(default_factory=dict)


@dataclass
class TickRecord:
    t: float
    state_name: str
    pos: tuple[float, float, float]
    vel: tuple[float, float, float]
    roll: float
    pitch: float
    alt_est: float | None
    horiz_err: float | None
    rc: tuple[int, int, int, int] | None
    tx: bool
    safety: str
    fc_failsafe: bool
    kf: tuple[float | None, float | None]
    pid: tuple


@dataclass
class RunResult:
    name: str
    seed: int
    duration_s: float
    touchdown_err_m: float | None
    touchdown_vz_ms: float | None
    touchdown_t: float | None
    terminal_state: str
    safety_trip: str
    fc_failsafe_used: bool
    fc_failsafe_landed: bool
    max_alt_m: float
    max_horiz_err_m: float
    detection_rate: float
    peak_err_descend_m: float | None  # peak horizontal error while in DESCEND (tracking quality under load)
    t_track_to_touchdown_s: float | None  # first TRACK entry -> touchdown (convergence speed)
    frames: int
    control_ticks: int
    tx_ticks: int
    transitions: list
    records: list[TickRecord]
    transport: str
    detector: str

    @property
    def landed(self) -> bool:
        return self.touchdown_err_m is not None


class World:
    def __init__(self, spec: ScenarioSpec) -> None:
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)
        self.dyn = Dynamics(spec.vehicle)
        self.dyn.reset(spec.initial_pos, spec.initial_vel)
        self.env = Environment(spec.env, np.random.default_rng(spec.seed + 1))
        self.fc = MockFC(spec.fc, np.random.default_rng(spec.seed + 2))
        R_bc = spec.true_R_bc if spec.true_R_bc is not None else spec.vehicle_cfg["camera"]["R_bc"]
        self.cam = CameraSim(spec.calib_path, spec.tags_cfg, R_bc, spec.env.camera, np.random.default_rng(spec.seed + 3))
        self.t = 0.0
        self.dt = 1.0 / spec.physics_hz
        self.fc.arm()  # pilot is already hovering when the run starts
        self.armed = False  # override engaged
        self._cam_next = 0.0
        self._pending: list[tuple[float, int, TagPose]] = []
        self._seq = 0
        self.frames = 0
        self.frames_with_detection = 0

    def step_physics(self) -> None:
        s = self.spec
        if not self.armed and self.t >= s.arm_at_s:
            self.fc.engage_override()
            self.armed = True
        wind = self.env.wind(self.t, self.dt)
        a_alt, a_var = self.env.alt_measurement(self.t, self.dt, self.dyn.state.alt, float(self.dyn.state.vel[2]))
        self.fc.alt_sensor.alt_m, self.fc.alt_sensor.vario_ms = a_alt, a_var
        self.fc.vbat_v = self.env.vbat(self.t)
        self.fc.update(self.t, self.dyn.state)
        roll_c, pitch_c, thrust = self.fc.commands()
        self.dyn.step(self.dt, roll_c, pitch_c, thrust, wind)
        self.t += self.dt

    def camera_due(self) -> bool:
        return self.t >= self._cam_next

    def capture(self) -> list[TagPose]:
        """Render + detect at the camera rate; poses become deliverable after the latency."""
        self._cam_next += 1.0 / self.spec.env.camera.fps
        _, poses = self.cam.observe(self.dyn.state, self.t, self.env, self.env.pad_position(self.t))
        self.frames += 1
        if poses:
            self.frames_with_detection += 1
        for p in poses:
            self._seq += 1
            heapq.heappush(self._pending, (self.t + self.spec.env.camera.latency_s, self._seq, p))
        return poses

    def deliverable_poses(self) -> list[TagPose]:
        out = []
        while self._pending and self._pending[0][0] <= self.t:
            out.append(heapq.heappop(self._pending)[2])
        return out


class DirectRunner:
    def __init__(self, spec: ScenarioSpec) -> None:
        self.spec = spec
        self.world = World(spec)
        self.loop = ControlLoop(spec.vehicle_cfg, spec.tags_cfg)
        self.parser = Parser()
        self.records: list[TickRecord] = []
        self.tx_ticks = 0
        self.control_ticks = 0
        self._telem: dict = {}
        self._latest: dict[int, tuple[TagPose, float]] = {}

    def _msp(self, cmd: int, payload: bytes = b"") -> bytes | None:
        """Round-trip one request through the codec and the FC exactly as the serial path does."""
        reply = self.world.fc.handle(msp.encode_v1(cmd, payload))
        for f in self.parser.feed(reply):
            if f.cmd == cmd and f.direction == msp.DIR_FROM_FC:
                return f.payload
        return None

    def _poll_telemetry(self, t: float) -> None:
        p = self._msp(msp.MSP_ATTITUDE)
        if p is not None:
            self._telem["att"] = (msp.unpack_attitude(p), t)
        p = self._msp(msp.MSP_ALTITUDE)
        if p is not None:
            self._telem["alt"] = (msp.unpack_altitude(p), t)
        p = self._msp(msp.MSP_STATUS)
        if p is not None:
            self._telem["status"] = (msp.unpack_status(p), t)
        p = self._msp(msp.MSP_ANALOG)
        if p is not None:
            self._telem["analog"] = (msp.unpack_analog(p), t)

    def run(self) -> RunResult:
        w, s = self.world, self.spec
        dt_c, dt_t = 1.0 / s.control_hz, 1.0 / s.telemetry_hz
        next_c = next_t = 0.0
        stalled = False
        while w.t < s.duration_s:
            if w.camera_due():
                w.capture()
            for p in w.deliverable_poses():
                self._latest[p.tag_id] = (p, w.t)
            if w.t >= next_t:
                next_t += dt_t
                self._poll_telemetry(w.t)
            if w.t >= next_c:
                next_c += dt_c
                self.control_ticks += 1
                if s.companion_stall_at_s is not None and w.t >= s.companion_stall_at_s:
                    stalled = True
                att = self._telem.get("att")
                alt = self._telem.get("alt")
                st = self._telem.get("status")
                an = self._telem.get("analog")
                inp = Inputs(
                    now=w.t,
                    poses=list(self._latest.values()),
                    attitude=att[0] if att else None,
                    t_attitude=att[1] if att else None,
                    altitude=AltitudeSample(alt[0][0], alt[0][1], alt[1]) if alt else None,
                    status=st[0] if st else None,
                    vbat_v=an[0]["vbat_v"] if an else None,
                )
                out = self.loop.step(inp)
                tx = out.channels is not None and not stalled
                if tx:
                    rc = s.vehicle_cfg["rc"]
                    w.fc.handle(msp.encode_v1(msp.MSP_SET_RAW_RC, msp.pack_rc(out.channels, rc["min_us"], rc["max_us"], rc["mid_us"])))
                    self.tx_ticks += 1
                m = self.loop.rc.map
                rc4 = tuple(out.channels[m.index(c)] for c in "AETR") if out.channels else None
                d = w.dyn.state
                self.records.append(
                    TickRecord(
                        w.t,
                        out.state.name,
                        tuple(d.pos),
                        tuple(d.vel),
                        d.roll_deg,
                        d.pitch_deg,
                        inp.altitude.alt_m if inp.altitude else None,
                        out.horiz_err,
                        rc4,
                        tx,
                        self.loop.safety.trip_reason or "",
                        w.fc.failsafe,
                        (self.loop.kf_x.pos if self.loop.kf_x.valid else None, self.loop.kf_y.pos if self.loop.kf_y.valid else None),
                        (self.loop.guidance.pid_x.last_terms, self.loop.guidance.pid_y.last_terms, self.loop.alt.pid.last_terms),
                    )
                )
                if out.state is S.DISARM or (not w.fc.armed and w.armed and w.dyn.state.on_ground and w.t > s.arm_at_s + 1.0):
                    # landed and disarmed (by the node or by the FC failsafe): run 1 s more and stop
                    s = copy.copy(s)
                    s.duration_s = min(s.duration_s, w.t + 1.0)
            w.step_physics()
        return self._result()

    def _result(self) -> RunResult:
        w, s = self.world, self.spec
        td = w.dyn.touchdown_pos
        pad = w.env.pad_position(w.dyn.touchdown_t) if w.dyn.touchdown_t is not None else np.zeros(3)
        err = float(np.hypot(td[0] - pad[0], td[1] - pad[1])) if td is not None else None
        herr = [r.horiz_err for r in self.records if r.horiz_err is not None]
        derr = [r.horiz_err for r in self.records if r.state_name == "DESCEND" and r.horiz_err is not None]
        t_track = next((t for t, _, b, _ in self.loop.fsm.transitions if b is S.TRACK), None)
        t_td = w.dyn.touchdown_t
        return RunResult(
            name=s.name,
            seed=s.seed,
            duration_s=w.t,
            touchdown_err_m=err,
            touchdown_vz_ms=w.dyn.touchdown_vz,
            touchdown_t=w.dyn.touchdown_t,
            terminal_state=self.loop.fsm.state.name,
            safety_trip=self.loop.safety.trip_reason or "",
            fc_failsafe_used=w.fc.failsafe,
            fc_failsafe_landed=w.fc.landed_by_failsafe,
            max_alt_m=w.dyn.max_alt,
            max_horiz_err_m=max(herr) if herr else 0.0,
            detection_rate=(w.frames_with_detection / w.frames) if w.frames else 0.0,
            peak_err_descend_m=max(derr) if derr else None,
            t_track_to_touchdown_s=(t_td - t_track) if (t_td is not None and t_track is not None) else None,
            frames=w.frames,
            control_ticks=self.control_ticks,
            tx_ticks=self.tx_ticks,
            transitions=[(round(t, 2), a.name, b.name, r) for t, a, b, r in self.loop.fsm.transitions],
            records=self.records,
            transport="direct",
            detector="python",
        )


class PtyRunner:
    """Real-time: control/main.py as a subprocess on a pty + UDP. The world runs on
    the wall clock and the node has no idea it is talking to a simulator."""

    def __init__(self, spec: ScenarioSpec, config_path: str, tags_path: str, log_path: str | None = None) -> None:
        self.spec = spec
        self.world = World(spec)
        self.config_path, self.tags_path, self.log_path = config_path, tags_path, log_path
        self.records: list[TickRecord] = []
        self.rc_frames_seen = 0

    def run(self) -> RunResult:
        w, s = self.world, self.spec
        master, slave = pty.openpty()
        port = os.ttyname(slave)
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_addr = (s.vehicle_cfg["pose"]["udp_host"], s.vehicle_cfg["pose"]["udp_port"])
        stop = threading.Event()

        def serial_thread():
            while not stop.is_set():
                try:
                    data = os.read(master, 4096)
                except OSError:
                    break
                if data:
                    reply = w.fc.handle(data)
                    if reply:
                        os.write(master, reply)

        th = threading.Thread(target=serial_thread, daemon=True)
        th.start()
        cmd = [sys.executable, "-m", "control.main", "--port", port, "--config", self.config_path, "--tags", self.tags_path]
        if self.log_path:
            cmd += ["--log", self.log_path]
        proc = subprocess.Popen(cmd, cwd=ROOT, stderr=subprocess.PIPE, text=True)
        t0 = time.monotonic()
        stalled = False
        last_rec = 0.0
        try:
            while w.t < s.duration_s and proc.poll() is None:
                target = time.monotonic() - t0
                while w.t < target:
                    if w.camera_due():
                        w.capture()
                    for p in w.deliverable_poses():
                        # publisher clock = the node's clock: monotonic. t_capture is sim time, so
                        # translate it onto the wall clock the way a co-located vision node would.
                        wall = TagPose(t0 + p.t_capture, p.tag_id, p.x, p.y, p.z, p.rx, p.ry, p.rz, p.reproj_err, p.valid)
                        udp.sendto(pack_pose(wall), udp_addr)
                    if s.companion_stall_at_s is not None and w.t >= s.companion_stall_at_s and not stalled:
                        proc.send_signal(19)  # SIGSTOP: the companion freezes mid-flight
                        stalled = True
                    w.step_physics()
                    if w.t - last_rec >= 0.02:
                        last_rec = w.t
                        d = w.dyn.state
                        self.records.append(
                            TickRecord(
                                w.t,
                                "?",
                                tuple(d.pos),
                                tuple(d.vel),
                                d.roll_deg,
                                d.pitch_deg,
                                w.fc.alt_sensor.alt_m,
                                None,
                                tuple(w.fc.rc[:4]),
                                False,
                                "",
                                w.fc.failsafe,
                                (None, None),
                                (),
                            )
                        )
                if not w.fc.armed and w.armed and w.dyn.state.on_ground and w.t > s.arm_at_s + 2.0:
                    s = copy.copy(s)
                    s.duration_s = min(s.duration_s, w.t + 1.0)
                time.sleep(0.001)
        finally:
            stop.set()
            if stalled:
                proc.send_signal(18)  # SIGCONT so it can exit
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            stderr = proc.stderr.read() if proc.stderr else ""
            os.close(master)
            os.close(slave)
            udp.close()
        td = w.dyn.touchdown_pos
        err = float(math.hypot(td[0], td[1])) if td is not None else None
        term = "DISARM" if "landed and disarmed" in stderr else ("FAILSAFE" if w.fc.landed_by_failsafe else "UNKNOWN")
        trip = ""
        for line in stderr.splitlines():
            if "SAFETY TRIP" in line:
                trip = line.split("SAFETY TRIP:", 1)[1].split("—")[0].strip()
        return RunResult(
            name=s.name,
            seed=s.seed,
            duration_s=w.t,
            touchdown_err_m=err,
            touchdown_vz_ms=w.dyn.touchdown_vz,
            touchdown_t=w.dyn.touchdown_t,
            terminal_state=term,
            safety_trip=trip,
            fc_failsafe_used=w.fc.failsafe,
            fc_failsafe_landed=w.fc.landed_by_failsafe,
            max_alt_m=w.dyn.max_alt,
            max_horiz_err_m=0.0,
            detection_rate=(w.frames_with_detection / w.frames) if w.frames else 0.0,
            peak_err_descend_m=None,
            t_track_to_touchdown_s=None,
            frames=w.frames,
            control_ticks=0,
            tx_ticks=w.fc.rc_frames,
            transitions=[],
            records=self.records,
            transport="pty",
            detector="python",
        )
