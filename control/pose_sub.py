"""UDP subscriber for TagPose messages from the vision node.

Wire format (little-endian, 70 bytes), shared with vision/include/tag_pose.hpp:
    u64 t_capture_us | i32 tag_id | f64 x y z | f64 rx ry rz (Rodrigues, tag->camera) | f64 reproj_err | u8 valid | u8 pad
Keeps only the newest message per tag id. Never queues.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time
from dataclasses import dataclass

WIRE_FMT = "<QidddddddBx"
WIRE_SIZE = struct.calcsize(WIRE_FMT)


@dataclass(frozen=True)
class TagPose:
    t_capture: float  # seconds, monotonic clock of the publisher
    tag_id: int
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    reproj_err: float
    valid: bool

    @property
    def yaw(self) -> float:
        """Rotation about the camera Z axis, for logging."""
        from control.frames import rodrigues

        R = rodrigues((self.rx, self.ry, self.rz))
        return float(math.atan2(R[1, 0], R[0, 0]))


def pack_pose(p: TagPose) -> bytes:
    return struct.pack(WIRE_FMT, int(p.t_capture * 1e6), p.tag_id, p.x, p.y, p.z, p.rx, p.ry, p.rz, p.reproj_err, 1 if p.valid else 0)


def unpack_pose(b: bytes) -> TagPose:
    t_us, tid, x, y, z, rx, ry, rz, err, valid = struct.unpack(WIRE_FMT, b[:WIRE_SIZE])
    return TagPose(t_us / 1e6, tid, x, y, z, rx, ry, rz, err, bool(valid))


class PoseSubscriber:
    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.settimeout(0.1)
        self._lock = threading.Lock()
        self._latest: dict[int, tuple[TagPose, float]] = {}
        self._running = False
        self._thread = threading.Thread(target=self._run, name="pose_sub", daemon=True)
        self.received = 0

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.sock.close()

    def _run(self) -> None:
        while self._running:
            try:
                data, _ = self.sock.recvfrom(256)
            except TimeoutError:
                continue
            except OSError:
                break
            if len(data) < WIRE_SIZE:
                continue
            p = unpack_pose(data)
            with self._lock:
                self._latest[p.tag_id] = (p, time.monotonic())
                self.received += 1

    def latest(self) -> list[tuple[TagPose, float]]:
        """Newest pose per tag id with its local receive time."""
        with self._lock:
            return list(self._latest.values())
