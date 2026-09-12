"""Push simulator frames to the C++ vision node's MockSource over local TCP.
Protocol (little-endian): u64 t_capture_us | u32 width | u32 height | width*height gray bytes.
The node then publishes TagPose over UDP exactly as it would from a real camera."""

from __future__ import annotations

import socket
import struct

import numpy as np


class MockSourceClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5700) -> None:
        self.sock = socket.create_connection((host, port))

    def push(self, gray: np.ndarray, t_capture_s: float) -> None:
        h, w = gray.shape
        self.sock.sendall(struct.pack("<QII", int(t_capture_s * 1e6), w, h) + np.ascontiguousarray(gray, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.sock.close()
