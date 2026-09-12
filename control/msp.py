"""MSP (MultiWii Serial Protocol) v1/v2 codec and a thread-safe serial transport.

The codec is symmetric: the same encoder/parser serves a client (this node) or a
server (anything that answers on the other end of the wire). Field layouts follow
INAV's msp_protocol.h; verify against your firmware tree before trusting them.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass

MSP_STATUS = 101
MSP_RC = 105
MSP_RAW_GPS = 106
MSP_ATTITUDE = 108
MSP_ALTITUDE = 109
MSP_ANALOG = 110
MSP_SET_RAW_RC = 200

DIR_TO_FC = ord("<")
DIR_FROM_FC = ord(">")
DIR_ERROR = ord("!")

MAX_RC_CHANNELS = 16


class MSPError(Exception):
    pass


def crc8_dvb_s2(crc: int, byte: int) -> int:
    crc ^= byte
    for _ in range(8):
        crc = ((crc << 1) ^ 0xD5) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def xor_checksum(size: int, cmd: int, payload: bytes) -> int:
    chk = size ^ cmd
    for b in payload:
        chk ^= b
    return chk


def encode_v1(cmd: int, payload: bytes = b"", direction: int = DIR_TO_FC) -> bytes:
    size = len(payload)
    if size > 255:
        raise MSPError("payload too large for MSP v1")
    if not 0 <= cmd <= 255:
        raise MSPError("command id out of range for MSP v1")
    frame = bytearray(b"$M")
    frame.append(direction)
    frame.append(size)
    frame.append(cmd)
    frame += payload
    frame.append(xor_checksum(size, cmd, payload))
    return bytes(frame)


def encode_v2(function: int, payload: bytes = b"", flag: int = 0, direction: int = DIR_TO_FC) -> bytes:
    if len(payload) > 0xFFFF:
        raise MSPError("payload too large for MSP v2")
    body = struct.pack("<BHH", flag, function, len(payload)) + payload
    crc = 0
    for b in body:
        crc = crc8_dvb_s2(crc, b)
    return b"$X" + bytes([direction]) + body + bytes([crc])


@dataclass(frozen=True)
class Frame:
    version: int  # 1 or 2
    direction: int  # DIR_TO_FC / DIR_FROM_FC / DIR_ERROR
    cmd: int
    payload: bytes
    flag: int = 0


class Parser:
    """Incremental MSP parser. Feed bytes, get frames. Corrupt frames are dropped
    and counted; the parser resynchronises on the next '$'."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self.bad_frames = 0

    def feed(self, data: bytes) -> list[Frame]:
        self._buf += data
        out: list[Frame] = []
        while True:
            start = self._buf.find(b"$")
            if start < 0:
                self._buf.clear()
                break
            if start > 0:
                del self._buf[:start]
            if len(self._buf) < 3:
                break
            kind = self._buf[1]
            if kind == ord("M"):
                consumed, frame = self._try_v1()
            elif kind == ord("X"):
                consumed, frame = self._try_v2()
            else:
                del self._buf[:1]
                continue
            if consumed == 0:
                break  # need more bytes
            del self._buf[:consumed]
            if frame is not None:
                out.append(frame)
        return out

    def _try_v1(self) -> tuple[int, Frame | None]:
        b = self._buf
        if len(b) < 6:
            return 0, None
        direction, size, cmd = b[2], b[3], b[4]
        if direction not in (DIR_TO_FC, DIR_FROM_FC, DIR_ERROR):
            self.bad_frames += 1
            return 1, None
        total = 6 + size
        if len(b) < total:
            return 0, None
        payload = bytes(b[5 : 5 + size])
        if xor_checksum(size, cmd, payload) != b[5 + size]:
            self.bad_frames += 1
            return 1, None
        return total, Frame(1, direction, cmd, payload)

    def _try_v2(self) -> tuple[int, Frame | None]:
        b = self._buf
        if len(b) < 9:
            return 0, None
        direction = b[2]
        if direction not in (DIR_TO_FC, DIR_FROM_FC, DIR_ERROR):
            self.bad_frames += 1
            return 1, None
        flag, function, size = struct.unpack_from("<BHH", b, 3)
        total = 9 + size
        if len(b) < total:
            return 0, None
        crc = 0
        for x in b[3 : 8 + size]:
            crc = crc8_dvb_s2(crc, x)
        if crc != b[8 + size]:
            self.bad_frames += 1
            return 1, None
        return total, Frame(2, direction, function, bytes(b[8 : 8 + size]), flag)


# ---- payload codecs (shared by client and any MSP server) ----


def pack_rc(channels: list[int], rc_min: int = 1000, rc_max: int = 2000, mid: int = 1500) -> bytes:
    ch = [int(c) for c in channels][:MAX_RC_CHANNELS]
    ch += [mid] * (max(0, 8 - len(ch)))
    ch = [max(rc_min, min(rc_max, c)) for c in ch]
    return struct.pack(f"<{len(ch)}H", *ch)


def unpack_rc(payload: bytes) -> list[int]:
    n = len(payload) // 2
    return list(struct.unpack(f"<{n}H", payload[: 2 * n]))


def pack_attitude(roll_deg: float, pitch_deg: float, yaw_deg: float) -> bytes:
    return struct.pack("<hhh", int(round(roll_deg * 10)), int(round(pitch_deg * 10)), int(round(yaw_deg)))


def unpack_attitude(payload: bytes) -> tuple[float, float, float]:
    roll_dd, pitch_dd, yaw_d = struct.unpack("<hhh", payload[:6])
    return roll_dd / 10.0, pitch_dd / 10.0, float(yaw_d)


def pack_altitude(alt_m: float, vario_ms: float) -> bytes:
    return struct.pack("<ih", int(round(alt_m * 100)), int(round(vario_ms * 100)))


def unpack_altitude(payload: bytes) -> tuple[float, float]:
    alt_cm, vario_cms = struct.unpack("<ih", payload[:6])
    return alt_cm / 100.0, vario_cms / 100.0


def pack_status(cycle_us: int, i2c_err: int, sensors: int, mode_flags: int, profile: int) -> bytes:
    return struct.pack("<HHHIB", cycle_us, i2c_err, sensors, mode_flags, profile)


def unpack_status(payload: bytes) -> dict:
    cycle, i2c_err, sensors, mode_flags, profile = struct.unpack("<HHHIB", payload[:11])
    return {"cycle_us": cycle, "i2c_err": i2c_err, "sensors": sensors, "mode_flags": mode_flags, "profile": profile}


def pack_analog(vbat_v: float, mah: int, rssi: int, amps: float) -> bytes:
    return struct.pack("<BHHh", int(round(vbat_v * 10)), mah, rssi, int(round(amps * 100)))


def unpack_analog(payload: bytes) -> dict:
    vbat, mah, rssi, amps = struct.unpack("<BHHh", payload[:7])
    return {"vbat_v": vbat / 10.0, "mah": mah, "rssi": rssi, "amps": amps / 100.0}


# ---- transport ----


class MSPLink:
    """Thread-safe request/response client over a pyserial-compatible port.

    Any object with read(n)/write(b)/reset_input_buffer() works; a real serial
    port is the default, and the port path is whatever the config says it is.
    """

    def __init__(self, port: str, baud: int, timeout_s: float, ser=None) -> None:
        if ser is None:
            import serial  # local import: keeps the codec importable without pyserial

            ser = serial.Serial(port, baud, timeout=timeout_s)
        self.ser = ser
        self.timeout_s = timeout_s
        # Two locks: writes are short and must never wait behind a request that is
        # blocked reading (the 50 Hz RC stream shares this port with the 20 Hz poll thread).
        self._wlock = threading.Lock()
        self._rlock = threading.Lock()
        self._parser = Parser()

    def close(self) -> None:
        self.ser.close()

    def send(self, cmd: int, payload: bytes = b"") -> None:
        with self._wlock:
            self.ser.write(encode_v1(cmd, payload))

    def request(self, cmd: int, expect_len: int | None = None, retries: int = 2) -> bytes | None:
        for _ in range(retries):
            with self._rlock:
                self.ser.reset_input_buffer()
                self._parser = Parser()
                self.send(cmd)
                frame = self._read_reply(cmd)
            if frame is not None and (expect_len is None or len(frame.payload) >= expect_len):
                return frame.payload
        return None

    def _read_reply(self, cmd: int) -> Frame | None:
        deadline = time.monotonic() + 2 * self.timeout_s + 0.05
        while time.monotonic() < deadline:
            data = self.ser.read(64)
            if not data:
                continue
            for fr in self._parser.feed(data):
                if fr.direction == DIR_FROM_FC and fr.cmd == cmd:
                    return fr
                if fr.direction == DIR_ERROR and fr.cmd == cmd:
                    return None
        return None

    # convenience wrappers

    def set_rc(self, channels: list[int], rc_min: int = 1000, rc_max: int = 2000, mid: int = 1500) -> None:
        self.send(MSP_SET_RAW_RC, pack_rc(channels, rc_min, rc_max, mid))

    def attitude(self) -> tuple[float, float, float] | None:
        p = self.request(MSP_ATTITUDE, 6)
        return None if p is None else unpack_attitude(p)

    def altitude(self) -> tuple[float, float] | None:
        p = self.request(MSP_ALTITUDE, 6)
        return None if p is None else unpack_altitude(p)

    def status(self) -> dict | None:
        p = self.request(MSP_STATUS, 11)
        return None if p is None else unpack_status(p)

    def analog(self) -> dict | None:
        p = self.request(MSP_ANALOG, 7)
        return None if p is None else unpack_analog(p)
