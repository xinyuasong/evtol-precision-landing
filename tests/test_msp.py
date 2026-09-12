import struct

import pytest

from control import msp
from control.msp import DIR_FROM_FC, DIR_TO_FC, MSPLink, Parser, crc8_dvb_s2


def test_v1_roundtrip_all_payload_sizes():
    for n in (0, 1, 7, 16, 255):
        payload = bytes(range(n)) if n <= 255 else b""
        payload = bytes(i & 0xFF for i in range(n))
        frame = msp.encode_v1(msp.MSP_SET_RAW_RC, payload)
        got = Parser().feed(frame)
        assert len(got) == 1
        f = got[0]
        assert (f.version, f.direction, f.cmd, f.payload) == (1, DIR_TO_FC, msp.MSP_SET_RAW_RC, payload)


def test_v1_xor_checksum_known_vector():
    # $M< size=0 cmd=101 chk=0^101=101
    assert msp.encode_v1(101) == b"$M<\x00\x65\x65"


def test_v1_payload_too_large():
    with pytest.raises(msp.MSPError):
        msp.encode_v1(200, bytes(256))


def test_v2_roundtrip():
    payload = b"\x01\x02\x03\xff"
    frame = msp.encode_v2(0x2001, payload, flag=0, direction=DIR_FROM_FC)
    got = Parser().feed(frame)
    assert len(got) == 1
    f = got[0]
    assert (f.version, f.direction, f.cmd, f.payload, f.flag) == (2, DIR_FROM_FC, 0x2001, payload, 0)


def test_crc8_dvb_s2_known_vector():
    # CRC-8/DVB-S2 of "123456789" is 0xBC (standard check value)
    crc = 0
    for b in b"123456789":
        crc = crc8_dvb_s2(crc, b)
    assert crc == 0xBC


def test_corrupt_v1_checksum_dropped_and_resync():
    good = msp.encode_v1(108, b"\x01\x02\x03\x04\x05\x06")
    bad = bytearray(good)
    bad[-1] ^= 0xFF
    p = Parser()
    got = p.feed(bytes(bad) + good)
    assert len(got) == 1 and got[0].payload == b"\x01\x02\x03\x04\x05\x06"
    assert p.bad_frames == 1


def test_corrupt_v2_crc_dropped():
    good = msp.encode_v2(0x1234, b"abc")
    bad = bytearray(good)
    bad[-1] ^= 0x01
    p = Parser()
    assert p.feed(bytes(bad)) == []
    assert p.bad_frames == 1


def test_parser_handles_split_and_garbage():
    f1 = msp.encode_v1(101, b"\x00" * 11, DIR_FROM_FC)
    f2 = msp.encode_v2(0x10, b"xyz", direction=DIR_FROM_FC)
    stream = b"\x00garbage$$M" + f1 + b"$Xnoise" + f2
    p = Parser()
    out = []
    for i in range(0, len(stream), 3):  # feed in 3-byte chunks
        out += p.feed(stream[i : i + 3])
    assert [(f.version, f.cmd) for f in out] == [(1, 101), (2, 0x10)]


def test_rc_pack_pads_and_clamps():
    b = msp.pack_rc([900, 2100, 1500])
    ch = msp.unpack_rc(b)
    assert ch == [1000, 2000, 1500, 1500, 1500, 1500, 1500, 1500]
    assert len(msp.unpack_rc(msp.pack_rc([1500] * 20))) == 16


def test_payload_codecs_roundtrip():
    assert msp.unpack_attitude(msp.pack_attitude(-12.3, 4.5, 270)) == (-12.3, 4.5, 270.0)
    assert msp.unpack_altitude(msp.pack_altitude(3.21, -0.45)) == (3.21, -0.45)
    s = msp.unpack_status(msp.pack_status(250, 0, 0b111, 0b11, 1))
    assert s["cycle_us"] == 250 and s["mode_flags"] == 0b11
    a = msp.unpack_analog(msp.pack_analog(23.4, 1200, 99, 12.5))
    assert a["vbat_v"] == 23.4 and a["amps"] == 12.5


class FakeSerial:
    """Answers MSP_ATTITUDE, ignores everything else, and can inject noise."""

    def __init__(self, noise=b""):
        self.rx = bytearray()
        self.written = []
        self.noise = noise

    def write(self, b):
        self.written.append(b)
        for f in Parser().feed(b):
            if f.cmd == msp.MSP_ATTITUDE:
                self.rx += self.noise + msp.encode_v1(msp.MSP_ATTITUDE, msp.pack_attitude(1.5, -2.0, 90), DIR_FROM_FC)

    def read(self, n):
        out = bytes(self.rx[:n])
        del self.rx[:n]
        return out

    def reset_input_buffer(self):
        self.rx.clear()

    def close(self):
        pass


def test_link_request_and_send():
    link = MSPLink("unused", 115200, 0.01, ser=FakeSerial(noise=b"\xff$M>\x01"))
    assert link.attitude() == (1.5, -2.0, 90.0)
    assert link.altitude() is None  # fake never answers -> retries then None
    link.set_rc([1500, 1500, 1000, 1500])
    last = link.ser.written[-1]
    f = Parser().feed(last)[0]
    assert f.cmd == msp.MSP_SET_RAW_RC and struct.unpack("<8H", f.payload)[2] == 1000
