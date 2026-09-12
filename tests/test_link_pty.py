"""MSPLink and Telemetry over a real serial device (a pty), answered by a minimal
MSP responder that lives entirely in this test. Proves the transport path the
control node uses on hardware, byte for byte."""

import os
import pty
import threading
import time

import pytest

from control import msp
from control.msp import DIR_FROM_FC, MSPLink, Parser
from control.telemetry import Telemetry


class Responder(threading.Thread):
    def __init__(self, fd):
        super().__init__(daemon=True)
        self.fd = fd
        self.rc_frames = []
        self.stop = False
        self.parser = Parser()

    def run(self):
        while not self.stop:
            try:
                data = os.read(self.fd, 256)
            except OSError:
                break
            for f in self.parser.feed(data):
                if f.cmd == msp.MSP_ATTITUDE:
                    os.write(self.fd, msp.encode_v1(f.cmd, msp.pack_attitude(2.5, -1.0, 180), DIR_FROM_FC))
                elif f.cmd == msp.MSP_ALTITUDE:
                    os.write(self.fd, msp.encode_v1(f.cmd, msp.pack_altitude(1.75, 0.2), DIR_FROM_FC))
                elif f.cmd == msp.MSP_STATUS:
                    os.write(self.fd, msp.encode_v1(f.cmd, msp.pack_status(250, 0, 1, 0b11, 0), DIR_FROM_FC))
                elif f.cmd == msp.MSP_ANALOG:
                    os.write(self.fd, msp.encode_v1(f.cmd, msp.pack_analog(23.7, 100, 50, 5.0), DIR_FROM_FC))
                elif f.cmd == msp.MSP_SET_RAW_RC:
                    self.rc_frames.append(msp.unpack_rc(f.payload))


@pytest.fixture
def link():
    master, slave = pty.openpty()
    r = Responder(master)
    r.start()
    lk = MSPLink(os.ttyname(slave), 115200, 0.05)
    yield lk, r
    r.stop = True
    lk.close()
    os.close(master)


def test_request_response_over_pty(link):
    lk, _ = link
    assert lk.attitude() == (2.5, -1.0, 180.0)
    assert lk.altitude() == (1.75, 0.2)
    assert lk.status()["mode_flags"] == 0b11
    assert lk.analog()["vbat_v"] == 23.7


def test_rc_stream_arrives_intact(link):
    lk, r = link
    for i in range(20):
        lk.set_rc([1500 + i, 1400, 1000 + i, 1500])
    time.sleep(0.1)
    assert len(r.rc_frames) == 20
    assert r.rc_frames[-1][:4] == [1519, 1400, 1019, 1500]


def test_telemetry_poll_populates_all_fields(link):
    lk, _ = link
    t = Telemetry(lk, rate_hz=20)
    t.poll_once()
    t.poll_once()
    assert t.attitude()[0] == (2.5, -1.0, 180.0)
    assert t.altitude()[0] == (1.75, 0.2)
    assert t.status()[0]["mode_flags"] == 0b11
    assert t.analog()[0]["vbat_v"] == 23.7
    assert t.failures == 0
