"""20 Hz MSP poll thread. Keeps the latest attitude / altitude / status / analog
with local receive timestamps so the control loop can judge staleness."""

from __future__ import annotations

import threading
import time

from control.msp import MSPLink


class Telemetry:
    def __init__(self, link: MSPLink, rate_hz: float) -> None:
        self.link = link
        self.period = 1.0 / float(rate_hz)
        self._lock = threading.Lock()
        self._att: tuple[tuple[float, float, float], float] | None = None
        self._alt: tuple[tuple[float, float], float] | None = None
        self._status: tuple[dict, float] | None = None
        self._analog: tuple[dict, float] | None = None
        self._running = False
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
        self.polls = 0
        self.failures = 0

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)

    def poll_once(self) -> None:
        """One round of requests. Public so tests can drive it without a thread."""
        self.polls += 1
        a = self.link.attitude()
        t = time.monotonic()
        if a is not None:
            with self._lock:
                self._att = (a, t)
        else:
            self.failures += 1
        h = self.link.altitude()
        t = time.monotonic()
        if h is not None:
            with self._lock:
                self._alt = (h, t)
        # status/analog every other poll: lower priority
        if self.polls % 2 == 0:
            s = self.link.status()
            t = time.monotonic()
            if s is not None:
                with self._lock:
                    self._status = (s, t)
        else:
            v = self.link.analog()
            t = time.monotonic()
            if v is not None:
                with self._lock:
                    self._analog = (v, t)

    def _run(self) -> None:
        next_t = time.monotonic()
        while self._running:
            self.poll_once()
            next_t += self.period
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()

    def attitude(self):
        with self._lock:
            return self._att

    def altitude(self):
        with self._lock:
            return self._alt

    def status(self):
        with self._lock:
            return self._status

    def analog(self):
        with self._lock:
            return self._analog
