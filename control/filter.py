"""Constant-velocity Kalman filter, one instance per horizontal axis.

Gives a velocity estimate the D-term can use and coasts through dropped frames.
Measurement variance scales with range squared: pose noise grows with altitude.
"""

from __future__ import annotations

import numpy as np


class CVKalman:
    def __init__(self, q: float, r0: float, z_ref_m: float, coast_max_s: float) -> None:
        self.q = float(q)
        self.r0 = float(r0)
        self.z_ref = float(z_ref_m)
        self.coast_max = float(coast_max_s)
        self.reset()

    def reset(self) -> None:
        self.x = np.zeros(2)
        self.P = np.eye(2)
        self.initialised = False
        self.t_since_meas = 0.0

    def predict(self, dt: float) -> None:
        if not self.initialised or dt <= 0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        G = np.array([0.5 * dt * dt, dt])
        Q = np.outer(G, G) * self.q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t_since_meas += dt

    def update(self, z: float, range_m: float) -> None:
        if not self.initialised:
            self.x = np.array([float(z), 0.0])
            self.P = np.diag([self.r0, 1.0])
            self.initialised = True
            self.t_since_meas = 0.0
            return
        r = self.r0 * (max(range_m, 1e-3) / self.z_ref) ** 2
        H = np.array([1.0, 0.0])
        y = z - H @ self.x
        S = H @ self.P @ H + r
        K = (self.P @ H) / S
        self.x = self.x + K * y
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        self.t_since_meas = 0.0

    @property
    def valid(self) -> bool:
        return self.initialised and self.t_since_meas <= self.coast_max

    @property
    def pos(self) -> float:
        return float(self.x[0])

    @property
    def vel(self) -> float:
        return float(self.x[1])


class LowPass:
    """First-order low-pass with a time constant. Also exposes the filtered derivative."""

    def __init__(self, tau_s: float) -> None:
        self.tau = float(tau_s)
        self.y: float | None = None
        self.dy = 0.0

    def reset(self) -> None:
        self.y = None
        self.dy = 0.0

    def step(self, x: float, dt: float) -> float:
        if self.y is None or dt <= 0:
            self.y = float(x)
            self.dy = 0.0
            return self.y
        alpha = dt / (self.tau + dt)
        prev = self.y
        self.y = prev + alpha * (float(x) - prev)
        self.dy = (self.y - prev) / dt
        return self.y
