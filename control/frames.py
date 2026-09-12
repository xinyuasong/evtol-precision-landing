"""Coordinate frames.

Camera (OpenCV): X right, Y down, Z forward out of the lens.
Body (FRD):      X forward (nose), Y right, Z down.
Level:           body de-rotated by roll and pitch so Z is along gravity.
                 X still points along the vehicle's heading (not north).

SIGN CONVENTION, written down once:
    The tag's position in the level frame is the displacement the vehicle must
    travel. err = p_level[:2]. A positive err_x means "move forward".
    Inverting this makes the vehicle accelerate away from the pad at full authority.
"""

from __future__ import annotations

import numpy as np


def R_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def R_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def R_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


class FrameTransformer:
    def __init__(self, R_bc) -> None:
        self.R_bc = np.asarray(R_bc, dtype=float).reshape(3, 3)
        # sanity: must be a rotation
        if not np.allclose(self.R_bc @ self.R_bc.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(self.R_bc), 1.0):
            raise ValueError("R_bc is not a proper rotation matrix")

    def camera_to_body(self, p_cam) -> np.ndarray:
        return self.R_bc @ np.asarray(p_cam, dtype=float)

    def body_to_level(self, p_body, roll_deg: float, pitch_deg: float) -> np.ndarray:
        """Rotate a body-frame vector into the gravity-levelled frame.

        Body = R_x(roll) R_y(pitch) applied to level (aerospace ZYX with yaw
        dropped), so level = R_y(pitch) R_x(roll) body.
        """
        R_lb = R_y(np.radians(pitch_deg)) @ R_x(np.radians(roll_deg))
        return R_lb @ np.asarray(p_body, dtype=float)

    def camera_to_level(self, p_cam, roll_deg: float, pitch_deg: float) -> np.ndarray:
        return self.body_to_level(self.camera_to_body(p_cam), roll_deg, pitch_deg)


def rodrigues(rvec) -> np.ndarray:
    """Rotation vector -> rotation matrix (no OpenCV dependency in control/)."""
    r = np.asarray(rvec, dtype=float).reshape(3)
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def apply_tag_offset(p_cam, rvec, offset_xy_m) -> np.ndarray:
    """Move a detected tag's origin to the pad origin.

    offset_xy_m is the tag's position in the pad frame (X right, Y up on the
    print, shared by every tag on the pad). rvec is the tag->camera rotation
    the detector solved. The pad origin in the camera frame is the tag origin
    minus the offset expressed in camera coordinates.
    """
    off_tag = np.array([float(offset_xy_m[0]), float(offset_xy_m[1]), 0.0])
    return np.asarray(p_cam, dtype=float) - rodrigues(rvec) @ off_tag
