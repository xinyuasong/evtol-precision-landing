"""Pure-Python AprilTag detector behind the same interface and wire format as the
C++ node. Lets the repo run and CI pass without a C++ build.

Corner order (detector output and object points) is top-left, top-right,
bottom-right, bottom-left in the tag frame (X right, Y up, Z out of the face).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from control.pose_sub import TagPose

FAMILIES = {"tag36h11": cv2.aruco.DICT_APRILTAG_36h11, "tag25h9": cv2.aruco.DICT_APRILTAG_25h9, "tag16h5": cv2.aruco.DICT_APRILTAG_16h5}


@dataclass
class DetectorConfig:
    family: str = "tag36h11"
    min_marker_perimeter_rate: float = 0.02
    reproj_err_max_px: float = 2.0
    # CORNER_REFINE_APRILTAG costs ~6x SUBPIX on OpenCV 4.13 (86 vs 15 ms on a 640x480
    # frame) for no measurable accuracy gain in sim. SUBPIX is the default.
    corner_refinement: str = "subpix"  # subpix | apriltag | none
    min_marker_distance_rate: float = 0.125  # aruco default; load-bearing under blur and noise


def object_points(size_m: float) -> np.ndarray:
    s = float(size_m) / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)


class PyDetector:
    def __init__(self, K, dist, tag_sizes: dict[int, float], cfg: DetectorConfig | None = None) -> None:
        self.cfg = cfg or DetectorConfig()
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, dtype=np.float64).ravel()
        self.tag_sizes = {int(k): float(v) for k, v in tag_sizes.items()}
        d = cv2.aruco.getPredefinedDictionary(FAMILIES[self.cfg.family])
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = {"subpix": cv2.aruco.CORNER_REFINE_SUBPIX, "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG, "none": cv2.aruco.CORNER_REFINE_NONE}[self.cfg.corner_refinement]
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 23
        p.adaptiveThreshWinSizeStep = 10
        p.minMarkerPerimeterRate = self.cfg.min_marker_perimeter_rate
        p.minMarkerDistanceRate = self.cfg.min_marker_distance_rate
        self._det = cv2.aruco.ArucoDetector(d, p)

    def _reproj(self, obj, img, rvec, tvec) -> float:
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
        return float(np.mean(np.linalg.norm(proj.reshape(4, 2) - img, axis=1)))

    def _solve(self, obj, img):
        """IPPE_SQUARE first. OpenCV's IPPE returns the mirrored solution for an
        exactly fronto-parallel square (observed on 4.13); the reprojection error
        exposes it, and the iterative solver recovers the right pose."""
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return False, None, None, 0.0
        err = self._reproj(obj, img, rvec, tvec)
        if err > self.cfg.reproj_err_max_px:
            ok2, r2, t2 = cv2.solvePnP(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok2:
                e2 = self._reproj(obj, img, r2, t2)
                if e2 < err:
                    rvec, tvec, err = r2, t2, e2
        return True, rvec, tvec, err

    def detect(self, gray: np.ndarray, t_capture: float | None = None) -> list[TagPose]:
        t = time.monotonic() if t_capture is None else float(t_capture)
        corners, ids, _ = self._det.detectMarkers(gray)
        out: list[TagPose] = []
        if ids is None:
            return out
        for c, tid in zip(corners, ids.ravel()):
            tid = int(tid)
            if tid not in self.tag_sizes:
                continue
            obj = object_points(self.tag_sizes[tid])
            img = c.reshape(4, 2).astype(np.float32)
            ok, rvec, tvec, err = self._solve(obj, img)
            if not ok:
                continue
            t3, r3 = tvec.ravel(), rvec.ravel()
            out.append(
                TagPose(t, tid, float(t3[0]), float(t3[1]), float(t3[2]), float(r3[0]), float(r3[1]), float(r3[2]), err, err < self.cfg.reproj_err_max_px)
            )
        return out
