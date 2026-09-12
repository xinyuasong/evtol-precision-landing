"""Renders the nested tag pad as the downward camera would see it, then runs a
real detector on the rendered image. Ground truth never enters the pose path.

Geometry matches control/frames.py exactly, run backwards:
    world -> level (Z flipped) -> body (R_lb^T) -> camera (R_bc^T) -> image (K, dist)
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import yaml

from control.frames import R_x, R_y
from control.pose_sub import TagPose
from sim.dynamics import VehicleState
from sim.environment import CameraParams, Environment
from vision.py.detector_py import FAMILIES, DetectorConfig, PyDetector

QUIET_CELLS = 1  # white quiet zone width, in tag cells
MARKER_CELLS = 8  # 36h11: 6 data + 2 border


def load_calib(path) -> tuple[np.ndarray, np.ndarray, int, int]:
    with open(path) as f:
        c = yaml.safe_load(f)
    return np.array(c["camera_matrix"], float), np.array(c["distortion_coefficients"], float), int(c["image_width"]), int(c["image_height"])


class CameraSim:
    def __init__(self, calib_path, tags_cfg: dict, R_bc, params: CameraParams, rng: np.random.Generator, detector=None) -> None:
        self.K, self.dist, self.w, self.h = load_calib(calib_path)
        self.p = params
        self.rng = rng
        self.R_bc = np.asarray(R_bc, float).reshape(3, 3)
        self.tags = tags_cfg["tags"]
        self.family = tags_cfg["family"]
        sizes = {int(t["id"]): float(t["size_m"]) for t in self.tags}
        self.detector = detector or PyDetector(self.K, self.dist, sizes, DetectorConfig(family=self.family))
        d = cv2.aruco.getPredefinedDictionary(FAMILIES[self.family])
        cells = MARKER_CELLS + 2 * QUIET_CELLS
        self.bitmaps: dict[int, np.ndarray] = {}
        for t in self.tags:
            px = 40
            m = cv2.aruco.generateImageMarker(d, int(t["id"]), MARKER_CELLS * px)
            full = np.full((cells * px, cells * px), 255, np.uint8)
            full[QUIET_CELLS * px : (QUIET_CELLS + MARKER_CELLS) * px, QUIET_CELLS * px : (QUIET_CELLS + MARKER_CELLS) * px] = m
            self.bitmaps[int(t["id"])] = full
        # white board: bounding box of all tag footprints plus a margin
        fp = [
            (float(t["offset_xy_m"][1]), float(t["offset_xy_m"][0]), float(t["size_m"]) * cells / MARKER_CELLS / 2) for t in self.tags
        ]  # (world x, world y, half)
        m = 0.15  # white margin around the tags; a tag hard against the board edge gets merged with the board quad
        self.board = (min(x - h for x, _, h in fp) - m, max(x + h for x, _, h in fp) + m, min(y - h for _, y, h in fp) - m, max(y + h for _, y, h in fp) + m)
        self.frames = 0
        self.detections = 0

    # ---- geometry ----

    def world_to_cam(self, state: VehicleState, q_world: np.ndarray) -> np.ndarray:
        d = np.asarray(q_world, float) - state.pos
        p_level = np.array([d[0], d[1], -d[2]])
        R_lb = R_y(math.radians(state.pitch_deg)) @ R_x(math.radians(state.roll_deg))
        p_body = R_lb.T @ p_level
        return self.R_bc.T @ p_body

    def tag_corners_world(self, tag: dict, pad_pos: np.ndarray, half_scale: float = 1.0) -> np.ndarray:
        s = float(tag["size_m"]) / 2 * half_scale
        ox, oy = tag["offset_xy_m"]
        # Tag frame (right, up, out) embedded as a proper rotation in the world:
        # printed-right = world +Y (vehicle right), printed-up = world +X (forward), out = +Z.
        # Corner order TL, TR, BR, BL in the tag frame, matching the detector's object points.
        c_tag = np.array([[-s, s], [s, s], [s, -s], [-s, -s]], float) + np.array([ox, oy])
        return np.stack([c_tag[:, 1], c_tag[:, 0], np.zeros(4)], axis=1) + pad_pos

    def ground_truth(self, state: VehicleState, pad_pos=None) -> dict[int, np.ndarray]:
        """Tag origins in the camera frame — for tests only."""
        pad_pos = np.zeros(3) if pad_pos is None else pad_pos
        return {int(t["id"]): self.world_to_cam(state, np.array([t["offset_xy_m"][1], t["offset_xy_m"][0], 0.0]) + pad_pos) for t in self.tags}

    def project(self, p_cam: np.ndarray) -> np.ndarray | None:
        if np.any(p_cam[:, 2] <= 0.05):
            return None
        pts, _ = cv2.projectPoints(p_cam.astype(np.float64), np.zeros(3), np.zeros(3), self.K, self.dist)
        return pts.reshape(-1, 2).astype(np.float32)

    # ---- rendering ----

    def render(self, state: VehicleState, t: float, env: Environment | None = None, pad_pos=None) -> np.ndarray | None:
        p = self.p
        pad_pos = np.zeros(3) if pad_pos is None else np.asarray(pad_pos, float)
        img = np.full((self.h, self.w), 110, np.uint8)  # ground
        # pad board (white)
        x0, x1, y0, y1 = self.board
        board = np.array([[x0, y1, 0], [x1, y1, 0], [x1, y0, 0], [x0, y0, 0]], float) + pad_pos
        bp = self.project(np.array([self.world_to_cam(state, q) for q in board]))
        if bp is not None:
            cv2.fillConvexPoly(img, bp.astype(np.int32), 235)
        occluded = env is not None and env.occluded(t, state.alt)
        if not occluded:
            for tag in sorted(self.tags, key=lambda x: -float(x["size_m"])):
                cells = MARKER_CELLS + 2 * QUIET_CELLS
                corners = self.tag_corners_world(tag, pad_pos, cells / MARKER_CELLS)
                ip = self.project(np.array([self.world_to_cam(state, q) for q in corners]))
                if ip is None:
                    continue
                # cull if entirely outside the frame
                if ip[:, 0].max() < 0 or ip[:, 1].max() < 0 or ip[:, 0].min() > self.w or ip[:, 1].min() > self.h:
                    continue
                bm = self.bitmaps[int(tag["id"])]
                side_px = float(np.max(np.linalg.norm(np.roll(ip, -1, axis=0) - ip, axis=1)))
                if side_px < 6:
                    continue
                if side_px < bm.shape[0] / 2:  # pre-shrink for tiny tags: avoids aliasing
                    k = max(1, int(bm.shape[0] / side_px))
                    bm = cv2.resize(bm, (bm.shape[1] // k, bm.shape[0] // k), interpolation=cv2.INTER_AREA)
                n = bm.shape[0]
                src = np.array([[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]], np.float32)
                H = cv2.getPerspectiveTransform(src, ip)
                warped = cv2.warpPerspective(bm, H, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                mask = cv2.warpPerspective(np.full_like(bm, 255), H, (self.w, self.h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                img[mask > 0] = warped[mask > 0]
        # motion blur from horizontal speed + angular rate
        alt = max(state.alt, 0.05)
        fx = self.K[0, 0]
        v_img = np.array([state.vel[1], -state.vel[0]]) / alt * fx  # camera x = body y; camera y = -body x
        omega = np.radians([state.pitch_rate_dps, state.roll_rate_dps]) * fx
        flow = v_img + omega
        L = float(np.linalg.norm(flow)) * p.exposure_s
        if L >= 1.5:
            k = int(min(31, L))
            kern = np.zeros((k, k), np.float32)
            ang = math.atan2(flow[1], flow[0])
            c = k // 2
            for i in range(k):
                x = int(round(c + (i - c) * math.cos(ang)))
                y = int(round(c + (i - c) * math.sin(ang)))
                kern[min(k - 1, max(0, y)), min(k - 1, max(0, x))] = 1
            kern /= kern.sum()
            img = cv2.filter2D(img, -1, kern)
        # rolling shutter shear: horizontal row shift growing down the frame, driven by vibration
        if p.rolling_shutter_shear_px > 0:
            shear = p.rolling_shutter_shear_px * math.sin(2 * math.pi * p.vibration_hz * t)
            M = np.array([[1, shear / self.h, -shear / 2], [0, 1, 0]], np.float32)
            img = cv2.warpAffine(img, M, (self.w, self.h), borderMode=cv2.BORDER_REPLICATE)
        # lighting + sensor noise
        f = img.astype(np.float32) * p.gain + p.offset
        if p.noise_sigma > 0:
            f += self.rng.normal(0, p.noise_sigma, f.shape).astype(np.float32)
        return np.clip(f, 0, 255).astype(np.uint8)

    def observe(self, state: VehicleState, t: float, env: Environment | None = None, pad_pos=None) -> tuple[np.ndarray | None, list[TagPose]]:
        if env is not None and env.frame_dropped():
            return None, []
        img = self.render(state, t, env, pad_pos)
        self.frames += 1
        poses = self.detector.detect(img, t)
        self.detections += len(poses)
        return img, poses
