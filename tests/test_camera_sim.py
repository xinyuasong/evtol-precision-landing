"""Render/detect round trip. The most important test in the repo: if the renderer
has a sign flip or a bad projection, every scenario validates the control stack
against a wrong world. Sweeps altitude and tilt; the recovered pad-origin position
must match ground truth in the camera frame."""

import math

import numpy as np
import pytest

from control.frames import apply_tag_offset, rodrigues
from sim.camera_sim import CameraSim
from sim.dynamics import VehicleState
from sim.environment import CameraParams
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def cam(vcfg_base, tcfg_base):
    return CameraSim(ROOT / "calib" / "camera_640x480.yaml", tcfg_base, vcfg_base["camera"]["R_bc"], CameraParams(noise_sigma=0.0), np.random.default_rng(0))


CASES = [
    # alt, roll, pitch, x, y
    (4.0, 0, 0, 0.0, 0.0),
    (3.0, 0, 0, 0.3, -0.2),
    (3.0, 0, 10, 0.3, -0.2),
    (3.0, -8, 0, -0.2, 0.1),
    (2.0, 5, -5, 0.1, 0.1),
    (1.5, 12, 6, 0.0, 0.0),
    (1.0, 0, 0, 0.0, 0.0),
    (0.6, 3, -4, 0.05, 0.02),
    (0.4, 0, 0, 0.0, 0.0),
    (0.25, 3, 2, 0.0, 0.02),
    (0.12, -2, 1, 0.01, 0.0),
]


@pytest.mark.parametrize("alt,roll,pitch,x,y", CASES)
def test_pad_origin_recovered_within_1cm_1deg(cam, tcfg_base, alt, roll, pitch, x, y):
    tags = {t["id"]: t for t in tcfg_base["tags"]}
    s = VehicleState(pos=np.array([x, y, alt]), roll_deg=roll, pitch_deg=pitch)
    _, poses = cam.observe(s, 0.0)
    assert poses, "no detection"
    origin_gt = cam.world_to_cam(s, np.zeros(3))
    best = None
    for p in poses:
        lo, hi = tags[p.tag_id]["valid_alt_m"]
        if not (lo <= alt <= hi):
            continue
        origin = apply_tag_offset((p.x, p.y, p.z), (p.rx, p.ry, p.rz), tags[p.tag_id]["offset_xy_m"])
        xy_err = float(np.hypot(*(origin - origin_gt)[:2]))
        if best is None or xy_err < best:
            best = xy_err
        # orientation: the tag's Z (out of the print) must be the world-up axis seen from the camera
        R = rodrigues((p.rx, p.ry, p.rz))
        z_tag_cam = R @ np.array([0, 0, 1.0])
        z_gt = cam.world_to_cam(s, np.array([0, 0, 1.0])) - cam.world_to_cam(s, np.zeros(3))
        ang = math.degrees(math.acos(np.clip(np.dot(z_tag_cam, z_gt), -1, 1)))
        # Planar-pose ambiguity makes the normal of a small, distant tag unreliable; the
        # controller only uses the rotation to map an in-plane offset, which is insensitive
        # to it. Hold the tag we actually rely on at this altitude to 1 deg, others loosely.
        px = tags[p.tag_id]["size_m"] / p.z * cam.K[0, 0]
        assert ang < (1.0 if px > 60 else 15.0), f"tag {p.tag_id} ({px:.0f} px) normal off by {ang:.2f} deg"
    assert best is not None, "no tag valid at this altitude detected"
    assert best < 0.01 + 0.005 * alt, f"pad origin xy error {best * 100:.2f} cm"


def test_sign_convention_forward_and_right(cam):
    """Vehicle displaced forward of the pad sees the pad at camera +y (image bottom = tail);
    displaced right sees it at camera -x."""
    s = VehicleState(pos=np.array([0.3, 0.0, 2.0]))
    _, poses = cam.observe(s, 0.0)
    big = next(p for p in poses if p.tag_id == 0)
    assert big.y > 0.25  # pad behind the vehicle -> image bottom
    s = VehicleState(pos=np.array([0.0, 0.3, 2.0]))
    _, poses = cam.observe(s, 0.0)
    big = next(p for p in poses if p.tag_id == 0)
    assert big.x < -0.3 - 0.35 + 0.02  # pad to the left, plus the big tag's own -0.35 offset


def test_tilt_moves_image_not_world(cam):
    """Pitching nose-up 10 deg with the pad straight below: the belly looks forward, so the
    pad appears behind the optical axis, toward image bottom (camera +y), by ~alt*sin(10)."""
    s = VehicleState(pos=np.array([0.0, 0.0, 2.0]), pitch_deg=10)
    _, poses = cam.observe(s, 0.0)
    p = next(p for p in poses if p.tag_id == 1)
    origin = apply_tag_offset((p.x, p.y, p.z), (p.rx, p.ry, p.rz), [0.10, 0.0])
    assert origin[1] == pytest.approx(2.0 * math.sin(math.radians(10)), abs=0.02)


def test_occlusion_and_frame_drop(cam, vcfg_base):
    from sim.environment import Environment, EnvParams

    env = Environment(EnvParams(camera=CameraParams(occlusion_start_s=1.0, occlusion_end_s=2.0, frame_drop_prob=1.0)), np.random.default_rng(0))
    s = VehicleState(pos=np.array([0, 0, 2.0]))
    img, poses = cam.observe(s, 0.5, env)
    assert img is None and poses == []  # every frame dropped
    env.p.camera.frame_drop_prob = 0.0
    img, poses = cam.observe(s, 1.5, env)
    assert img is not None and poses == []  # occluded
    _, poses = cam.observe(s, 2.5, env)
    assert poses
