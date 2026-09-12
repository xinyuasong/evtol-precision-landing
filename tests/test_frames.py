import math

import numpy as np
import pytest

from control.frames import FrameTransformer, apply_tag_offset, rodrigues

R_BC = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]


def test_camera_to_body_axes():
    ft = FrameTransformer(R_BC)
    # tag right in the image -> right of the vehicle
    assert np.allclose(ft.camera_to_body([1, 0, 0]), [0, 1, 0])
    # tag toward image bottom -> behind the vehicle (image-top is the nose)
    assert np.allclose(ft.camera_to_body([0, 1, 0]), [-1, 0, 0])
    # optical axis -> straight down
    assert np.allclose(ft.camera_to_body([0, 0, 1]), [0, 0, 1])


def test_rejects_non_rotation():
    with pytest.raises(ValueError):
        FrameTransformer([[1, 0, 0], [0, 2, 0], [0, 0, 1]])


def test_level_frame_is_identity_when_level():
    ft = FrameTransformer(R_BC)
    p = ft.camera_to_level([0.3, -0.2, 2.0], 0.0, 0.0)
    assert np.allclose(p, [0.2, 0.3, 2.0])


def test_pitch_tilt_removes_phantom_error():
    """Tag on the optical axis at 3 m with the nose 10 deg up: the belly looks
    forward, so the tag is really 3*sin(10) = 0.521 m ahead."""
    ft = FrameTransformer(R_BC)
    p = ft.camera_to_level([0, 0, 3.0], roll_deg=0.0, pitch_deg=10.0)
    assert p[0] == pytest.approx(3 * math.sin(math.radians(10)), abs=1e-9)
    assert p[1] == pytest.approx(0.0, abs=1e-9)
    assert p[2] == pytest.approx(3 * math.cos(math.radians(10)), abs=1e-9)


def test_roll_tilt_sign():
    """Roll right (right wing down): the belly swings left, so a tag on the
    optical axis is really to the vehicle's left (negative Y)."""
    ft = FrameTransformer(R_BC)
    p = ft.camera_to_level([0, 0, 2.0], roll_deg=15.0, pitch_deg=0.0)
    assert p[1] == pytest.approx(-2 * math.sin(math.radians(15)), abs=1e-9)
    assert p[0] == pytest.approx(0.0, abs=1e-9)


def test_combined_tilt_hand_computed():
    ft = FrameTransformer(R_BC)
    roll, pitch = math.radians(5), math.radians(-8)
    p_body = np.array([0.4, -0.1, 2.5])
    Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    expected = Ry @ Rx @ p_body
    got = ft.body_to_level(p_body, 5.0, -8.0)
    assert np.allclose(got, expected)
    # a rotation preserves range
    assert np.linalg.norm(got) == pytest.approx(np.linalg.norm(p_body))


def test_rodrigues_matches_definition():
    R = rodrigues([0, 0, math.pi / 2])
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])
    R = rodrigues([math.pi, 0, 0])
    assert np.allclose(R, np.diag([1, -1, -1]))
    assert np.allclose(rodrigues([0, 0, 0]), np.eye(3))


def test_tag_offset_resolves_to_pad_origin():
    # A tag seen face-on by a downward camera has tag->camera rotation diag(1,-1,-1):
    # printed-right = camera +x, printed-up = camera -y.
    face_on = [math.pi, 0, 0]
    # tag sits 0.2 m to the pad's printed-right; seen at camera (0.2, 0, 1) -> pad origin (0, 0, 1)
    assert np.allclose(apply_tag_offset([0.2, 0.0, 1.0], face_on, [0.2, 0.0]), [0, 0, 1])
    # tag 0.3 m printed-up of the origin appears at camera -y
    assert np.allclose(apply_tag_offset([0.0, -0.3, 1.0], face_on, [0.0, 0.3]), [0, 0, 1])
    # with an extra 90 deg yaw about the camera axis the offset rotates too
    R = rodrigues([0, 0, math.pi / 2]) @ rodrigues(face_on)
    rv = cv2_free_rvec(R)
    assert np.allclose(apply_tag_offset([0.0, 0.2, 1.0], rv, [0.2, 0.0]), [0, 0, 1], atol=1e-9)


def cv2_free_rvec(R):
    """Matrix -> rotation vector, for the test only."""
    th = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if th < 1e-9:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-9:  # axis from the symmetric part
        k = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        k[1] *= np.sign(R[0, 1]) if R[0, 1] != 0 else 1
        k[2] *= np.sign(R[0, 2]) if R[0, 2] != 0 else 1
        return k * th
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return ax * th
