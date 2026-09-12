"""The C++ detector and the Python detector must agree on identical frames. CI validates
the scenario suite with the Python detector; hardware runs the C++ one. Without this,
the Monte Carlo table does not transfer. Skipped when the C++ build is absent (CI builds it)."""

import json
import subprocess

import cv2
import numpy as np
import pytest

from sim.camera_sim import CameraSim
from sim.dynamics import VehicleState
from sim.environment import CameraParams
from tests.conftest import ROOT

TOOL = ROOT / "vision" / "build" / "parity_tool"

pytestmark = pytest.mark.skipif(not TOOL.exists(), reason="C++ vision build not present (make vision)")


@pytest.mark.parametrize("pos,roll,pitch", [((0.3, -0.2, 3.0), 0, 0), ((0.1, 0.1, 1.5), 5, -5), ((0.0, 0.02, 0.25), 3, 2), ((0.0, 0.0, 2.0), 0, 10)])
def test_cpp_and_python_agree(tmp_path, vcfg_base, tcfg_base, pos, roll, pitch):
    cam = CameraSim(ROOT / "calib" / "camera_640x480.yaml", tcfg_base, vcfg_base["camera"]["R_bc"], CameraParams(noise_sigma=3.0), np.random.default_rng(0))
    s = VehicleState(pos=np.array(pos), roll_deg=roll, pitch_deg=pitch)
    img, py = cam.observe(s, 0.0)
    png = tmp_path / "f.png"
    cv2.imwrite(str(png), img)
    out = subprocess.run(
        [str(TOOL), str(png), str(ROOT / "calib" / "camera_640x480.yaml"), str(ROOT / "config" / "tags.yaml")], capture_output=True, text=True, check=True
    )
    cpp = {d["tag_id"]: d for d in json.loads(out.stdout)}
    assert py, "python detector found nothing"
    assert set(cpp) == {p.tag_id for p in py}
    for p in py:
        c = cpp[p.tag_id]
        assert np.allclose([c["x"], c["y"], c["z"]], [p.x, p.y, p.z], atol=0.005), f"tag {p.tag_id} position differs"
        assert bool(c["valid"]) == p.valid
