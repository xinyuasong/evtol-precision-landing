import copy

import pytest
import yaml

from control.config import ConfigError, load_tags, load_vehicle
from tests.conftest import ROOT


def test_shipped_configs_load():
    v = load_vehicle(ROOT / "config" / "vehicle.yaml")
    t = load_tags(ROOT / "config" / "tags.yaml")
    assert v["loop"]["rate_hz"] == 50 and len(t["tags"]) == 3


def _write(tmp_path, cfg, name="v.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_missing_key_fails(vcfg, tmp_path):
    del vcfg["safety"]["min_vbat_v"]
    with pytest.raises(ConfigError, match="min_vbat_v"):
        load_vehicle(_write(tmp_path, vcfg))


def test_wrong_type_fails(vcfg, tmp_path):
    vcfg["guidance"]["x"]["kp"] = "fast"
    with pytest.raises(ConfigError, match="guidance.x.kp"):
        load_vehicle(_write(tmp_path, vcfg))


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda c: c["rc"].__setitem__("map", "AETX"), "permutation"),
        (lambda c: c["rc"].__setitem__("roll_sign", 2), "roll_sign"),
        (lambda c: c["guidance"].__setitem__("authority", 1.5), "authority"),
        (lambda c: c["altitude"]["descent"].__setitem__("hold_err_m", 9.0), "hold_err_m"),
        (lambda c: c["camera"].__setitem__("R_bc", [[1, 0], [0, 1]]), "3x3"),
    ],
)
def test_semantic_checks(vcfg, tmp_path, mutate, msg):
    c = copy.deepcopy(vcfg)
    mutate(c)
    with pytest.raises(ConfigError, match=msg):
        load_vehicle(_write(tmp_path, c))


def test_tags_duplicate_id_fails(tcfg, tmp_path):
    tcfg["tags"].append(dict(tcfg["tags"][0]))
    with pytest.raises(ConfigError, match="duplicate"):
        load_tags(_write(tmp_path, tcfg, "t.yaml"))
