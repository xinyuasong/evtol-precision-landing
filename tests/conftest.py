import copy
from pathlib import Path

import pytest

from control.config import load_tags, load_vehicle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def vcfg_base():
    return load_vehicle(ROOT / "config" / "vehicle.yaml")


@pytest.fixture(scope="session")
def tcfg_base():
    return load_tags(ROOT / "config" / "tags.yaml")


@pytest.fixture
def vcfg(vcfg_base):
    return copy.deepcopy(vcfg_base)


@pytest.fixture
def tcfg(tcfg_base):
    return copy.deepcopy(tcfg_base)
