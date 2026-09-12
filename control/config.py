"""Single source of truth for configuration. Fails at startup, not at altitude."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    pass


# Schema: nested dict of key -> type (or nested dict). Every leaf must be present
# with a value of the given type. Ints are accepted where floats are required.
_PID = {"kp": float, "ki": float, "kd": float, "i_limit": float, "out_limit": float, "d_tau_s": float}

VEHICLE_SCHEMA: dict[str, Any] = {
    "serial": {"port": str, "baud": int, "timeout_s": float},
    "loop": {"rate_hz": float, "pose_max_age_s": float},
    "telemetry": {"rate_hz": float, "armed_flag_bit": int, "override_flag_bit": int},
    "pose": {"udp_host": str, "udp_port": int, "max_reproj_err_px": float},
    "camera": {"R_bc": list},
    "filter": {"q": float, "r0": float, "z_ref_m": float, "coast_max_s": float},
    "guidance": {
        "stick_scale": float,
        "authority": float,
        "x": _PID,
        "y": _PID,
        "yaw": {"enabled": bool, "pid": _PID},
    },
    "altitude": {
        "pid": _PID,
        "hover_throttle_us": int,
        "throttle_authority": float,
        "lowpass_tau_s": float,
        "descent": {
            "climb_err_m": float,
            "climb_rate_ms": float,
            "hold_err_m": float,
            "rate_min_ms": float,
            "rate_per_m": float,
            "rate_max_ms": float,
        },
        "final": {"descent_rate_ms": float, "setpoint_floor_m": float},
        "abort": {"climb_to_m": float, "climb_rate_ms": float},
        "ground_effect": {"height_m": float, "ff_reduction": float},
        "touchdown": {"alt_m": float, "vario_ms": float, "max_throttle_us": int, "sustain_s": float, "stall_sustain_s": float},
    },
    "rc": {
        "mid_us": int,
        "range_us": int,
        "min_us": int,
        "max_us": int,
        "map": str,
        "roll_sign": int,
        "pitch_sign": int,
        "yaw_sign": int,
        "aux_channels": list,
    },
    "state_machine": {
        "detection_gap_s": float,
        "acquire_consecutive_frames": int,
        "acquire_settle_s": float,
        "acquire_sigma_max_m": float,
        "track_timeout_s": float,
        "track_lost_s": float,
        "track_ok_err_m": float,
        "track_stable_s": float,
        "descend_lost_s": float,
        "final_alt_m": float,
        "final_err_m": float,
        "final_timeout_s": float,
        "search_timeout_s": float,
        "abort_min_hold_s": float,
    },
    "safety": {
        "startup_grace_s": float,
        "loop_overrun_s": float,
        "attitude_timeout_s": float,
        "altitude_timeout_s": float,
        "max_alt_m": float,
        "max_horiz_err_m": float,
        "divergence_m": float,
        "divergence_window_s": float,
        "divergence_monotonic_frac": float,
        "min_vbat_v": float,
        "max_tilt_deg": float,
    },
}

TAGS_SCHEMA: dict[str, Any] = {"family": str, "tags": list}


def _check(node: Any, schema: Any, path: str) -> None:
    if isinstance(schema, dict):
        if not isinstance(node, dict):
            raise ConfigError(f"{path}: expected a mapping")
        for key, sub in schema.items():
            if key not in node:
                raise ConfigError(f"{path}.{key}: missing")
            _check(node[key], sub, f"{path}.{key}")
        return
    if schema is float:
        if isinstance(node, bool) or not isinstance(node, (int, float)):
            raise ConfigError(f"{path}: expected number, got {node!r}")
    elif schema is int:
        if isinstance(node, bool) or not isinstance(node, int):
            raise ConfigError(f"{path}: expected int, got {node!r}")
    elif schema is bool:
        if not isinstance(node, bool):
            raise ConfigError(f"{path}: expected bool, got {node!r}")
    elif not isinstance(node, schema):
        raise ConfigError(f"{path}: expected {schema.__name__}, got {node!r}")


def _validate_vehicle(cfg: dict) -> None:
    _check(cfg, VEHICLE_SCHEMA, "vehicle")
    r = cfg["camera"]["R_bc"]
    if len(r) != 3 or any(len(row) != 3 for row in r):
        raise ConfigError("vehicle.camera.R_bc: must be 3x3")
    rc = cfg["rc"]
    if sorted(rc["map"]) != sorted("AETR"):
        raise ConfigError("vehicle.rc.map: must be a permutation of AETR")
    for k in ("roll_sign", "pitch_sign", "yaw_sign"):
        if rc[k] not in (-1, 1):
            raise ConfigError(f"vehicle.rc.{k}: must be +1 or -1")
    if not (rc["min_us"] < rc["mid_us"] < rc["max_us"]):
        raise ConfigError("vehicle.rc: min < mid < max required")
    if not (0.0 < cfg["guidance"]["authority"] <= 1.0):
        raise ConfigError("vehicle.guidance.authority: must be in (0, 1]")
    if not (0.0 < cfg["guidance"]["stick_scale"] <= cfg["guidance"]["authority"]):
        raise ConfigError("vehicle.guidance.stick_scale: must be in (0, authority]")
    if not (0.0 < cfg["altitude"]["throttle_authority"] <= 1.0):
        raise ConfigError("vehicle.altitude.throttle_authority: must be in (0, 1]")
    d = cfg["altitude"]["descent"]
    if not (d["hold_err_m"] < d["climb_err_m"]):
        raise ConfigError("vehicle.altitude.descent: hold_err_m must be < climb_err_m")
    sm = cfg["state_machine"]
    if not (sm["final_err_m"] < sm["track_ok_err_m"]):
        raise ConfigError("vehicle.state_machine: final_err_m must be < track_ok_err_m")


def _validate_tags(cfg: dict) -> None:
    _check(cfg, TAGS_SCHEMA, "tags")
    ids = set()
    for i, t in enumerate(cfg["tags"]):
        p = f"tags.tags[{i}]"
        for k in ("id", "size_m", "offset_xy_m", "valid_alt_m"):
            if k not in t:
                raise ConfigError(f"{p}.{k}: missing")
        if t["id"] in ids:
            raise ConfigError(f"{p}: duplicate id {t['id']}")
        ids.add(t["id"])
        if t["size_m"] <= 0:
            raise ConfigError(f"{p}.size_m: must be positive")
        if len(t["offset_xy_m"]) != 2 or len(t["valid_alt_m"]) != 2:
            raise ConfigError(f"{p}: offset_xy_m and valid_alt_m must have 2 entries")
        if not t["valid_alt_m"][0] < t["valid_alt_m"][1]:
            raise ConfigError(f"{p}.valid_alt_m: [lo, hi] with lo < hi")


def load_vehicle(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _validate_vehicle(cfg)
    return cfg


def load_tags(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _validate_tags(cfg)
    return cfg
