"""Headless scenario runner.

    python -m sim.runner sim/scenarios/nominal_calm.yaml
    python -m sim.runner --all
    python -m sim.runner --all --monte-carlo 20
    python -m sim.runner sim/scenarios/companion_stall.yaml --transport pty

Scenario YAML: overrides on top of config/vehicle.yaml and config/tags.yaml,
plus world/environment parameters and pass criteria. Every number the sim
uses comes from here or from the two config files.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from control.config import _validate_tags, _validate_vehicle, load_tags, load_vehicle
from sim.dynamics import VehicleParams
from sim.environment import AltSensorParams, BatteryParams, CameraParams, EnvParams, WindParams
from sim.fc_model import FCParams
from sim.harness import ROOT, DirectRunner, PtyRunner, RunResult, ScenarioSpec

SCENARIO_DIR = ROOT / "sim" / "scenarios"
RESULTS_DIR = ROOT / "sim" / "results"


def _deep_update(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _dc(cls, d: dict | None):
    d = dict(d or {})
    for k, v in d.items():
        if isinstance(v, list):
            d[k] = tuple(v)
    return cls(**d)


def load_scenario(path: str | Path, seed: int | None = None, config_path=None, tags_path=None) -> ScenarioSpec:
    with open(path) as f:
        y = yaml.safe_load(f)
    vcfg = load_vehicle(config_path or ROOT / "config" / "vehicle.yaml")
    tcfg = load_tags(tags_path or ROOT / "config" / "tags.yaml")
    true_R_bc = copy.deepcopy(vcfg["camera"]["R_bc"])  # the world keeps the real mount; overrides only reach control/
    vcfg = _deep_update(vcfg, y.get("vehicle_config_overrides", {}))
    _validate_vehicle(vcfg)
    if "tags_overrides" in y:
        tcfg = _deep_update(tcfg, y["tags_overrides"])
        _validate_tags(tcfg)
    e = y.get("environment", {})
    env = EnvParams(
        wind=_dc(WindParams, e.get("wind")),
        alt=_dc(AltSensorParams, e.get("alt_sensor")),
        camera=_dc(CameraParams, e.get("camera")),
        battery=_dc(BatteryParams, e.get("battery")),
    )
    init = y.get("initial", {})
    return ScenarioSpec(
        name=y["name"],
        duration_s=float(y.get("duration_s", 60.0)),
        seed=int(seed if seed is not None else y.get("seed", 0)),
        initial_pos=tuple(init.get("pos", [0.5, -0.3, 3.0])),
        initial_vel=tuple(init.get("vel", [0, 0, 0])),
        arm_at_s=float(y.get("arm_at_s", 0.5)),
        physics_hz=float(y.get("physics_hz", 400)),
        vehicle=_dc(VehicleParams, y.get("vehicle")),
        fc=_dc(FCParams, y.get("fc")),
        env=env,
        vehicle_cfg=vcfg,
        true_R_bc=true_R_bc,
        tags_cfg=tcfg,
        companion_stall_at_s=y.get("companion_stall_at_s"),
        pass_criteria=y.get("pass", {}),
    )


def randomize(spec: ScenarioSpec, seed: int) -> ScenarioSpec:
    """Monte Carlo: perturb the initial condition deterministically from the seed."""
    rng = np.random.default_rng(seed)
    s = copy.copy(spec)
    s.seed = seed
    x, y, z = spec.initial_pos
    s.initial_pos = (x + rng.uniform(-0.5, 0.5), y + rng.uniform(-0.5, 0.5), max(1.5, z + rng.uniform(-0.5, 0.5)))
    return s


def evaluate(res: RunResult, crit: dict) -> tuple[bool, list[str]]:
    reasons = []
    if "touchdown_err_max_m" in crit:
        if res.touchdown_err_m is None:
            reasons.append("did not touch down")
        elif res.touchdown_err_m > crit["touchdown_err_max_m"]:
            reasons.append(f"touchdown error {res.touchdown_err_m:.3f} m > {crit['touchdown_err_max_m']}")
    if "terminal_state" in crit and res.terminal_state not in ([crit["terminal_state"]] if isinstance(crit["terminal_state"], str) else crit["terminal_state"]):
        reasons.append(f"terminal state {res.terminal_state}")
    if crit.get("no_safety_trip") and res.safety_trip:
        reasons.append(f"safety tripped: {res.safety_trip}")
    if crit.get("safety_trip_required") and not res.safety_trip:
        reasons.append("safety did not trip")
    if crit.get("safety_trip_contains") and crit["safety_trip_contains"] not in res.safety_trip:
        reasons.append(f"safety reason {res.safety_trip!r} lacks {crit['safety_trip_contains']!r}")
    if crit.get("fc_failsafe_landed") and not res.fc_failsafe_landed:
        reasons.append("FC failsafe did not land the aircraft")
    if crit.get("no_fc_failsafe") and res.fc_failsafe_used:
        reasons.append("FC failsafe engaged")
    if "max_alt_m" in crit and res.max_alt_m > crit["max_alt_m"]:
        reasons.append(f"max altitude {res.max_alt_m:.2f} m > {crit['max_alt_m']}")
    if "max_horiz_err_m" in crit and res.max_horiz_err_m > crit["max_horiz_err_m"]:
        reasons.append(f"max horizontal error {res.max_horiz_err_m:.2f} m > {crit['max_horiz_err_m']}")
    if "touchdown_vz_max_ms" in crit and res.touchdown_vz_ms is not None and abs(res.touchdown_vz_ms) > crit["touchdown_vz_max_ms"]:
        reasons.append(f"touchdown speed {abs(res.touchdown_vz_ms):.2f} m/s > {crit['touchdown_vz_max_ms']}")
    if crit.get("must_land") and res.touchdown_err_m is None:
        reasons.append("did not land")
    if crit.get("must_not_climb_away") and res.touchdown_err_m is None:
        reasons.append("did not land")
    if crit.get("transition_reason_contains") and not any(crit["transition_reason_contains"] in r for *_, r in res.transitions):
        reasons.append(f"no transition mentioning {crit['transition_reason_contains']!r}")
    if crit.get("visited_state") and not any(b == crit["visited_state"] for _, _, b, _ in res.transitions):
        reasons.append(f"never entered {crit['visited_state']}")
    if crit.get("never_visited_state") and any(b == crit["never_visited_state"] for _, _, b, _ in res.transitions):
        reasons.append(f"entered {crit['never_visited_state']}")
    if crit.get("landed_or_aborted") and res.touchdown_err_m is None and not res.safety_trip and res.terminal_state not in ("ABORT", "SEARCH"):
        reasons.append("neither landed nor aborted")
    return (not reasons), reasons


def _run_spec_summary(spec: ScenarioSpec) -> dict:
    res = DirectRunner(spec).run()
    ok, why = evaluate(res, spec.pass_criteria)
    row = summarize(res)
    row.update(passed=int(ok), reasons="; ".join(why))
    return row


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run_one(spec: ScenarioSpec, transport: str = "direct", log_path=None) -> RunResult:
    if transport == "pty":
        return PtyRunner(spec, str(ROOT / "config" / "vehicle.yaml"), str(ROOT / "config" / "tags.yaml"), log_path).run()
    return DirectRunner(spec).run()


def summarize(res: RunResult) -> dict:
    d = dataclasses.asdict(res)
    d.pop("records")
    d["transitions"] = json.dumps(res.transitions)
    return d


def write_records(res: RunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t",
                "state",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "roll",
                "pitch",
                "alt_est",
                "horiz_err",
                "rc_roll",
                "rc_pitch",
                "rc_thr",
                "rc_yaw",
                "tx",
                "safety",
                "fc_failsafe",
                "kf_x",
                "kf_y",
                "pid",
            ]
        )
        for r in res.records:
            rc = r.rc or (None, None, None, None)
            w.writerow(
                [
                    f"{r.t:.4f}",
                    r.state_name,
                    *[f"{v:.4f}" for v in r.pos],
                    *[f"{v:.4f}" for v in r.vel],
                    f"{r.roll:.3f}",
                    f"{r.pitch:.3f}",
                    r.alt_est,
                    r.horiz_err,
                    *rc,
                    int(r.tx),
                    r.safety,
                    int(r.fc_failsafe),
                    *r.kf,
                    json.dumps(r.pid),
                ]
            )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--monte-carlo", type=int, default=0, metavar="N")
    ap.add_argument("--transport", choices=["direct", "pty"], default="direct")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    paths = sorted(SCENARIO_DIR.glob("*.yaml")) if a.all else [Path(p) for p in a.scenarios]
    if not paths:
        ap.error("no scenarios given")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, table = [], []
    all_ok = True
    for p in paths:
        base = load_scenario(p, a.seed)
        if a.monte_carlo:
            seeds = list(range(1, a.monte_carlo + 1))
            specs = [randomize(base, s) for s in seeds]
        else:
            specs = [base]
        if a.monte_carlo and a.jobs > 1 and a.transport == "direct":
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                srows = list(ex.map(_run_spec_summary, specs))
        else:
            srows = []
            for spec in specs:
                res = run_one(spec, a.transport)
                ok, why = evaluate(res, spec.pass_criteria)
                row = summarize(res)
                row.update(passed=int(ok), reasons="; ".join(why))
                srows.append(row)
                if not a.monte_carlo:
                    write_records(res, out_dir / f"{spec.name}_seed{spec.seed}.csv")
                if not a.quiet and not a.monte_carlo:
                    e = f"{res.touchdown_err_m * 100:.1f} cm" if res.touchdown_err_m is not None else "n/a"
                    pk = f"{res.peak_err_descend_m * 100:.0f} cm" if res.peak_err_descend_m is not None else "n/a"
                    tt = f"{res.t_track_to_touchdown_s:.1f} s" if res.t_track_to_touchdown_s is not None else "n/a"
                    print(
                        f"[{'PASS' if ok else 'FAIL'}] {spec.name}: touchdown {e}, peak DESCEND err {pk}, track->touchdown {tt}, terminal {res.terminal_state}, safety {res.safety_trip or '-'}, fc_failsafe {res.fc_failsafe_used}, detections {res.detection_rate:.0%}"
                    )
                    for t, s0, s1, r in res.transitions:
                        print(f"        {t:7.2f}  {s0:>8} -> {s1:<8} {r}")
                    for r in why:
                        print(f"        ! {r}")
        rows.extend(srows)
        n_ok = sum(r["passed"] for r in srows)
        errs = [r["touchdown_err_m"] for r in srows if r["touchdown_err_m"] is not None]
        peaks = [r["peak_err_descend_m"] for r in srows if r["peak_err_descend_m"] is not None]
        tts = [r["t_track_to_touchdown_s"] for r in srows if r["t_track_to_touchdown_s"] is not None]
        table.append(
            (
                base.name,
                n_ok,
                len(srows),
                statistics.median(errs) if errs else None,
                _pct(errs, 0.95),
                statistics.median(peaks) if peaks else None,
                statistics.median(tts) if tts else None,
            )
        )
        all_ok &= n_ok == len(srows)
        # incremental: one line per scenario so a long Monte Carlo can be watched
        print(f"[done] {base.name}: {n_ok}/{len(srows)} pass, touchdown median {statistics.median(errs) * 100 if errs else float('nan'):.1f} cm", flush=True)
        with open(out_dir / ("monte_carlo.csv" if a.monte_carlo else "scenarios.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(out_dir / ("monte_carlo.csv" if a.monte_carlo else "scenarios.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print()

    def fmt(v, unit="cm", k=100):
        return f"{v * k:.1f} {unit}" if v is not None else "—"

    print("| Scenario | Pass | Touchdown median | Touchdown p95 | Peak DESCEND err (median) | TRACK→touchdown (median) |")
    print("|---|---|---|---|---|---|")
    for name, ok, n, med, p95, pk, tt in table:
        print(f"| {name} | {ok}/{n} ({ok / n:.0%}) | {fmt(med)} | {fmt(p95)} | {fmt(pk)} | {fmt(tt, 's', 1)} |")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
