# Vision-guided multirotor precision landing — simulation-validated control stack

A companion-computer flight stack that lands a multirotor on an AprilTag pad by moving
virtual sticks over MSP, plus a simulator that speaks the same protocol on the other end
of a pseudo-terminal. **The flight code cannot tell the simulator from hardware**, and a
test makes sure it never learns to.

## Status — read this first

**This stack has not flown. Everything below is validated in simulation.** There is no
aircraft, no flight controller, and no camera in this repo; the mock flight controller in
`sim/fc_model.py` is the single hardware boundary. The camera intrinsics in `calib/` are a
synthetic placeholder, and the stick-sign convention (`sim/fc_model.py`,
`tests/test_rc_convention.py`) is a reading of the firmware source that a bench test has
not confirmed. The altitude loop assumes **rangefinder-grade altitude**; a bare barometer
drifts metres in ground effect and would make it unflyable near touchdown.

Scenario results (one seed each, Python detector, direct transport):

| Scenario | Tests | Result |
|---|---|---|
| nominal_calm | baseline | **0.6 cm**, DISARM |
| demo_offset_approach | 2.2 m offset approach from 4 m (the media run) | 9.7 cm, DISARM |
| wind_steady_5ms | integrator authority | **1.2 cm** |
| wind_gust_8ms | disturbance rejection | **FAIL** — descends 3.0 → 0.55 m, then a gust takes the pad out of frame; SEARCH has no position reference, so the vehicle drifts downwind and never re-acquires (findings #16) |
| tag_occluded_2s | tag-loss recovery | 2.4 cm via ABORT → re-acquire |
| tag_lost_at_1m | give-up path | ABORT → SEARCH → HANDBACK → FC failsafe lands |
| tag_lost_in_final | FINAL commit | 0.8 cm, no abort |
| high_latency_150ms | latency margin | 0.3 cm |
| pose_noise_high | filter performance | 0.8 cm |
| rolling_shutter_shear | vibration realism | 0.7 cm |
| alt_sensor_dropout | single-source altitude | safety trips → FC lands, 2.7 cm |
| alt_bias_drift | biased altimeter | 0.4 cm, touchdown detected from stalled descent |
| ground_effect_strong | cushion | 0.6 cm |
| companion_stall | failsafe layers | Pi frozen → FC LAND, 1.6 cm |
| battery_sag | safety trip | trips → FC lands, 6.9 cm |
| sign_inversion | the classic bug | NOT caught in flight (findings #12); HANDBACK → FC lands 55 m away |
| touchdown_detect_fails | FINAL timeout | timeout disarms, 1.7 cm |
| moving_pad_0p5mps | stretch | 1.8 cm |

16 of 17 pass. The two honest negatives: the gust scenario exposes that SEARCH holds attitude
rather than position, so any tag loss in wind is unrecoverable (findings #16), and an inverted camera transform is not detectable from 3 m with this field of
view — the bench test is where that bug is meant to die, and the simulator proved the
guide's claim that a 5 m error gate catches it is wrong.

**Monte Carlo has not been run yet.** `python -m sim.runner --all --monte-carlo N`
exists and is seeded; the table will go here with the failures reported as-is.

### What the simulator tests less than its name suggests
With one altitude source, `alt_sensor_dropout` collapses to "no altitude → stop commanding
→ FC lands". The realistic failure — a rangefinder losing returns over grass while a baro
keeps reporting, and an estimator riding it out on a drifting source — needs a second
source and fusion (`control/altitude.py` has the interface; no second implementation).
`ground_effect_strong` likewise tests the throttle loop against extra thrust, not a
barometer being wrong at touchdown.

## Documentation

- `media/` — rendered camera view, contact sheet and dashboard from a run (`make media`).
- `docs/GUIDE.md` — full setup, run, and test instructions plus a walkthrough of every file.
- `docs/findings.md` — the fifteen falsified assumptions. Start here.

## docs/findings.md
Fifteen design assumptions the simulator falsified, with evidence and what changed. That
file is the point of the project; start there.

## Architecture

```
  sim/camera_sim.py ──render──► real AprilTag detector ──UDP TagPose──►┐
  (world, wind, sensors)                                               │
                                                                       ▼
  sim/fc_model.py  ◄──MSP over pty (or in-process codec)──  control/main.py
  (mock FC: RC → lean/thrust, telemetry, RC-timeout failsafe)  (50 Hz loop, KF, PIDs, FSM, safety)
         │
         ▼
  sim/dynamics.py  (point mass + first-order attitude lag, drag, ground effect)
```

- `control/` — the flight code. `msp.py` codec + transport, `frames.py` camera→body→level,
  `filter.py` constant-velocity KF, `guidance.py` PIDs and stick synthesis, `altitude.py`
  Pi-owned altitude loop behind a source interface, `state_machine.py`
  IDLE→SEARCH→ACQUIRE→TRACK→DESCEND→FINAL→DISARM with ABORT and HANDBACK, `safety.py`
  envelope + watchdog. **No `sim`, `mock`, or `pty` anywhere in it**
  (`tests/test_no_sim_awareness.py`).
- `sim/` — mock hardware and world. Two transports: `pty` (control node as a subprocess on
  a real pseudo-terminal, byte-identical to hardware, real-time) and `direct` (in-process,
  same codec and same FC handler, skips only the wire; used for Monte Carlo).
  `tests/test_transport_parity.py` checks they agree.
- `vision/` — C++ detector with a pluggable frame source (camera / video / MockSource fed
  by the simulator over TCP) and a Python detector behind the same wire format.
  `tests/test_detector_parity.py` asserts they agree on identical frames.
- `config/` — every gain, limit and threshold. `control/config.py` schema-checks at startup.

Safety layering, in order: pilot AUX switch (hardware, out of scope here) → FC RC-timeout
failsafe (tested: `companion_stall`, `battery_sag`, `sign_inversion`) → the node stopping
its own stream on any safety trip → envelope limits.

## Quickstart

```
pip install -r requirements.txt
make test                                     # unit + camera round-trip + fast scenarios, no C++
python -m sim.runner sim/scenarios/nominal_calm.yaml
python -m sim.runner sim/scenarios/nominal_calm.yaml --transport pty   # real serial path, real-time
python -m sim.runner --all                    # ~8 min
python -m sim.viz sim/results/nominal_calm_seed0.csv
make vision && python -m pytest tests/test_detector_parity.py            # needs libopencv-dev, yaml-cpp
```

Hardware path (untested): `python -m control.main --port /dev/serial0` with the C++
`vision_node --source camera` publishing to UDP 5600. `tools/bench_rc.py` sends fixed
sticks props-off to confirm channel map and sign first.

## Layout
See `docs/findings.md`, `sim/scenarios/*.yaml` for every scenario's parameters and pass
criteria, and `tools/tune_horizontal.py` for the linear surrogate used to tune gains.
