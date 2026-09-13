# Full guide: what this project is, how to run it, how to test it

Written for someone opening this repo cold. Read Part 1 to understand it, Part 2 to run it,
Part 3 to know what every file does.

---

# PART 1 — What this project is

## The one-sentence version

A drone flight-control program that lands a multirotor on a printed marker using a camera,
plus a simulator that pretends to be the drone so the program can be tested and proven
without an aircraft.

## The problem it solves

GPS gets you within 1–3 metres. Landing on a charging pad or a boat deck needs 5–10 cm.
The gap is closed with vision: a downward camera sees a printed AprilTag (a black-and-white
fiducial marker, like a chunky QR code designed for robots), works out where the pad is
relative to the aircraft, and steers toward it.

## Why the pad has three tags of different sizes

This is the thing that looks strange in the rendered frames, so it's worth explaining first.

A tag has to be big enough to decode from altitude and small enough to stay inside the frame
at touchdown. Those requirements fight each other. With this camera (500 px focal length,
640x480), a 40 cm tag is 50 px wide at 4 m — decodes fine — but at 30 cm altitude it would be
667 px wide, far outside the frame. A 2.5 cm tag is readable at 20 cm and invisible at 3 m.

So the pad carries three: **40 cm, 10 cm, and 2.5 cm**, each with an altitude band it's
valid in (`config/tags.yaml`). The controller uses whichever is usable right now, and hands
off between them as it descends. In the contact sheet you can watch it happen: big tag only
at 4 m, big + medium at 2 m, medium + small at 0.5 m, small only at 0.1 m, nothing at
touchdown.

**They are side by side, not nested.** The original design called for concentric tags —
small ones printed inside the big one's white centre. That does not work: the AprilTag
decoder samples the middle of every cell, so an inner tag lands on the outer tag's samples
and corrupts its code. The outer tag then fails to decode at *every* altitude. That's
finding #1 in `docs/findings.md`, and it's why the pad looks like three tags in a row.

**Why it looks like the drone lands on the tiny tag.** Each tag's position on the pad is
recorded as an offset from the pad origin — the actual touchdown point. Tag 0 sits 35 cm to
one side, tag 1 sits 10 cm to the other side, and tag 2 sits *on* the origin. Whichever tag
is detected, the controller subtracts that tag's offset and gets the same pad origin, so it
sees one continuous position signal with no jump at handoff. The amber crosshair in the
rendered frames is that resolved origin. The drone isn't aiming at the small tag; the small
tag just happens to be printed where the drone is aiming.

## Why there are two computers

This is the architectural spine, and it's worth understanding before anything else.

A flight controller (FC) — a small STM32 board running firmware like INAV or Betaflight —
keeps the aircraft upright. It runs a gyro loop at 1–8 kHz. If that loop stalls for 20 ms,
the aircraft flips and falls. It is hard real-time.

Computer vision needs OpenCV, NumPy, a filesystem, and hundreds of MB of RAM. That means
Linux (a Raspberry Pi). Linux is **not** hard real-time — the scheduler can pause your
process for tens of milliseconds whenever it feels like it.

So you don't put vision on the flight controller and you don't put stabilisation on Linux.
You split them:

- **Flight controller**: keeps the aircraft level. Never waits for the Pi.
- **Raspberry Pi ("companion computer")**: looks at the world, decides where to go, and
  tells the FC by **moving virtual sticks** — exactly like a human pilot's radio does.

The payoff: if the Pi crashes, the FC just sees the RC signal stop and runs its normal
radio-failure failsafe (land or return home). The safety property falls out of the
architecture instead of being bolted on. **This repo is the Pi's software.**

## What the Pi sends: MSP

MSP (MultiWii Serial Protocol) is a binary request/response protocol over a serial wire
(UART). The Pi sends `MSP_SET_RAW_RC` at 50 Hz — literally "roll stick = 1520 µs, pitch
stick = 1480 µs, throttle = 1500 µs" — and polls `MSP_ATTITUDE` and `MSP_ALTITUDE` back
at 20 Hz. That's the whole interface. `control/msp.py` implements it from scratch.

## The trick that makes this repo interesting

There is no drone here. No flight controller, no camera, no aircraft. **Everything below
the flight-controller boundary is simulated.** But the simulation isn't a special mode
inside the flight code — it's on the other end of a real wire.

The simulator opens a **pseudo-terminal** (`pty.openpty()`), which is a Linux facility that
looks exactly like a serial port — same file, same read/write, same everything. The mock
flight controller sits on one end speaking real MSP bytes. The control node opens the other
end with `--port /dev/pts/7` exactly as it would open `/dev/serial0` on a Pi.

**The flight code cannot tell the difference, and there is a test that makes sure it never
learns to** (`tests/test_no_sim_awareness.py` greps `control/` for the words "sim", "mock",
"pty", and fails the build on a hit). That constraint is the strongest claim the repo
makes: the code validated in simulation is bit-identical to the code that would fly.

## The other thing that matters: the camera is simulated honestly

A lazy simulator computes the true position of the pad and adds a bit of random noise. That
tests nothing — it validates your controller against a world your own code invented.

This one **renders an actual image**: it projects the printed tags through the camera
intrinsics, warps the bitmaps into the frame, adds motion blur from the vehicle's own
velocity, adds sensor noise, optionally shears rows to fake a rolling shutter — then runs
the **real AprilTag detector** on that image, the same one that would run on hardware. If
the detector fails to decode a blurry tag, the control loop gets nothing, exactly as it
would in flight.

`tests/test_camera_sim.py` checks this pipeline against ground truth across altitude and
tilt: render at a known pose, detect, recover the pad origin, assert it matches within
~1 cm. Without that test every scenario could pass while proving nothing.

## What the flight code actually does, step by step

Every 20 ms (50 Hz), `control/main.py` runs one tick:

1. **Read the newest tag pose** from the vision node (UDP). Newest only — never a queue. A
   200 ms-old pose is worse than no pose.
2. **Check it's usable**: reprojection error below threshold, tag ID known, and the tag is
   one that's valid at the current altitude (the 40 cm tag is useless at 20 cm).
3. **Resolve to the pad origin.** The pad has three tags of different sizes side by side.
   Each one's offset from the pad centre is in `config/tags.yaml`, so whichever tag is
   detected, the controller sees one continuous position signal — no jump when the big tag
   leaves the frame and the small one takes over.
4. **Transform frames.** The measurement is in the camera's frame. Rotate it into the
   body frame (fixed rotation, how the camera is bolted on), then de-rotate by the
   vehicle's roll and pitch into a gravity-levelled frame. *This step is not optional:* a
   10° tilt at 3 m altitude injects 52 cm of phantom position error, the controller tilts
   further to chase it, and the loop diverges.
5. **Filter.** A constant-velocity Kalman filter per horizontal axis. It smooths noise, gives
   a velocity estimate the D-term needs, and coasts through dropped frames.
6. **Run the state machine.** IDLE → SEARCH → ACQUIRE → TRACK → DESCEND → FINAL → DISARM,
   with ABORT (recoverable: climb, hold, re-search) and HANDBACK (give up: stop
   transmitting, let the FC land it).
7. **Check safety.** Loop overrun, stale telemetry, altitude ceiling, implausible error,
   battery, tilt, divergence. Any trip **stops the RC stream**, which is the correct failure
   action — it delegates to firmware already designed to handle radio loss.
8. **Compute sticks.** Horizontal: PID on position error. Vertical: this Pi owns throttle,
   so it runs its own altitude loop with a hover feedforward, ground-effect compensation,
   and a descent rate gated on horizontal accuracy (drift off-centre → descent pauses;
   drift far → climb).
9. **Send** `MSP_SET_RAW_RC`.

The FINAL state is worth a note: below 20 cm the smallest tag is leaving the field of view,
prop wash kicks up dust, and ground effect destabilises. Continuing to steer on vision makes
it worse. So FINAL **commits**: ignores vision, holds the last horizontal command (which
carries the integrator's wind estimate), descends straight down, and disarms on touchdown —
or on a timeout, if touchdown is never detected.

## The findings file — read this one

`docs/findings.md` is 15 entries, each one a design assumption the simulator disproved, with
the evidence and what changed. Examples:

- Concentric nested AprilTags **don't work** — the decoder samples the centre of each cell,
  so an inner tag corrupts the outer tag's code. Verified in sim.
- A Kalman `r0 = 0.01` is a *variance* — 10 cm sigma. No gain set was stable until it was
  corrected to 0.0004.
- A ground-effect hover at 6 cm looks exactly like a touchdown to an altitude+vario detector.
- `max_horiz_err_m = 5 m` cannot catch an inverted camera transform: the tag leaves the field
  of view at 1.4 m, so the 5 m gate never fires.
- The 50 Hz RC stream was blocking behind the 20 Hz telemetry poll's serial lock — a bug
  only the real pty transport could expose.

A repo that lands a drone is ordinary. A repo that documents fifteen assumptions its own
simulator disproved is the thing worth talking about.

---

# PART 2 — Running it

## What you need

- **Linux or macOS.** Windows needs WSL2 — `pty.openpty()` doesn't exist natively on Windows,
  so the real-transport tests won't run otherwise.
- **Python 3.10 or newer** (developed on 3.12).
- About 500 MB of disk for the OpenCV wheel.
- No GPU, no special hardware, no drone.

## Step 1 — Open a terminal and set up

```bash
cd ~/                                    # or wherever you keep projects
unzip evtol-precision-landing.zip        # if you're starting from the zip
cd evtol

python3 -m venv .venv
source .venv/bin/activate                # you should see (.venv) in your prompt
pip install -r requirements.txt
```

That last step takes 1–3 minutes (OpenCV is large). If `python3 -m venv` fails on Ubuntu,
run `sudo apt install python3-venv` first.

**Every command below assumes you are in the `evtol` directory with the venv activated.**
If you open a new terminal, run `cd ~/evtol && source .venv/bin/activate` again.

## Step 2 — Run the tests (do this first)

```bash
python -m pytest -m "not slow"
```

Expected: **134 passed, 4 skipped, 1 deselected** in about 2 minutes.

The 4 skips are the C++ detector parity tests — they skip because the C++ binary isn't
built. The 1 deselected is the real-time transport test. Both are covered below.

If this passes, the whole stack is working on your machine.

To see what's actually being tested, add `-v`:

```bash
python -m pytest -m "not slow" -v
```

You'll see names like `test_pitch_tilt_removes_phantom_error`,
`test_touchdown_stalled_descent_despite_biased_altimeter`,
`test_control_has_no_simulation_awareness`.

To run one file:

```bash
python -m pytest tests/test_camera_sim.py -v      # the render→detect round trip
python -m pytest tests/test_msp.py -v             # the protocol codec
python -m pytest tests/test_state_machine.py -v   # every FSM transition
```

## Step 3 — Fly one landing

```bash
python -m sim.runner sim/scenarios/nominal_calm.yaml
```

Takes about 15 seconds. Output:

```
[PASS] nominal_calm: touchdown 1.7 cm, peak DESCEND err 12 cm, track->touchdown 14.2 s,
       terminal DISARM, safety -, fc_failsafe False, detections 89%
           0.58      IDLE -> SEARCH   armed + override engaged
           0.60    SEARCH -> ACQUIRE  18 consecutive detections
           1.62   ACQUIRE -> TRACK    pose stable sigma=0.041 m
           4.22     TRACK -> DESCEND  centred < 0.2 m for 2.0 s
          13.68   DESCEND -> FINAL    committed: low and centred
          16.84     FINAL -> DISARM   touchdown detected
```

**How to read it.** The vehicle starts 3 m up and about 0.6 m off the pad. It acquires the
tag, settles, spends ~9 s descending while holding centre, commits at 20 cm, and touches
down 1.7 cm from the pad centre. Every state transition prints its reason.

A per-tick CSV lands in `sim/results/nominal_calm_seed0.csv` — 40-odd columns including
position, velocity, attitude, Kalman state, every PID term separately, all four RC channels,
and the FSM state.

## Step 4 — Run the same landing over a real serial port

```bash
python -m sim.runner sim/scenarios/nominal_calm.yaml --transport pty
```

This one runs in **real time** (~20 s) because it actually is real time. What happens:

1. The simulator opens a pseudo-terminal.
2. It launches `control/main.py` as a **separate OS process** with `--port /dev/pts/N`.
3. Real MSP bytes flow over that wire in both directions at 50 Hz and 20 Hz.
4. Pose data goes over a real UDP socket.

The control node has no idea it isn't on a Pi. This is the transport that found the
serial-lock bug. Use `--transport direct` (the default) for speed; use `pty` when you want
the byte-exact path.

## Step 5 — Run the whole scenario suite

```bash
python -m sim.runner --all
```

**Takes about 8 minutes.** 17 scenarios: calm, steady wind, gusts, tag occlusion, tag loss
at two different altitudes, high latency, heavy image noise, altitude sensor dropout,
altitude bias drift, strong ground effect, companion computer freezing mid-flight, battery
sag, an inverted camera transform, rolling-shutter shear, a moving pad, and a deliberately
broken touchdown detector.

Every scenario is a YAML file in `sim/scenarios/` with its parameters and its pass criteria.
Open one:

```bash
cat sim/scenarios/wind_gust_8ms.yaml
```

## Step 6 — Monte Carlo (the real numbers)

A single seed is an anecdote. This runs every scenario across N randomized initial
conditions, seeded so any result reproduces exactly:

```bash
python -m sim.runner --all --monte-carlo 30 --out sim/results/mc
```

**This takes about 3–4 hours on one core.** Run it in the background and come back:

```bash
nohup python -m sim.runner --all --monte-carlo 30 --out sim/results/mc > mc.log 2>&1 &
tail -f mc.log        # watch it; Ctrl-C stops watching, not the run
```

It prints one line per scenario as it finishes and rewrites `sim/results/mc/monte_carlo.csv`
after each, so you can look at partial results any time. The final table has, per scenario:
pass rate, touchdown median, touchdown p95, median peak horizontal error during DESCEND, and
median time from TRACK to touchdown.

**Why the last two columns exist:** touchdown error alone is misleading. The DESCEND→FINAL
gate already requires error under 8 cm before committing, so touchdown error mostly measures
the final 20 cm of open-loop descent — not tracking quality. Peak DESCEND error and
convergence time can't be gamed by the gate, so they're what actually show whether latency or
noise is hurting the loop.

To run fewer seeds while you're experimenting:

```bash
python -m sim.runner sim/scenarios/nominal_calm.yaml --monte-carlo 10
```

## Step 7 — Look at a flight

```bash
python -m sim.viz sim/results/nominal_calm_seed0.csv
```

Six panels: trajectory from above, horizontal error over time, altitude (true vs estimated),
the FSM state timeline, the PID terms broken out separately, and the RC channels. Needs a
display; on a headless box, copy the CSV somewhere with a screen.

## Step 8 — Make pictures of a run

The simulator renders real camera frames, so you can save them. This produces an animated
GIF of the camera view, a six-frame contact sheet, and a six-panel dashboard:

```bash
python tools/record_run.py sim/scenarios/demo_offset_approach.yaml --out media
```

Takes about 90 seconds and writes three files into `media/`:

| File | What it is |
|---|---|
| `<scenario>_camera.gif` | The camera view through the whole flight, with overlays and a status bar |
| `<scenario>_frames.png` | Six stills spread across the run, side by side |
| `<scenario>_dashboard.png` | Trajectory, error, altitude, state timeline, PID terms, RC channels |

Both demo runs at once:

```bash
make media
```

### How to read the overlay

- **Green box** — the tag square, reprojected from the pose `solvePnP` recovered. This is the
  detector's own answer drawn back onto the image, not ground truth. If the pose solution were
  wrong, the box would not sit on the tag. It's a visual assertion.
- **Green label** — tag ID and reprojection error in pixels. Under ~0.5 px is a clean fit;
  the controller rejects anything over 2 px.
- **Amber crosshair** — the pad origin after that tag's offset has been applied. All three
  tags should put the crosshair in the same place; that's the handoff working.
- **Status bar** — simulation time, mission state, true altitude, horizontal error, and how
  many tags were detected in that frame.

### Which scenario to record

`demo_offset_approach.yaml` is the one built for showing the system off: the vehicle starts
2.2 m from the pad and 4 m up, so the pad sits in the **top-left corner** of the frame. The
pilot hovers manually for the first 4 seconds — state `IDLE`, autonomy computing nothing —
then flips the MSP-override switch. The stack acquires, flies across to the pad, centres,
descends, hands off between all three tags, and lands.

Any scenario works:

```bash
python tools/record_run.py sim/scenarios/wind_gust_8ms.yaml --out media     # the failure
python tools/record_run.py sim/scenarios/tag_occluded_2s.yaml --out media   # loses the tag, recovers
python tools/record_run.py sim/scenarios/pose_noise_high.yaml --out media   # noisy, dim, blurry
```

Useful flags: `--gif-stride N` keeps every Nth frame (higher = smaller file), `--gif-scale`
sets the resolution (1.0 keeps full 640x480), `--seed N` picks a different random draw.

### Why the image shakes

That's real, and it comes out of the physics rather than being added for effect. The vehicle
tilts to accelerate, so the camera tilts with it; `camera_sim.py` applies motion blur whose
length and direction come from the vehicle's own velocity and angular rate over the exposure
time. Wind and gusts move the airframe, and the frame moves with it. In the
`rolling_shutter_shear` scenario the rows are additionally sheared at the vibration
frequency, which is what a rolling-shutter sensor does on a vibrating airframe — and which
makes `solvePnP` return confident, wrong answers.

## Step 9 — Build the C++ detector (optional)

The vision node has two implementations: Python (`vision/py/detector_py.py`, used by
everything above) and C++ (`vision/src/`, what would run on the Pi). They must agree, or
results validated with one don't transfer to the other.

```bash
sudo apt install cmake libopencv-dev libyaml-cpp-dev     # Ubuntu/Debian
make vision
python -m pytest tests/test_detector_parity.py -v
```

That test feeds identical rendered frames to both detectors and asserts the recovered poses
match within 5 mm. **Caveat: this has never been compiled.** It is written but unbuilt —
expect compiler complaints on the first attempt. Fixing them is a genuine next task.

## Step 10 — The real-time transport test

```bash
python -m pytest -m slow -v
```

Runs the same world through both transports and asserts they land the same way. ~40 seconds
of wall clock because the pty run is real-time.

## Everything at once

```bash
make ci          # lint + all non-slow tests
make test        # tests only
make scenarios   # the 17-scenario suite
make vision      # the C++ build
```

## Changing things

**Gains and limits** are all in `config/vehicle.yaml`. Nothing in `control/` has a magic
number; `control/config.py` validates the file at startup and refuses to run on a missing or
nonsensical key, so a typo fails immediately instead of at 3 m altitude.

Try making it worse, then rerun `nominal_calm`:

```bash
# edit config/vehicle.yaml, under guidance:
#   x: {kp: 3.00, ...}     <- triple the proportional gain
python -m sim.runner sim/scenarios/nominal_calm.yaml
```

You should see oscillation and a worse touchdown, or a divergence trip.

**Gain tuning without the renderer** (seconds instead of minutes) — a linear surrogate of
the horizontal loop with the real filter and the real PID:

```bash
python tools/tune_horizontal.py
```

**A new scenario**: copy any YAML in `sim/scenarios/`, change the environment block and the
pass criteria, and `python -m sim.runner sim/scenarios/yours.yaml`.

---

# PART 3 — What every file does

## `control/` — the flight code (this is what would run on the Pi)

| File | What it does |
|---|---|
| `main.py` | The 50 Hz loop. `ControlLoop` is pure computation (no threads, no sockets — that's why it's directly testable); `Node` wraps it with the serial link, telemetry thread, pose subscriber, and logger. |
| `msp.py` | MSP v1 and v2 from scratch: XOR checksum, CRC8/DVB-S2, an incremental parser that resynchronises after corruption, thread-safe transport with separate read and write locks. |
| `frames.py` | Camera → body → gravity-levelled transforms, and the sign convention written down in one place. The comment at the top is the thing that prevents the classic fly-away bug. |
| `filter.py` | Constant-velocity Kalman filter per axis, with measurement variance that scales with range². Plus a first-order low-pass for altitude. |
| `guidance.py` | The PID class (anti-windup, filtered derivative, prefers a measured derivative over differencing) and RC channel synthesis honouring the FC's channel map and stick signs. |
| `altitude.py` | The Pi-owned altitude loop: hover feedforward, ground-effect compensation, descent staging gated on horizontal accuracy, and a touchdown detector with two signatures. The altitude source is an interface so a rangefinder can be fused in later without touching the controller. |
| `state_machine.py` | The mission FSM. Every transition has an explicit condition and logs its reason. |
| `safety.py` | Envelope limits and the watchdog. Trips latch. A trip stops transmission. |
| `pose_sub.py` | UDP subscriber. Keeps the newest pose per tag ID. Defines the wire format shared with the C++ node. |
| `telemetry.py` | The 20 Hz MSP poll thread. |
| `logger.py` | Queue → CSV on a background thread so the control loop never waits on disk. |
| `config.py` | Schema validation. Fails at startup, not at altitude. |

## `sim/` — the mock hardware and world (never imported by `control/`)

| File | What it does |
|---|---|
| `fc_model.py` | The mock flight controller. Parses MSP RC, maps sticks to lean angle and thrust, serves attitude/altitude/status/battery, and implements the RC-timeout failsafe. The stick sign convention lives here in one place with the firmware reasoning as a comment. |
| `dynamics.py` | Point-mass translation with first-order attitude lag, quadratic drag, and a ground-effect model. |
| `camera_sim.py` | Renders the pad from the current pose through the calibration matrix, applies blur/noise/shear, and runs the **real** detector on the result. |
| `environment.py` | Wind (steady + Dryden-style gusts), altitude sensor noise/bias/drift/dropout, lighting, occlusion, frame drops, latency, battery sag, a moving pad. |
| `harness.py` | Wires it together. Two transports: `DirectRunner` (in-process, still through the MSP codec and the FC handler) and `PtyRunner` (subprocess on a real pseudo-terminal). |
| `runner.py` | Loads scenario YAML, runs, evaluates pass criteria, writes CSVs, prints the table, does Monte Carlo. |
| `viz.py` | Matplotlib replay. |
| `scenarios/*.yaml` | 17 declarative scenarios. |

## `vision/` — the detector

`py/detector_py.py` (Python, used by the sim and CI) and `src/` (C++, for the Pi) implement
the same thing: AprilTag 36h11 detection, planar PnP with `SOLVEPNP_IPPE_SQUARE`, an
iterative fallback for the fronto-parallel case where IPPE returns a mirrored solution, and
a reprojection-error gate. `include/frame_source.hpp` abstracts the frame source so the same
detector runs on a real camera, a video file, or frames pushed by the simulator.

## `tests/` — 19 files, 134 tests

The ones worth knowing about:

- `test_camera_sim.py` — the render/detect round trip across altitude and tilt. **The most
  important test in the repo.** If the renderer is wrong, every scenario passes while proving
  nothing.
- `test_no_sim_awareness.py` — greps `control/` for simulator vocabulary. Mechanical
  enforcement of the project's central claim.
- `test_link_pty.py` — the MSP transport over a real pseudo-terminal.
- `test_rc_convention.py` — pins the stick sign convention to a physical assertion
  ("pitch channel 1700 → accelerates backward") so a bench test can overturn it in two lines.
- `test_frames.py` — hand-computed transform expectations, including the tilt-compensation case.
- `test_state_machine.py` — every transition including aborts and timeouts.
- `test_fc_model.py` — the mock FC's failsafe is a tested behaviour, not decoration.

## `config/`, `calib/`, `tools/`, `docs/`

- `config/vehicle.yaml` — every gain, limit, threshold.
- `config/tags.yaml` — the pad: three tags, their sizes, their offsets from the pad origin,
  and the altitude band each one is valid in.
- `calib/camera_640x480.yaml` — **placeholder** intrinsics, marked as such. A real camera
  needs `calib/calibrate.py` run on 25–35 chessboard images.
- `tools/tune_horizontal.py` — the linear surrogate for gain tuning.
- `tools/bench_rc.py` — props-off hardware bench tool: sends fixed sticks, prints attitude.
- `tools/replay.py` — re-run the control loop against a logged flight with new gains.
- `docs/findings.md` — the fifteen falsified assumptions.

---

# PART 4 — What is not done

Being explicit, because the README shouldn't oversell:

1. **It has never flown.** No aircraft, no flight controller, no camera.
2. **The Monte Carlo is incomplete.** Four of seventeen scenarios have 30-seed numbers.
   `alt_sensor_dropout` passes only 21/30 — the single-seed run hid that. Run Step 6 to get
   the full table.
3. **The C++ detector has never been compiled.** It's written; CI is where it first builds.
4. **`wind_gust_8ms` fails.** The vehicle holds ±0.5 m in 8 m/s gusts but never satisfies the
   "error under 20 cm for 2 seconds" gate, so it never starts descending and eventually hands
   back. The gate needs to be gust-aware. This is a real failure, reported as one.
5. **The stick sign convention is unverified.** A 60-second props-off bench test settles it;
   until then it's a reading of the firmware source, isolated so it's a two-line fix.
6. **The altitude loop assumes rangefinder-grade altitude.** With one altitude source,
   `alt_sensor_dropout` collapses to "no altitude → stop commanding → let the FC land",
   which tests less than its name suggests. Real barometer-vs-rangefinder fusion is future
   work; the interface for it exists.
7. **Guidance is single-loop PID**, not a cascade. P on position, D on Kalman velocity.
   A proper cascade (position → velocity setpoint → lean angle) is the textbook answer and
   would be the next control improvement.

---

# Quick reference

```bash
cd ~/evtol && source .venv/bin/activate         # every new terminal

python -m pytest -m "not slow"                  # 134 tests, ~2 min
python -m pytest -m "not slow" -v               # with test names
python -m pytest tests/test_camera_sim.py -v    # one file

python -m sim.runner sim/scenarios/nominal_calm.yaml              # one landing, 15 s
python -m sim.runner sim/scenarios/nominal_calm.yaml --transport pty   # real serial, 20 s
python -m sim.runner --all                                        # 17 scenarios, 8 min
nohup python -m sim.runner --all --monte-carlo 30 --out sim/results/mc > mc.log 2>&1 &

python -m sim.viz sim/results/nominal_calm_seed0.csv              # interactive plots
python tools/record_run.py sim/scenarios/demo_offset_approach.yaml --out media   # GIF + stills
make media                                                        # both demo runs
python tools/tune_horizontal.py                                   # gain sweep

make ci                                          # lint + tests
make scenarios                                   # all 17 scenarios
make vision                                      # C++ build (never yet compiled)
```
