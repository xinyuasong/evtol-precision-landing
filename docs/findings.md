# Findings — assumptions the simulator falsified

Every entry: what the design guide assumed, what the simulator showed, what changed.
Kept current as scenarios land. The `[REF]` sections of the design guide were
reconstructed defaults; this is the list of the ones that broke on contact with a
simulator that renders and detects instead of injecting ground truth.

## 1. Concentric nested AprilTags corrupt the outer code
**Assumed:** three tag36h11 tags of decreasing size printed inside one another, so the
same pad origin is visible from 12 m down to 8 cm.
**Shown:** `cv2.aruco` decodes a tag by sampling the central ~74% of every cell
(`perspectiveRemoveIgnoredMarginPerCell = 0.13`). An inner tag lands on the four centre
samples of the outer tag and flips its bits; the outer tag stops decoding at every
altitude. Only 27 of 587 tag36h11 IDs even have a white 2x2 centre, and the inner
footprint would have to be < 13 mm to dodge the samples.
**Changed:** side-by-side tags registered to one origin through per-tag offsets
(`config/tags.yaml`) — what AprilTag calls a tag bundle. This forced the wire format to
carry the full tag→camera rotation instead of yaw only, because an in-plane offset must be
rotated by the actual pose (face-on is a 180° rotation about X, not a yaw). Concentric pads
need a hollow-centre family (`tagCustom48h12`), which OpenCV does not ship.

## 2. The Kalman measurement variance was ten centimetres
**Assumed:** `r0 = 0.01` for the constant-velocity KF, described as "measurement noise".
**Shown:** `r0` is a variance; 0.01 m² is a 10 cm σ, against a real pose noise of ~1 cm.
The filter trusted the model over the measurement so heavily that the position estimate
lagged by hundreds of milliseconds. `tools/tune_horizontal.py` found no Kp/Kd pair that
settled: every one limit-cycled at 0.2–1.2 m amplitude, noise-free and latency-free.
**Changed:** `r0 = 0.0004` (2 cm σ at 2 m, scaled with range²), `q = 1.0`. With that,
Kp/Kd = 1.0/1.6 settles a 0.5 m offset in ~2.5 s with < 10 cm overshoot. The guide's
0.45/0.02/0.25 gave a 10 s oscillation even after the variance fix.

## 3. `CORNER_REFINE_APRILTAG` costs 6x, not 15%
**Assumed:** the AprilTag corner refiner "costs ~15% throughput and buys a large accuracy
improvement".
**Shown:** on OpenCV 4.13, 640x480, one frame: 86 ms with `CORNER_REFINE_APRILTAG`,
15 ms with `CORNER_REFINE_SUBPIX`, 11 ms with none. Pad-origin error under SUBPIX is
< 1 cm across the altitude/tilt sweep (`tests/test_camera_sim.py`).
**Changed:** SUBPIX is the default; the refiner is a config field. The guide's latency
budget line "Detection + PnP 12–30 ms" is only true without the AprilTag refiner.

## 4. A ground-effect hover looks exactly like a touchdown
**Assumed:** touchdown = altitude < 10 cm and vario ≈ 0, sustained 0.5 s.
**Shown:** in ground effect the vehicle settles into a stable hover at 6–10 cm with the
throttle only ~20% below hover; altitude and vario both sit inside the detector's window.
The detector fired, the state machine disarmed, and the vehicle dropped 6 cm at 1.0 m/s.
**Changed:** the detector also requires the commanded throttle to be well below hover
(`touchdown.max_throttle_us`), the altitude threshold is 4 cm, and FINAL lets the setpoint
run 15 cm below the ground so the P term keeps pushing through the cushion. Because that
floor would otherwise push forever if detection never fires, FINAL has a hard timeout that
disarms anyway (`touchdown_detect_fails` scenario).

## 5. The 50 Hz RC stream blocked behind the 20 Hz telemetry poll
**Assumed:** one lock on the serial port is fine; requests are short.
**Shown:** only over the real pty transport, never in direct mode. `MSPLink.request()`
held the port lock while waiting up to 150 ms for a reply; `set_rc()` from the control
thread queued behind it, the loop overran 132 ms on its second tick, and the safety layer
(correctly) stopped the stream. The direct transport calls the codec synchronously and
cannot exhibit this.
**Changed:** separate write and request locks. Writes never wait behind a read. This is
the class of bug the pty transport exists to catch, and it caught one on the first run.

## 6. Exact fronto-parallel squares break `SOLVEPNP_IPPE_SQUARE`
**Assumed:** IPPE_SQUARE is the right solver for a planar tag, full stop.
**Shown:** for an exactly fronto-parallel, pixel-aligned square (which a level hover over
the pad produces in simulation and can produce on a rig), OpenCV 4.13's IPPE returned the
mirrored solution: identity rotation, reprojection error of a full tag width. Off-axis by
any amount it is fine.
**Changed:** solve with IPPE_SQUARE, check reprojection, fall back to ITERATIVE when it
exceeds the gate. Cheap, and the gate was already there.

## 7. Detection continuity is a time, not a tick count
**Assumed:** "tag seen in 5 consecutive frames" implemented as a per-control-tick counter.
**Shown:** the control loop runs at 50 Hz and the camera at 30 Hz, so two of every five
ticks see no new pose. The counter reset constantly, ACQUIRE flapped back to SEARCH five
times in six seconds, and TRACK dropped out with "tag lost" while the tag was in view.
**Changed:** continuity is judged by time since the last detection
(`state_machine.detection_gap_s`); the counter only resets when that gap is exceeded.

## 8. Telemetry that has never arrived is not "stale"
**Assumed:** staleness = `now - t_last > timeout`, with `t_last = None` treated as stale.
**Shown:** on the first control tick the poll thread had not completed a request yet;
with the FC armed, the altitude-stale trip latched at t = 0 and the run ended before it
began.
**Changed:** a startup grace (`safety.startup_grace_s`); a value that has never arrived
trips only after the grace, a value that has arrived trips on its own timeout.

## 9. The stick sign convention is an unverified assumption shared by both sides
**Assumed:** pitch channel above mid = nose down = fly forward.
**Shown:** nothing in simulation can show this. The mock FC and `control/` share the
convention, so any consistent choice lands the vehicle and a shared error is invisible to
every scenario. Reading Betaflight/INAV: `rcCommand = rcData - 1500`, positive command
targets positive angle, positive pitch is nose up — which argues for the opposite sign.
**Changed:** the convention lives in exactly one place (`PITCH_STICK_SIGN` /
`ROLL_STICK_SIGN` in `sim/fc_model.py`) with `rc.pitch_sign = -1` in `config/vehicle.yaml`
to match, and `tests/test_rc_convention.py` asserts the physical statement
(channel 1700 → accelerates −X). A 60-second props-off bench test on real hardware
settles it; flipping it is a two-line change and the test says which two.

## 10. The lens decides the acquisition altitude, not the tag
**Assumed:** a 0.40 m tag acquires from 10–12 m.
**Shown:** with a 500 px focal length at 640x480, 0.40 m at 10 m is 20 px — 2.5 px per
cell — and does not decode. Reliable acquisition starts around 6 m (33 px).
**Changed:** `valid_alt_m` for the large tag is [1.0, 6.0]; scenarios start at 3 m. The
placeholder intrinsics are marked as such; a longer lens or a bigger tag changes this.

## 11. A tag hard against the board edge gets merged with the board
**Assumed:** a 5 cm white margin around the tags is enough quiet zone.
**Shown:** aruco's `filterTooCloseCandidates` treats the white board's outline and the
large tag's outline as duplicate quads when their corners are within 12.5% of the tag
perimeter of each other, and keeps the larger — the board, which is not a marker. The
large tag vanished at specific image positions only. Lowering `minMarkerDistanceRate`
fixed it but that filter is what suppresses duplicate quads under blur and noise.
**Changed:** the board margin is 15 cm, so the two quads are never near-duplicates. The
filter stays at its default. On a real pad: leave a wide white border around the outer tag.

## 12. `max_horiz_err_m = 5 m` can never catch a sign inversion
**Assumed:** the guide's Safety layer says the 5 m plausibility gate "catches the
sign-inversion bug: error grows instead of shrinking, and this trips before the vehicle
builds real speed."
**Shown:** from 3 m the tag leaves a 640x480 field of view at ~1.4–1.9 m of error. The gate
never sees 5 m. Under an inverted `R_bc` the vehicle accelerates away at full authority,
loses the tag at 1.4 m, and coasts. A divergence detector (error growing monotonically
over a window) was tried; the monotonic growth lasts only ~1.2 s before the tag is gone,
and any window short enough to catch it also fires on a wind-gust excursion.
**Changed:** the 5 m gate stays as a measurement-sanity check and is no longer described
as the inversion catch. A monotonic-divergence trip exists (3 s window, 85% monotonic)
and is verified not to fire on gusts; it would catch an inversion from higher altitude.
The `sign_inversion` scenario asserts what the stack actually guarantees: the runaway
loses the tag, SEARCH times out, HANDBACK stops the stream, the FC failsafe lands. The
bench test (guide Stage 2: move the tag right, watch the roll channel) is where this bug
is meant to die, and nothing in flight replaces it.

## 13. FINAL with neutral sticks drifts downwind
**Assumed:** FINAL ignores vision and "takes the last few centimetres open-loop" with
neutral horizontal sticks.
**Shown:** in a 5 m/s steady wind the vehicle landed 1.08 m off the pad: 20 cm at
0.15 m/s is 1.3 s of free drift under a 0.6 m/s² wind load.
**Changed:** FINAL holds the last horizontal command instead of neutral; the integrator's
value is the controller's estimate of the wind. Touchdown error in the same scenario: 1.2 cm.

## 14. Gains were expressed in stick fraction, so "raise authority" changed loop gain
**Assumed:** `authority` is a safety clip you raise once the vehicle flies well.
**Shown:** controller output was scaled by `authority` before becoming a stick value, so
raising it from 0.20 to 0.50 multiplied every gain by 2.5. The gust scenario at 0.50
oscillated at ±12.5° lean and diverged — not because the wind won, but because the loop
was retuned by accident.
**Changed:** `guidance.stick_scale` (fixed) sets the lean per unit of controller output;
`guidance.authority` only clips. Raising authority now adds headroom without touching
gain. (The gust scenario still fails, for a different reason: see README Status.)

## 15. Simulator parameter corrections (recorded against myself)
- Drag coefficient 0.30 N/(m/s)² was ~10x too high for a 2 kg airframe; 5 m/s of wind
  became 3.75 m/s², beyond any authority. Now 0.05.
- The run started with an unarmed FC: the vehicle fell 3 m before the override engaged.
  The pilot now hovers manually until the switch is flipped, as on the real aircraft.
- The mock FC failsafe disarmed at 12 cm while still on the ground-effect cushion and
  dropped the aircraft. It now disarms on the ground.
