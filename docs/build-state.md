# Build state

> What is proven, what is written but unproven, and what is not built. Updated as rungs land.

## Proven — runs and is tested

| Component | File | Evidence |
| --- | --- | --- |
| Reactive search, dropout reassignment | `central/mission.py` | `test_pipeline.py`, 9 checks |
| Track-pair conditioning | `central/drive.py` | covered by both suites |
| Waypoint motion controller | `central/motion.py` | `test_motion.py`: 8 start poses, sign checks, monotonic approach, heading hold |
| Stall detection | `central/motion.py` | wedge scenario fires in 1.7 s |
| Arena simulator | `central/sim.py` | straight line, spin, watchdog stop at 0.5 s |
| Waypoint execution end to end | `central/demo_waypoints.py` | 10/10 scenarios, worst gap 5.0 cm |
| Video feasibility gate | `tools/video3d/feasibility.py` | validated against three synthetic clips of known motion, plus a real phone clip |
| Depth reconstruction | `tools/video3d/reconstruct.py` | runs on MPS, 92 ms per frame at QVGA; geometry unverified pending a good clip |
| Solver and allocator | `central/solver.py`, `central/allocator.py` | `test_solver.py`, 13 checks, worst solve 1.1 ms |
| Integrated mission | `central/mission_sim.py` | 4 of 4 species in 43.2 s, identical across runs |
| Marker pose | `central/markers.py` | `test_markers.py`; worst 1.80 cm through real corner extraction |

## Written, not yet proven

| Component | File | Missing |
| --- | --- | --- |
| World model | `central/world.py` | no solver on top of it yet |

## Not built

- Species detection from real camera frames. The simulator supplies sightings directly.
- Biodiversity scoring and recruitment. Blocked on the scoring rule.
- Anything on physical hardware. Nothing in this repo has touched a robot.
- Integrated simulated mission: solver drives the simulator through the full corridor scenario.
- Depth reconstruction.
- Species detection and biodiversity scoring.

## Findings so far

### The feasibility tool called a real walk rotation-only

- The first verdict on `IMG_0300.MOV` was DO NOT ATTEMPT, on the claim that the camera never translated. That claim was false: the clip walks forward along corridors, confirmed by inspecting frames and by epipolar geometry.
- Cause: the parallax test used a homography residual. A homography absorbs nearly all of a corridor walk, so the residual stayed small while the translation was real.
- Fixed by comparing essential-matrix inliers against homography inliers, which a rotation cannot fake. The same bug had also rejected the synthetic clip of known sideways motion; it now passes.
- The clip now reads MARGINAL: usable, but forward-dominant, which triangulates the view ahead poorly. A strafing clip is still worth shooting.

### Calibration is worth fifteen minutes per camera

- Marker pose with a proper calibration: worst 1.80 cm.
- The same code with guessed intrinsics: 16.5 cm mean, nine times worse.
- The flip ambiguity in planar markers returned confidently wrong poses, 753 mm and 55 degrees off, from exact corners. A wrong pose is worse than no pose because the robot drives on it. Ill-conditioned geometry is now rejected instead of guessed.

### COLMAP recovers the cameras but not the corridor

- 124 of 124 frames registered, 1.245 px mean reprojection error, solved HFOV 57.6 deg. Pose recovery was a complete success.
- The reconstructed structure is not: 21 points per frame in the first third of the walk, 9 in the middle, and **0 in the final third** (14 points from 28 frames).
- Tracking and triangulating are different problems. Matching only needs a point recognisable in two frames; triangulating needs the two views to see it from different positions. Walking toward a scene gives nearly parallel rays.
- A GPU would cut the 17 minutes to about one. It would not add a point to the empty third. See [sfm-colmap-result.md](sfm-colmap-result.md).

### The assumed field of view was fine after all

- COLMAP solved 57.6 deg against the 60 deg the reconstruction assumed. Within 2.4 deg.
- An earlier note here treated COLMAP's 2304 px starting prior as evidence the FOV was wrong. That prior is an initialisation heuristic, not a measurement.

### Depth on the laptop is fast enough

- Depth Anything V2 Small, MPS, 320x240: 92 ms per frame, about 11 fps.
- Two robots at 5 fps each is viable locally. Full rate on three robots is not.
- This is the number that decides whether a cloud GPU is needed for the live loop. It is not.

### The controller pushed against a wall for forty seconds

- Found by driving the simulator into the corridor block.
- The robot never arrived and never reported failure, so nothing upstream could learn the attempt was hopeless.
- Fixed. A robot commanded to move that does not move is now reported as stalled after one second. This is the execution-failed release condition the allocator needs.

### `types.py` shadowed the standard library

- A module named `types` in `central/` captures that name for every import in the package.
- Renamed to `geometry.py`.

## Numbers measured so far

| Quantity | Value | Source |
| --- | --- | --- |
| Waypoint arrival accuracy | about 5 cm | simulator, tolerance-limited |
| Depth inference, QVGA, MPS | 92 ms per frame, 10.9 fps | measured on this laptop |
| Marker pose, calibrated | 1.80 cm, 0.48 deg worst | rendered QVGA, real ArUco extraction |
| Marker pose, guessed intrinsics | 16.5 cm, 4.4 deg mean | same path, no calibration |
| COLMAP registration | 124/124 frames, 1.245 px | 17 min, CPU only |
| COLMAP solved HFOV | 57.6 deg | vs 60 deg assumed |
| Points per frame, final third of walk | 0 | the forward-motion degeneracy, measured |
| Worst solve, ten-place graph | 1.1 ms of a 16 ms frame | `test_solver.py` |
| Simulated mission | 43.2 s, 4 of 4 species | `mission_sim.py` |
| Stall detection latency | 1.7 s | simulator |
| Firmware command timeout | 500 ms | firmware constant, honoured by the simulator |
| Control rate | 20 Hz | `demo_waypoints.py` |

- Every distance above depends on `MOTION.track_width_m` and `MOTION.max_speed_mps`, which are **unmeasured placeholders**. Measure both before trusting any figure in metres.

## Blocked on the team

| Item | Blocks |
| --- | --- |
| A strafing video, phone, per the capture guide | the whole 3D path |
| Track width, wheel centre to centre, metres | every distance the motion controller produces |
| Top speed, timed over 2 m at full command | the same |
| Arena shape and place count | the world graph |
| Biodiversity scoring rule | branch ranking in the allocator |
