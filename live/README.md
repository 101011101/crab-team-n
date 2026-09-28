# live — robots to world to map to commands, in one process

The offline chain in [`pipeline/`](../pipeline/README.md) turns recorded video into a
world. This turns a running robot into one, and drives it.

| File | Does |
| --- | --- |
| `preflight_live.py` | Checks every link. **Run this first, every time.** |
| `discover.py` | Finds robots on whatever subnet you are actually on. |
| `robot_source.py` | `Fleet`: newest frame in, track speeds out, per robot. |
| `map_service.py` | Frames → COLMAP registration → 2D map. Own thread, slow. |
| `pose_source.py` | Where each robot is: SfM fix, dead reckoning between fixes. |
| `live_loop.py` | The state machine at 10 Hz. Map + pose in, track speeds out. |
| `run_live.py` | Wires the three threads together and prints a status board. |
| `bridge.py` | `TelemetryBridge`: serves the dashboard JSON on :8787. Read-only. |

## Three threads, three rates

    Fleet        MJPEG in, /cmd out                  ~10 fps per robot
    MapService   frames -> 3D model -> 2D map        one round per ~25 s
    LiveLoop     map + pose -> track speeds          10 Hz

**The control loop never waits for the map.** COLMAP takes tens of seconds; a robot
that stops driving while it waits for a better map is worse than a robot driving on a
map that is 30 seconds old. The map is swapped in atomically and the loop reads
whatever is current.

## How a robot knows where it is

Two sources. The first is correct, the second is always available.

1. **SfM registration.** Register the robot's current frame against the existing
   model. The registered camera centre *is* the robot, in exactly the frame the map
   is drawn in — there is no extrinsic calibration to get wrong, because there is no
   extrinsic.
2. **Dead reckoning.** Integrate the commanded track speeds. Drifts without bound,
   wrong the moment a wheel slips, but smooth and never absent.

A registration result **corrects** the dead reckoner. Between corrections the dead
reckoner carries the robot.

## The weak link, stated plainly

**Registering 320x240 ESP32 frames against a model built from 1080p phone frames is
unproven and is the most likely thing to fail.**
[`docs/robot1-reconstruction.md`](../docs/robot1-reconstruction.md) measured QVGA
frames at a median 116 SIFT features against 2,340 for the phone, and found the focal
unidentifiable at that size.

`run_live.py` prints the registration rate every second. If it sits at zero, the
system is running on dead reckoning alone and the status board says so out loud. That
is a degraded mode, not a broken one — but do not mistake it for working.

## Safety

- Ctrl-C stops every robot before exiting. So does any exception in any thread.
- A robot with no usable pose for 8 s is commanded to stop, not driven blind.
- `--dry-run` sends no drive command at all. Use it first, every time.
- The robot's own 500 ms watchdog stops it if the laptop goes away.

## Measured

One map round, 3 frames against the 166-camera prior, this laptop, CPU only:

| Stage | Cost |
| --- | --- |
| Feature extraction, 3 frames at 1600 px | ~4 s |
| Matching, 180 pairs (55 priors x 3 new) | ~3 s |
| `image_registrator` + map rebuild | ~1 s |
| **Round total** | **8.3 s** |

**3 of 3 frames registered, poses returned in map coordinates.** Those were phone
frames, from the same camera as the prior model — which is what makes this a test of
the *wiring*, not of the robot camera.

The prior images are subsampled (`prior_stride`, default 3). All 166 against 6 new
frames is 996 pairs and blows the round budget; every third still gives each new
frame plenty to lock onto.

## What is proven and what is not

| | |
| --- | --- |
| proven | The offline chain. 179/183 frames, 0.52 px, a map, a working loop in sim. |
| proven | The live chain end to end: frames in, registered, pose out, map written. |
| proven | `BotLink` video and commands, against real hardware, in earlier sessions. |
| **not proven** | Registering **robot-camera** frames into the phone-built model. |
| **not proven** | The whole loop against real robots. Never run with hardware attached. |

Every piece is individually exercised and the seams between them are tested. The
end-to-end run needs robots on a bench.

## Hooking up the dashboard on `main`

`dashboard/` on the `main` branch already takes real hardware:

    NEXT_PUBLIC_CAM_0 = http://<robot-1-ip>:81/stream
    NEXT_PUBLIC_CAM_1 = http://<robot-2-ip>:81/stream

Those are the same MJPEG URLs `Fleet` reads, and both can read them at once. The
dashboard's reconstruction panel wants a splat; this branch produces a sparse cloud
and a 2D grid, not a splat, so that panel stays on its own data until someone trains
one (`tools/video3d/train_splat_colab.ipynb` needs a CUDA machine).
