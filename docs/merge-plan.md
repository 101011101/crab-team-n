# Merging with `htnrobomaze` (origin/main)

> Analysis 2026-09-19. Their branch was read, not merged. Nothing was modified.

## Summary

- The two codebases are **complementary, not duplicated**. They stop at teleop; we start at the planner.
- Their transport and firmware are better than ours. Our marker solver is better than theirs.
- **Neither side has a single measured metric constant.** Both are all nulls. They were more disciplined about refusing to fake them.
- They have a real camera calibration tool. We have none. That is the gap our placeholders have been waiting on.

## Take theirs

| Thing | Why |
| --- | --- |
| **Firmware** `firmware/maze_robot/` | Watchdog runs in its own FreeRTOS task at priority 4 pinned to core 1, ticking every 5 ms, independent of the network loop. Ours checks in `loop()` and only while already driving. |
| **Transport** UDP binary, `MZ01` magic | Arming, session tokens that invalidate on stop, replay rejection, stale-command rejection. Ours is an unauthenticated HTTP GET any host on the LAN can drive. |
| **Clock sync** | Maps ESP32 uptime to host monotonic with an uncertainty bound, and the camera stream carries `X-Capture-Ms` and `X-Boot-Id`. Without it a pose is stamped at HTTP receipt, which on a phone hotspot is the dominant error for a moving robot. |
| **Camera calibration** `perception/calibration.py` | 9x6 board, 15 sample minimum, 1.0 px RMS gate, SHA-256 signature that invalidates on any resolution or mirror change. We have nothing. |
| **Ground projection** | Pixel to metric floor position. Our `vision.py` gives only a bearing and an area fraction; this is what turns a blob into a target. |
| **Teleop** | We have none, and it is both the bring-up path and the demo fallback. |
| **Pose lifecycle** `manager.py` | Aging, last-known hold with decaying confidence, provenance, cross-robot pose chaining. |
| **Config loader** `localization/config.py` | A null placeholder can never silently become an identity pose. Worth stealing whatever else is decided. |

## Keep ours

| Thing | Why |
| --- | --- |
| **Marker solver** `markers.py` | Joint PnP over every corner of every tag, against their averaging of independent per-tag poses. Three solver flags with LM refinement on every branch, against one unrefined closed form. Leave-one-out rejection of a mis-measured map entry, which they have no equivalent of. And a measured end-to-end number through real corner extraction, 1.80 cm, against no accuracy number at all. |
| **Planner** `solver.py`, `allocator.py`, `world.py` | Their `coordination/` directory contains one docstring. |
| **Motion** `motion.py`, `navigator.py` | They stop at teleop. |
| **Simulator** `sim.py`, `mission_sim.py`, `mission_race.py` | Theirs fakes the transport only, and says so in its own docstring. Not a duplicate. |
| **3D pipeline** `tools/video3d/` | Theirs stops at dataset prep. Their AprilTag pose export is a real input to ours. |
| **Camera pan/tilt** | We have servos. Their calibration refuses anything but a fixed forward camera. |

## Landmines — settle before anyone measures an arena

1. **Tag frame z axis is flipped.** Theirs: *x right, y down, z into the printed face*. Ours: *x right, y up, z out of the printed face*. A 180 degree rotation about the tag x axis. Feed a measured tag entry across unconverted and every tag faces backwards: the robot localises behind the wall, with a clean reprojection error and no exception raised. **This is the one that will cost a day if it is missed.**
2. **`yaw_deg` means different things.** Theirs is plain Euler yaw on raw tag axes. Ours is the compass direction the printed face points, because our rotation already stands the tag upright. Same field name, different meaning, not interchangeable even after fixing the axis flip.
3. **Tag family differs.** tag36h11 via pupil-apriltags against DICT_4X4_50 via cv2.aruco. Printed sheets are not interchangeable. Recommendation: **tag36h11 through `cv2.aruco.DICT_APRILTAG_36h11`** — keeps our solver, keeps one OpenCV dependency, gains 36h11's much lower false-positive rate and their printable sheets. Requirements must then say `opencv-contrib-python`, not `opencv-python`.
4. **Wire units.** Theirs takes float -1 to 1 and owns the acceleration ramp in firmware. Ours assumes -4095 to 4095 throughout `motion.py`, `drive.py`, `sim.py`. One conversion point, and it must be exactly one.
5. **Turn-rate sign.** Their teleop and odometry use counter-clockwise positive; our `_tracks` takes clockwise positive and negates internally. Both are self-consistent alone. Merging the motion paths must not mix them.

## What already agrees — do not let anyone "fix" these

- Body frame is x forward, y left, z up on both sides.
- The optical-to-body matrix is byte-identical: `[[0,0,1],[-1,0,0],[0,-1,0]]`.
- Both use `T_A_B maps B into A`, and both compute the robot pose as `T_W_C · inv(T_R_C)`.
- Both keep camera-to-world as the primary pose and the raw tag-in-camera transform alongside.
- Both flatten to 2D with `atan2(R[1,0], R[0,0])`. Both are metres and radians internally.

## Real values worth taking

| Value | Source |
| --- | --- |
| Robot 1 at `172.20.10.2`, robot 2 at `172.20.10.4`, phone hotspot DHCP | `config/robot_1.yaml`, `robot_2.yaml` |
| Ports 4210 command, 8080 camera, 4211 clock | `robot_config.h` |
| PCA9685 at I2C `0x5F` | `motor_controller.cpp` |
| **Robot 1 needs `rotate_180` and `flip_horizontal`** | `config/robot_1.yaml`. A genuine per-robot mount fact; robots 2 and 3 do not. |
| Watchdog 300 ms, max command age 150 ms, telemetry 100 ms, motor tick 5 ms | `robot_config.h` |

- Robot 3's IP is marked "Example only". Do not trust it.
- Their committed intrinsics matrices are **test fixtures invented inside a test file**. They are not hardware data.

## The hole neither side has filled

**Pose to motion on real hardware does not exist on either branch.** Ours drives `SimRobot`; theirs stops at teleop. The loop

```
camera frame -> markers.estimate_pose() -> Navigator.step() -> RobotClient.set_desired()
```

is the single biggest gap in the merged system, and it is maybe 150 lines.

Second hole: **until somebody tapes tags to a wall and measures them, both localization stacks return `None` on every frame.** Same for intrinsics. Their `tools/calibrate_camera.py` must run on each robot before anything downstream means anything.

## Drop

- `06.2_WiFi_Cam_TcpServer/` and `05.3_Multi_Functional_Car/` — vendor samples. The first implements exactly one command, `CMD_VIDEO`, and cannot drive a robot.
- Our `firmware/scout_bot/scout_bot.ino` — superseded. Salvage the `/cam` servo aim, battery voltage, WS2812 status ring and buzzer into theirs.
- Our `botlink.py` — superseded by their `robot_client.py` and `camera_stream.py`.
- Our `config.py` `BOTS` block — superseded by their `robot_N.yaml`.
