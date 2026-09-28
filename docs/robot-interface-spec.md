# Robot interface spec

> For the robotics team. Defines the contract between one robot and the laptop. Reference implementation: [`firmware/scout_bot/`](../firmware/scout_bot/).

## Design rule

- The robot is deliberately dumb. It holds no map, no goal, and no plan.
- The robot does not know where it is. Pose is computed on the laptop from markers.
- All kinematics, path logic, and autonomy live on the laptop.
- Do not add control logic to the ESP32. Every addition there is a thing that cannot be debugged from a laptop.
- Keep the wire protocol at the lowest useful level. Wrap everything else on the computer.

## What the laptop needs from each robot

| Data | Transport | Rate | Notes |
| --- | --- | --- | --- |
| Video | MJPEG, `GET :81/stream` | 10 to 15 fps | QVGA 320x240, JPEG quality 12 |
| Single frame | `GET :81/jpg` | On demand | Fallback if the stream stalls |
| Status | `GET :80/status` | On demand, about 1 Hz | JSON, see below |

- Nothing else is required. No odometry. No IMU. No ultrasonic.
- The kit has no wheel encoders. Wheel odometry is not available. This is why markers are mandatory for pose.

### Status format

```json
{"id":1,"uptime_ms":48210,"battery_v":7.42,"cmd_age_ms":95,
 "left":1900,"right":1400,"driving":true,"pan":90,"tilt":90}
```

| Field | Meaning |
| --- | --- |
| `id` | Robot serial number. Must be unique and must match the label on the chassis. |
| `uptime_ms` | Milliseconds since boot. A reset shows as this dropping. |
| `battery_v` | Pack voltage. The laptop warns the operator below a set threshold. |
| `cmd_age_ms` | Milliseconds since the last accepted drive command. |
| `left`, `right` | Track speeds currently applied. |
| `driving` | True when the motors are commanded non-zero. |
| `pan`, `tilt` | Camera servo angles in degrees. |

## What the robot must accept from the laptop

Four endpoints. Nothing more is needed.

| Endpoint | Parameters | Effect |
| --- | --- | --- |
| `GET :80/cmd` | `l`, `r`, each -4095 to 4095 | Set left and right track speed. Resets the watchdog. |
| `GET :80/stop` | none | Stop the motors now. |
| `GET :80/cam` | `pan` 0 to 180, `tilt` 80 to 180 | Aim the camera. Absolute angles. |
| `GET :80/status` | none | Return the JSON above. |

### Why left/right and not forward/rotation

- Differential speeds and (linear, angular) are equivalent. Either can produce the other.
- The laptop converts. `l = v + k*w`, `r = v - k*w`, where `k` is the track width constant.
- Keeping `l` and `r` on the wire means no kinematics and no calibration constant live in firmware.
- Turning radius, arcs, spins, and straight lines are all one command. No new endpoint is ever needed for a new motion.

### Why absolute camera angles and not deltas

- The laptop holds the desired camera pose.
- A dropped request loses one update. A dropped delta accumulates a permanent offset.

## Safety, non-negotiable

- The robot stops its motors after 500 ms with no `/cmd`. The robot enforces this, not the laptop.
- Movement is therefore a standing intent, not an event. The laptop resends the current command at 10 Hz.
- A laptop crash, a Wi-Fi drop, or an operator stop all halt the robot within 500 ms.
- Do not raise the timeout. Do not move the check to the laptop.

## Network

- Station mode. All robots join one router. Do not use soft AP: a laptop joins one network at a time, so robots in AP mode cannot all be reached.
- Static IP per robot. Last octet is `50 + BOT_ID`.
- `BOT_ID` must be unique, must match `id` in `/status`, and must match the physical label.
- Serial numbers break assignment ties, so a duplicate ID causes two robots to receive the same task.
- Router carries no internet traffic. Cloud access, if used, goes over a separate laptop interface.

## Robotics team deliverables

| Item | Why |
| --- | --- |
| Rigid camera mount | Vibration causes motion blur. Blur breaks both marker detection and depth. This is the highest value mechanical task. |
| Camera aimed to see wall markers | Markers supply pose. A camera that only sees floor sees no markers. Confirm the tilt range covers the marker height. |
| Checkerboard calibration per camera | Intrinsics are per-camera. Ship the `.npz` with the robot, named by `BOT_ID`. Budget 15 minutes each. |
| Unique `BOT_ID`, flashed and labelled | Duplicate IDs cause duplicate task assignment. |
| Known low-voltage cutoff | Motor behaviour changes as the pack drains. Record the voltage at which driving becomes unreliable. |
| Measured track width | The laptop needs `k` to convert rotation to track speeds. |
| Two robots confirmed before the third is added | The demo is built for two. The third is additive. |

## Physical constraints from the vision side

- Markers must be 10 to 20 cm for detection at 2 to 3 m at 320x240. Print large.
- Place several markers so more than one is usually in view. Multiple visible markers sharply improve pose accuracy.
- Keep lighting constant. Auto white balance is disabled in firmware so a single colour threshold stays valid.
- Do not rely on wall texture for anything. Markers supply the features.

## Adding a capability later

- A new ant ability is a new endpoint plus a new operator in the planner.
- Example: an arm becomes `GET :80/arm?angle=<0-180>` and a `capability GRAB`.
- The same rule applies. Absolute values, no state held on the robot, no autonomy.
