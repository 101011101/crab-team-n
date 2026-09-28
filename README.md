# Crab Team N

A swarm of cheap camera robots explores a maze like ants, builds one shared world from all their views, concentrates on high-biodiversity areas, and clears blocked paths for each other.

- `docs/` — vision, PRD, architecture, pitches. Read [docs/refined-prd.md](docs/refined-prd.md) first.
- [docs/robot-interface-spec.md](docs/robot-interface-spec.md) — the robot/laptop contract. For the robotics team.
- [docs/tickets.md](docs/tickets.md) — proof components and their fixed interfaces.
- [docs/video-capture-guide.md](docs/video-capture-guide.md) — how to shoot video that reconstructs. Read before filming.
- `firmware/scout_bot/` — robot firmware. Flash to each car.
- `central/` — laptop coordinator. Vision, control, operator display. Talks to real robots.
- `pipeline/` — **video → 3D world → 2D map.** Five stages. [pipeline/README.md](pipeline/README.md)
- `control/` — **the control loop.** Reach field, push planning, simulator. [control/README.md](control/README.md)
- `live/` — **robots wired to all of it.** Video in, map out, commands back. [live/README.md](live/README.md)
- `results/` — committed outputs, and what each one shows. [results/README.md](results/README.md)
- `viewers/` — single-file HTML pages with the data baked in. [viewers/README.md](viewers/README.md)
- `tools/video3d/` — measurement toolbox: feasibility gate, monocular depth, splat notebook. Not the pipeline.
- `../05.3_Multi_Functional_Car/` — original Freenove sketch. Reference only. Not used.

## Where a thing lives

| You want | Go to |
| --- | --- |
| Turn video into a world | `pipeline/`, run `run_all.sh` |
| Drive robots in that world | `control/run_sim.py` |
| See what came out | `results/README.md` |
| Drive real hardware | `live/`, after [docs/live-bringup.md](docs/live-bringup.md) |
| Robot comms only | `central/`, and the bring-up order below |
| Look at the 3D world | `viewers/world3.html` |

`pipeline/` and `control/` run offline on recorded video and need no robot. `live/`
and `central/` need hardware.

The whole chain, end to end:

    robot camera -> pipeline (3D world) -> stage4 (2D map) -> control (state machine)
         ^                                                          |
         +---------------------- track speeds ----------------------+

`live/run_live.py` runs that loop against real robots. Start with
[docs/live-bringup.md](docs/live-bringup.md); the subnet step is where this fails.

## State

See [docs/build-state.md](docs/build-state.md) for what is proven, what is written but unproven, and what is blocked on the team.

Regression check before committing any rung:

    cd central && ./.venv/bin/python test_pipeline.py

## Bring-up order

Do not skip a step. Each step needs the step before it.

1. **Set the Wi-Fi.** In `firmware/scout_bot/scout_bot.ino`, set `WIFI_SSID`, `WIFI_PASS`, and the `staticIP` / `gateway` octets to match your router.
2. **Flash car 1** with `#define BOT_ID 1`. Flash car 2 with `#define BOT_ID 2`. Nothing else changes between the two.
3. **Read the serial monitor at 115200 baud.** It prints the robot IP. Confirm it matches `config.py`.
4. **Open `http://<robot-ip>/`** in a browser. Drive with WASD. Confirm the camera image appears. This proves the robot before any laptop code runs.
5. **Install the laptop side:** `cd central && python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt`
6. **Set the robot IPs** in `central/config.py` → `BOTS`.
7. **Run preflight:** `./.venv/bin/python preflight.py`. It checks commands, video, and motors on every robot. Fix all failures before continuing.
8. **Calibrate the colour** in arena light: `./.venv/bin/python calibrate.py`. Click a green block. Adjust the sliders until the mask is solid white on blocks and black elsewhere. Press `P`. Paste the printed values into `config.py`.
9. **Run the mission:** `./.venv/bin/python mission.py`. Press `SPACE` to start.

## Operator keys

| Key | Action |
| --- | --- |
| `SPACE` | Start or stop the mission. Stop halts both robots. |
| `1` / `2` | Force a robot offline. Demonstrates dropout and reassignment without touching hardware. |
| `R` | Restore all force-dropped robots. |
| `M` | Show the detection mask instead of the camera image. Use when detection misbehaves. |
| `S` | Write `survey_<timestamp>.json`. |
| `Q` | Quit. Stops both robots and prints the final survey. |

## Interfaces

Robot, port 80:

| Route | Meaning |
| --- | --- |
| `/cmd?l=<n>&r=<n>` | Set left and right track speed, -4095 to 4095. Resets the watchdog. |
| `/stop` | Stop now. |
| `/status` | JSON: id, uptime, battery volts, command age, current speeds. |
| `/` | Manual WASD page. Debug only. |

Robot, port 81:

| Route | Meaning |
| --- | --- |
| `/stream` | MJPEG stream. |
| `/jpg` | One JPEG frame. Fallback. |

## Safety

- The robot stops its motors after 500 ms without a `/cmd`. The robot enforces this, not the laptop.
- Movement is therefore a standing intent. The laptop resends the current command at 10 Hz.
- A laptop crash, a Wi-Fi drop, or `Q` all stop the robots within 500 ms.

## Tuning

All values are in `central/config.py`.

| Symptom | Change |
| --- | --- |
| Blocks not detected | `hsv_low` / `hsv_high`. Recalibrate in the real light. |
| Random green noise detected | Raise `min_area_px`. |
| Robot stops too far from the block | Lower `claim_area_fraction`. |
| Robot drives into the block | Raise `claim_area_fraction`. |
| Robot weaves toward the target | Lower `steer_gain`. |
| Robot turns too slowly to track | Raise `steer_gain`. |
| Robot hums but does not move | Raise `min_move`. |
| Same block counted twice | Raise `refractory_s` or `merge_window_s`. |
| Two blocks counted as one | Lower `merge_window_s` or `merge_size_ratio`. |

## Known limits

State these in the demo. Do not claim more.

- **No global map.** There are no robot coordinates and no arena map. Each robot reacts only to its own current camera frame.
- **Deduplication is approximate.** Two sightings cannot be proven to be the same block without positions. Repeat sightings by one robot are blocked by a refractory timer. Sightings by both robots are merged if they are close in time and apparent size. Two different blocks of similar size, found by both robots within `merge_window_s`, will merge and the count will read low.
- **Reassignment is coverage bias, not area handover.** A sector only sets which way a robot turns while searching. On a dropout, the survivor widens its sweep. No unexplored region is tracked.
- **Distance is apparent size.** A large far block and a small near block look the same to the controller.
- **No obstacle avoidance.** The ultrasonic sensor is present on the hardware but unused. Keep the arena clear.

## Untested

- The firmware is not compile-checked in this repo. Build it in the Arduino IDE.
- `firmware/scout_bot/camera_pins.h` holds the Freenove ESP32-WROVER CAM pin map. Verify it against the camera example sketch that already worked on your board.
- The laptop pipeline is verified against synthetic frames only. No robot hardware was connected.
