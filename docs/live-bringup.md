# Live bring-up: getting laptop and robots onto one hotspot

> The single most common failure is a subnet mismatch. Read step 2 before anything else.

## 1. Pick the network

Any of these works. They differ only in how much control you have over the subnet.

| Option | Subnet | Notes |
| --- | --- | --- |
| **Travel router** | you choose | Best. Set it to `192.168.1.x` and nothing else changes. |
| Home router | usually `192.168.1.x` | Matches the firmware default. |
| Android hotspot | usually `192.168.43.x` | Works. Needs step 2. |
| iPhone hotspot | `172.20.10.x` | Works. Needs step 2. Max 5 devices. |
| macOS Internet Sharing | `192.168.2.x` | Works. Needs step 2. |
| **University / venue Wi-Fi** | anything | **Usually blocks client-to-client traffic. Avoid.** |

The laptop and every robot must be on the **same** network, and that network must let
clients talk to each other. Many phone hotspots isolate clients; a travel router never
does.

## 2. Make the robots' addresses match the network

The firmware compiles a static IP of `192.168.1.(50 + BOT_ID)`. On any other subnet
the robots join the Wi-Fi and are then unreachable.

**Easiest fix — let the network assign the address.** In `firmware/scout_bot/scout_bot.ino`:

    #define USE_STATIC_IP false

Flash, then find them:

    cd crab-team-n/live && ../central/.venv/bin/python discover.py

It scans your subnet, prints every robot that answers `/status`, and prints the
`BOTS = [...]` block to paste into `central/config.py`.

**Alternative — keep static IPs.** Edit `staticIP`, `gateway` and `dns` in the `.ino`
to match your subnet, reflash, and update `central/config.py` to match.

Either way, set `WIFI_SSID` and `WIFI_PASS` to the hotspot **before** flashing. They
are compiled in.

## 3. Flash each robot

One robot per `BOT_ID`. Nothing else differs between them.

    BOT_ID 1  ->  scout-1
    BOT_ID 2  ->  scout-2

Serial monitor at 115200 baud prints the address it got. Confirm it.

## 4. Prove the robot before any laptop code runs

Open `http://<robot-ip>/` in a browser. Drive with WASD. The camera image must appear.
If this does not work, nothing downstream will.

## 5. Check every link

    cd crab-team-n/live
    ../central/.venv/bin/python preflight_live.py \
        --work ../../out/world3 --prior ../../out/world3/sparse

Six checks in dependency order: laptop address, subnet match, ping, command port 80,
video port 81, and the prior COLMAP world. Fix them top down. Do not skip one.

## 6. Dry run — no robot moves

    ../central/.venv/bin/python run_live.py \
        --prior ../../out/world3/sparse --work ../../out/world3 --dry-run

Watch the status board. You want `frames` climbing for every robot, and the map
service completing rounds. Give it two or three rounds (about a minute) before judging
the registration rate.

## 7. Live

    ../central/.venv/bin/python run_live.py \
        --prior ../../out/world3/sparse --work ../../out/world3

Ctrl-C stops every robot.

## What you are watching on the status board

- **`frames`** climbing — video is flowing. Flat means the stream is down.
- **`registered` / `rate`** — how often a robot frame locks into the 3D world. **This
  is the number that matters.** Zero means the system is navigating on dead reckoning
  alone; it will drift and it will not recover on its own.
- **`conf`** per robot — 1.0 just after a fix, decaying with distance travelled since.
- **`mode`** per robot — `drive`, `push`, `explore`, `idle`, `stale-pose`, `offline`.

## If nothing registers

Expected, and the honest answer is that this is the unproven link. In order:

1. Give it longer. One round is 6 frames; the rate is meaningless before three rounds.
2. Improve the light. QVGA frames in dim light carry almost no SIFT features.
3. Drive the robot **across** the scene, not into it. Forward motion triangulates at a
   near-zero angle. See [video-capture-guide.md](video-capture-guide.md).
4. Rebuild the prior world from robot-camera footage instead of phone footage, so the
   live frames match the model's camera. This is the real fix and it costs one 30-minute
   offline run.

## Ports, for reference

| Port | Path | Use |
| --- | --- | --- |
| 80 | `/cmd?l=&r=` | track speeds, -4095..4095 |
| 80 | `/stop` | stop now |
| 80 | `/status` | JSON: id, uptime, battery, command age |
| 81 | `/stream` | MJPEG multipart |

Movement is a continuously refreshed **intent**, not an event: the robot's 500 ms
watchdog stops the motors unless the command is resent. `BotLink` does that for you.
