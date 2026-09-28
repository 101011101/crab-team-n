# Live telemetry contract

The one interface between the Python fleet (`crab-team-n/live/`) and the Next.js
dashboard (`dashboard/`). Both sides are built against this file. Change it here first.

    GET http://<laptop>:8787/api/live/state      poll at 5 Hz, cheap, no streaming
    GET http://<laptop>:8787/api/live/health     liveness only

CORS: `Access-Control-Allow-Origin: *`. The dashboard runs on :3000 and the bridge on
:8787, so without it every fetch fails silently in the browser.

## `/api/live/state`

```json
{
  "t": 123.4,
  "source": "crab-team-n/live",
  "units": { "metres_per_unit": 1.0, "calibrated": false },
  "frame": { "originX": -9.2, "originY": -7.1, "cell": 0.1554, "cols": 119, "rows": 91 },
  "grid": "<base64 of cols*rows bytes, row-major, row 0 = originY>",
  "robots": [
    { "id": "scout-1", "x": 1.2, "y": -0.4, "theta": 0.78,
      "status": "driving", "mode": "push", "confidence": 0.82,
      "online": true, "frames": 412, "camUrl": "http://192.168.1.51:81/stream" }
  ],
  "boxes": [ { "id": "box2", "x": -3.1, "y": 2.2, "w": 0.62, "h": 0.47 } ],
  "goal": [ -5.5, 1.96 ],
  "mapService": { "rounds": 3, "submitted": 18, "registered": 11,
                  "rate": 0.61, "lastRoundS": 8.3, "error": "" },
  "events": [ { "t": 46.5, "text": "crab1 route OPEN after moving box30" } ]
}
```

### Grid byte values

| Byte | Meaning | Dashboard `CELL_*` |
| --- | --- | --- |
| 0 | UNKNOWN — not seen. **Not** the same as blocked. | `CELL_UNKNOWN` (0) |
| 1 | FREE — floor observed here | `CELL_FREE` (1) |
| 2 | STATIC — obstacle, cannot be moved | `CELL_OCCUPIED` (2) |
| 3 | BOX — obstacle, **pushable** | `CELL_OCCUPIED` (2) |

The dashboard has no fourth state, so 3 collapses to `CELL_OCCUPIED`. Pushable cells
are still distinguishable through `boxes[]`, which carries their footprints.

### UNITS — read this before drawing anything

`calibrated` is **not a constant**. It follows the map the fleet is driving on, and
both cases happen in normal use:

| Map | `calibrated` | Why |
| --- | --- | --- |
| `live/arena_map.py` (room frame) | **true** | Built from `pipeline/common/constants.py`, whose scale comes from the printed 0.15 m tag edge. A measured metre. |
| `pipeline/stage4_map2d.py` (SfM) | **false** | Monocular SfM fixes a scene only up to a 7-DoF similarity. There is no metre in it at all. |

So:

- `calibrated: true` — the numbers ARE metres. Print `m`.
- `calibrated: false` — the numbers are MODEL UNITS. Do not print `m`; say "model
  units" or use no unit at all.

Getting this wrong is not a cosmetic bug. It is a dashboard confidently reporting a
distance in metres that is not metres. Read the flag; never assume either case.

### Field notes

- `theta` — radians, 0 along +x, CCW positive. Same convention as the dashboard.
- `y` — the map's second ground axis. Stage 4 calls it `z`; it is the same axis.
- `status` — `idle` | `turning` | `driving` | `waiting` | `pushing` | `touching`.
  These are the dashboard's `RobotStatus` values exactly; do not invent new ones.
- `mode` — the control loop's task: `drive` | `push` | `explore` | `idle` |
  `stale-pose` | `offline`. Free-form, shown as text.
- `confidence` — 1.0 just after an SfM fix, decaying with distance dead-reckoned
  since. **0.0 means the pose is a guess.** Show it; a robot drawn confidently at a
  position nobody measured is the worst thing this dashboard can do.
- `camUrl` — MJPEG straight from the robot. The browser fetches it directly, not
  through the bridge.
- `events` — newest last, at most the last 50.
- `goal` — may be `null` before the first map arrives.
- `robots` — always every configured robot, including offline ones.

## When there is no map yet

`frame`, `grid`, `boxes` and `goal` are `null`. `robots` still lists every configured
robot with `online` and `camUrl`, so the dashboard can show camera feeds and
connectivity before any reconstruction exists. The dashboard must render this state
rather than treat it as an error — it is the normal first 30 seconds of every run.
