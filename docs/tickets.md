# Tickets — proof components

> Goal: prove the three risky parts in isolation before integrating. Each ticket ships a runnable demo and a test. Interfaces are fixed here so the parts meet cleanly.

## Dependency order

```
T1 motion controller  ──┐
                        ├─► T3 simulator ──► integrated demo
T2 world + solver     ──┘

T4 video to 3D        (independent, blocked on a supplied video)
```

- T1 and T2 share no code. Build in parallel.
- T3 consumes both interfaces below. It must not redefine them.
- T4 touches nothing else.

## Fixed interfaces

Do not change these without updating this file first.

```python
# central/geometry.py
@dataclass(frozen=True)
class Pose:
    x: float          # metres, world frame
    y: float
    theta: float      # radians, 0 points along +x, CCW positive

@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    heading: float | None = None   # None means arrive at any heading
    tolerance_m: float = 0.05
    heading_tolerance_rad: float = 0.15
```

```python
# central/motion.py
class WaypointController:
    def step(self, pose: Pose, wp: Waypoint) -> tuple[int, int]:
        """Return (left, right) track speeds, -4095..4095."""
    @property
    def arrived(self) -> bool: ...
    def reset(self) -> None: ...
```

```python
# central/world.py
class World:
    def place(self, pid: str) -> Place
    def edge(self, a: str, b: str) -> Edge
    def path(self, a: str, b: str) -> list[str] | None   # REACHABLE edges only
    def open_branches(self) -> list[str]                 # FRONTIER places
    def observe(self, fact) -> None                      # only writer of its layer
```

```python
# central/solver.py
Result = Solved(plan: list[Task]) | Blocked(reason) | Unknown(goal)
def solve(world: World, goal: Goal) -> Result

# central/allocator.py
def assign(world: World, agents: list[Agent]) -> dict[int, Task]
```

---

## T1 — Waypoint motion controller

- **Deliverable:** `central/motion.py`, `central/geometry.py`, `central/test_motion.py`
- **Blocked by:** nothing
- **Why:** the robot cannot accept a waypoint. It has no pose. This is the missing layer between planner and radio.

**Behaviour**

- Turn in place when the bearing error exceeds a threshold. Drive with proportional steering when it does not. This avoids wide arcs that overshoot.
- Convert `(v, w)` to track speeds: `l = v + k*w`, `r = v - k*w`, `k` from the measured track width in `config.py`.
- Scale the track pair to clear the stall floor. Never clamp each track separately: during a turn one track is legitimately near zero, and clamping turns a correction into a spin. Reuse the rule already in `behavior.py`.
- Report `arrived` when inside `tolerance_m`, and inside `heading_tolerance_rad` when a heading is given.

**Acceptance**

- Converges from at least 8 start poses spread across headings, including directly behind the waypoint.
- Turn direction is correct for waypoints left and right. Assert the sign, do not eyeball it.
- No oscillation: the pose error decreases monotonically over the final approach.
- A waypoint already satisfied returns `(0, 0)` and `arrived` immediately.
- All tests run with no hardware.

---

## T2 — World model, solver, allocator

- **Deliverable:** `central/world.py`, `central/solver.py`, `central/allocator.py`, `central/test_solver.py`
- **Blocked by:** nothing
- **Why:** this is the architecture in [architecture.md](architecture.md). Prove the recursion and the allocation rules before any robot depends on them.

**Behaviour**

- Implement the entities, facts and four operators from [architecture.md](architecture.md) exactly.
- `solve` returns three outcomes, not two. `Unknown` is what produces an EXPLORE task.
- Allocation rules, in order: discovery implies ownership; unclaimed forks enter the pool; a free bot takes the deepest pooled branch, ties to the lower serial; a latched bot holds until a release fires.
- Release conditions: resolved, invalidated, execution failed, timed out, robot lost.

**Acceptance**

- The corridor trace in [architecture.md](architecture.md) reproduces exactly: `survey(R2)` fails, `reach(R2)` fails, `e2` is BLOCKED, `clear(e2)` is emitted, and after it resolves `survey(R2)` succeeds.
- A HOLE cause yields `Blocked`, not a clear task, and the parent tries an alternate route.
- A latched agent does not change task when a nearer branch appears.
- Two free agents and one pooled branch resolves to the lower serial, deterministically, across repeated runs.
- Solver runs in under one frame time on a ten-place graph.

---

## T3 — 2D simulated world

- **Deliverable:** `central/sim.py`, `central/demo_sim.py`
- **Blocked by:** T1 and T2 interfaces
- **Why:** proves states lead to waypoints lead to motion, with no hardware and no vision.

**Behaviour**

- Differential-drive kinematics: integrate `(l, r)` into a pose. Same units the real robot takes.
- A 2D arena: two rooms, a corridor, a pushable block, coloured species markers.
- Robots discover places by proximity. Discovery writes to the world model.
- Top-down render: arena, robots, headings, current waypoint, per-robot state, task assignment, the goal tree.
- Headless mode writes a text log of every state transition and assignment.

**Acceptance**

- A full scenario runs end to end: explore room 1, find the corridor blocked, emit `clear`, a second robot clears it, room 2 opens, species found.
- The log shows each task emitted with its cause, and each assignment with the robot serial.
- Deterministic under a fixed seed. Two runs produce identical logs.
- No hardware, no network.

---

## T4 — Video to 3D

- **Deliverable:** `tools/video3d/feasibility.py`, `tools/video3d/reconstruct.py`, `tools/video3d/README.md`
- **Blocked by:** a supplied video
- **Why:** the 3D world is the wow factor and nobody has tested a learned depth model on a 320x240 rolling-shutter feed.

**Behaviour**

- `feasibility.py` runs first and needs no model. Per frame it reports blur (variance of Laplacian), texture (ORB keypoint count), trackability (inter-frame match inliers after RANSAC), and parallax (translation against rotation). It prints a verdict on whether reconstruction is worth attempting.
- `reconstruct.py` samples frames, runs Depth Anything V2 Small on MPS, back-projects to a point cloud, and writes a `.ply` plus a rendered preview.
- Keep heavy dependencies in `tools/video3d/requirements.txt`. Do not add torch to `central/`.

**Acceptance**

- `feasibility.py` runs on any supplied video and produces a numeric report plus a verdict.
- `reconstruct.py` produces a viewable point cloud from a phone video.
- The README states measured runtime and what failed.

---

## What the team must supply

| Item | For | Notes |
| --- | --- | --- |
| Phone video of the arena | T4 | 30 to 60 s. Walk slowly. **Translate, do not pan.** Pure rotation reconstructs nothing. |
| Robot camera video, same path | T4 | The real test. Compare against the phone video to separate a camera problem from a code problem. |
| Measured track width | T1 | Wheel centre to wheel centre, in metres. Until then a placeholder is used. |

- **No GPU is required for any ticket.** Depth Anything V2 Small runs on Apple Silicon. COLMAP runs on CPU. The simulator is pure Python.
- A cloud GPU only buys speed and quality at demo time. It is not needed to prove any of this.
