# control — the control loop

Builds what [`docs/reach-field.md`](../docs/reach-field.md) and
[`docs/push-planning.md`](../docs/push-planning.md) specify. Both were marked **"Not
built"**. This is that build, driven by a real map from `pipeline/`.

## The chunks

| File | Does |
| --- | --- |
| `grid.py` | Occupancy grid from a stage-4 map. UNKNOWN / FREE / BLOCKED, plus a pushable bit. |
| `reach_field.py` | One wavefront from the goal. Frontier cells. Gradient descent to waypoints. |
| `push.py` | Where to shove a box, and the three waypoints that do it. |
| `loop.py` | The loop: observe, flood, assign, act. |
| `run_sim.py` | Picks a scenario off the map, runs N robots, writes a trace. |

Reuses `central/`: `Pose`, `Waypoint`, `wrap_angle`. The field replaces the
hand-authored legs table, not the motion controller.

## One tick

1. **OBSERVE** — UNKNOWN cells within sight, with line of sight, become FREE.
2. **FLOOD** — if the grid changed, recompute the wavefront. One expansion serves every
   robot, so this is never per-robot work.
3. **ASSIGN** — a robot with no task gets one:
   - goal reachable → **DRIVE**
   - goal reachable if one box were gone → **PUSH** that box
   - neither → **EXPLORE** the nearest frontier
4. **ACT** — follow the waypoint. Refuse moves into blocked cells.

Exploring is not a separate behaviour. It is what the field says to do when the goal is
unreachable.

## How a blocker is identified

Delete a box, re-flood, ask whether the goal's distance became finite. That is the whole
test, and it is what `clear(edge)` is supposed to mean.

`blocking_boxes()` returns a **list**, least boxed-in first, because a box can gate the
route and still have nowhere to go. The caller tries each until one has a feasible
placement. A blocker with no legal push is not a plan.

## The push

Per the spec, do **not** predict where a pushed box ends up. Where it lands depends on
the friction across its contact patch, which is invisible and changes as it slides. So:

- Push one `INCREMENT` at a time, then re-measure.
- The route is declared open when the field says so, never when the robot believes it
  pushed far enough.
- Standoff is computed from the box's extent along the push axis, not typed as a
  constant.
- Three waypoints: **stage** (far enough back to turn), **contact**, **follow-through**
  (beyond the box). The robot never plans a path *through* the box; it plans to a point
  beyond it and the box comes along.

## Run it

    cd control
    ../central/.venv/bin/python run_sim.py ../results/world3/map2d.json \
        --robots 2 --seconds 400 --trace trace.json --png sim.png

The map is the **template**: any stage-4 map runs unchanged.

`run_sim.py` picks the scenario from the map rather than hand-placing one. It looks for a
goal that is genuinely gated by a carton *that can move* — reachable if the carton were
gone, unreachable while it stands. On the IMG_7373/4/5 map, 645 cells are gated and
`box2` alone gates 84 of them.

## Four bugs worth knowing about, because they will come back

- **No collision check.** The first version had none, so a robot drove into a carton and
  stranded itself inside blocked space, where the field has no finite distance and it can
  never recover. `steer()` now refuses the move.
- **Planning on un-inflated free space.** That treats the robot as a point, plans routes
  that graze a carton corner, and the body clips it. `Grid.drivable()` erodes the free
  mask by the chassis once, and **everything** — planning, wavefront, frontier, scenario
  setup — uses that one mask. When they disagree you get goals no robot can stand on.
- **Livelock pointing at an obstacle.** Refused, but turning toward the waypoint does not
  help: the waypoint is what it is driving at. The heading now biases aside.
- **Oscillating about a waypoint.** A robot turning in place has `v = 0`, so the collision
  counter never fires and it looks busy forever. The stall guard watches distance
  instead. `central/motion.py` watches for the same thing on the real robot.

## What it does not do

- No allocator. Robots pick tasks independently and only avoid pushing the same box.
  `central/allocator.py` is the real thing and is not wired in.
- No symbolic layer. `central/solver.py` and `world.py` are untouched.
- The box moves because the robot drove where it stood. That is a simulated consequence,
  not contact physics.
