# Crab Team N — Architecture

> Source: team chat 2026-09-19. Design intent is the team's. This document structures it; it does not add goals.

## Premise

- Define a small rule set. Derive every behaviour from it.
- Four operators and six predicates generate the whole demo.
- Collaboration is not written. It is what happens when one crab cannot satisfy a precondition alone.
- The name scales with the idea: N crabs, because allocation is over a tree, not over robots.

## Primitives

### Entities

| Entity | State |
| --- | --- |
| PLACE | UNKNOWN, FRONTIER, EXPLORED |
| EDGE | UNKNOWN, REACHABLE, BLOCKED(cause), IMPASSABLE |
| THING | position, class, weight |
| AGENT | position, capabilities, current task |

- `cause` is one of BLOCK, WALL, HOLE, UNSTABLE.
- BLOCK is removable. HOLE is not. The difference decides collaborate against route around.

### Facts

- `at(agent, place)`
- `reachable(place)`
- `blocked(edge, cause)`
- `holds(thing, place)`
- `surveyed(place)`
- `capable(agent, capability)`

### Operators

Each operator states preconditions and effects. Nothing else is needed.

| Operator | Preconditions | Effects |
| --- | --- | --- |
| `GOTO(a, p)` | a path of REACHABLE edges exists to p | `at(a, p)` |
| `EXPLORE(a, p)` | `at(a, p)` | `surveyed(p)`, new PLACEs and EDGEs discovered |
| `PUSH(a, t, e)` | `at(a, near(t))`, `capable(a, PUSH)`, agent count >= `weight(t)` | `not blocked(e)` |

- How `PUSH` chooses where to shove a block, and why it measures rather than predicts, is in [push-planning.md](push-planning.md).
| `OBSERVE(a, t)` | t is in view from `at(a)` | facts about t |

- Add a capability to add an ant type. No new architecture is required.
- `capable(a, DIG)` with a `DIG` operator handles HOLE. This is the stretch path.

## The solver

```
solve(goal) -> SOLVED | BLOCKED(reason) | UNKNOWN(goal)

  if holds(goal):
      return SOLVED

  ops = operators whose effects include goal          # OR node
  if ops is empty:
      return UNKNOWN(goal)                            # nothing known achieves it

  for op in ops ordered by promise:
      unmet = preconditions of op that do not hold    # AND node
      results = [solve(s) for s in unmet]
      if all results are SOLVED:
          return plan(op, results)

  return BLOCKED(best reason among results)
```

- Three outcomes, not two. `UNKNOWN` is what makes exploration fall out of the same mechanism.
- `SOLVED` produces a plan. `BLOCKED` produces a task for another crab. `UNKNOWN` produces an EXPLORE task.
- The tree is the coordination substrate. Crabs do not talk to each other. They read and write the tree.

### Recursion example

```
survey(R2)                                    goal
  requires reachable(R2)                      unmet
    GOTO(a, R2) requires path via e2           OR: only operator
      e2 is BLOCKED(BLOCK)                     unmet
        PUSH(b, block, e2)                     OR: removes the blocker
          requires at(b, near(block))          AND: solvable by GOTO
          requires capable(b, PUSH)            AND: true
          requires agents >= weight            AND: true at 1
        -> SOLVED
```

- Nothing in this trace is scripted. Discovery of the block creates the task.
- If `cause` were HOLE, `PUSH` would not apply, no operator would match, and the result is `BLOCKED`. The parent OR node then tries an alternate route. One mechanism covers both outcomes.

## Allocation

Settled rules. No preemption. No global contention.

```
1. Discovery implies ownership.
      A branch is found by a robot standing there. That robot is the
      nearest robot by construction. It continues onto the best fork.
2. Forks the finder did not take enter the pool.
3. A free bot drains the pool: deepest branch first.
      Ties break to the lower serial number.
4. A latched bot holds its goal until a release fires:
      resolved | invalidated | execution failed | timed out | robot lost
```

- Assignment is latched, not re-evaluated. A latched bot cannot thrash because nothing re-ranks it.
- Coverage is a guarantee, not a probability. Every open branch is queued. It gets a bot when one frees.
- Depth preference applies only to surplus bots. It never starves coverage.
- Serial numbers make assignment deterministic. The same world state gives the same assignment. This matters when the demo is run twice.
- Below about ten branches, greedy assignment is optimal enough. Do not add an assignment algorithm.

### Why no preemption is needed

- A closer branch cannot appear without a robot present to find it.
- That robot owns it immediately. No other robot is better placed.
- A robot deeper in the tree keeps its task. If the task fails or times out, the release fires and it re-enters the pool.

## World model

The world model is a cache. The control loop reads it. Perception writes it. The reader never blocks on the writer.

### Layers, one writer each

| Layer | Writer | Rate | Stale tolerated |
| --- | --- | --- | --- |
| Pose | Laptop, marker detect | 15 Hz | No. Never stale. |
| Dynamic objects: blocks, robots | Laptop, colour detect | 15 Hz | No. |
| Static geometry: walls, corridor shape | Cloud GPU, 3D reconstruction | When it arrives | Yes. |

- One writer per layer. Writers never conflict, so no timestamp arbitration is needed.
- A cloud result computed from old frames cannot regress a newer local observation, because it cannot write the dynamic layer.
- Cloud finds the shape of the room. The laptop tracks what moves in it.

### Why a stale map is safe here

- Marker pose is stateless. Each frame yields an absolute pose from one `solvePnP`. Cost is 1 to 3 ms.
- Pose does not read the map. A stale map cannot produce a wrong pose.
- This property depends on the marker decision. If pose came from SLAM, pose would derive from the map, and a stale map would drive a robot into a wall.

### Degradation ladder

```
full map -> partial map -> no map -> pure reactive (already built)
```

- Every rung is a working system. Losing the cloud drops one rung. Nothing stops.

## Latency budget

- Bandwidth and latency are different problems. Unlimited data fixes throughput only.
- Traffic is small: QVGA JPEG is about 10 KB. Three robots at 15 fps is about 3.6 Mbps. Reconstruction needs 1 to 2 fps, about 60 KB/s.

| Loop | Budget | Cloud round trip | Verdict |
| --- | --- | --- | --- |
| Motion control, 15 Hz | 66 ms | 100 to 250 ms typical, worse tail | Never put the cloud here |
| Reconstruction, 1 Hz | 1000 ms | 100 to 250 ms | Safe |

- A robot at 0.3 m/s with 200 ms latency steers on information from 6 cm ago. Jitter makes the lag unpredictable. This oscillates.
- Marker pose stays local always. A 2 ms job must not become a 200 ms job.

### Two networks

- Robots on a dedicated local router. No internet. No contention.
- Cloud GPU over a phone tether. The laptop runs both interfaces at once.
- Venue Wi-Fi congestion cannot reach robot control.

## Where a language model belongs

Two uses, both OUTSIDE the control loop, both additive. Neither can stop a robot if the API is slow or down; the worst case is a missing caption.

| Use | When it runs | What it does |
| --- | --- | --- |
| **Goal setting** | Once, at mission start | Turns an operator sentence into a goal spec the solver consumes. "Find everything in the far room and clear anything in the way" becomes `survey(R2)` plus a permission to clear removable causes. |
| **Live narrator** | Off the event log, asynchronously | Turns `assign 2: GOTO(2, CS) -- pool CS d1` into "crab 1 found the corridor blocked, so crab 2 was sent to check the south passage instead of waiting." |

### Why it must not go inside the loop

- **The determinism is the claim.** The solver derives collaboration from an unsatisfied precondition. Nobody wrote "call for help". Put a model in that path and the claim becomes unprovable.
- **Latency.** The motion loop has a 66 ms budget at 15 Hz. An API round trip is 300 to 800 ms. It does not fit.
- **Reproducibility.** The demo is run twice under judging. Deterministic means identical.
- **Debuggability.** A precondition checker names the condition that failed. A model does not.

### Not built

Neither exists yet. The solver takes goals from code today. Both are additive and can be built last.

## Motion and waypoints

- Full rules in `motion-and-waypoints.md`. Summary:
- Robot is a rectangle and collides as one. Planning margin 0.17 m, from the chassis diagonal.
- Unknown space is assumed open. Driving the route is the act of exploring.
- Routes prefer the centre of a gap. A 30° field of view sees nothing against a wall.
- Waypoints are a Bezier spline. The end tangent is clamped to the arrival direction. Handles shrink until the curve holds the margin.
- The box moves only when a robot drives into it.
- A placement counts only if it vacates the gap the box now denies. Rank by the whole plan, not by the world after the push.

## Layer stack

| Layer | Owns | Degenerate version (build this first) |
| --- | --- | --- |
| L3 mission | Goal set, biodiversity value | One hardcoded goal: survey everything |
| L2 planner | The solver and allocation | A fixed list of tasks |
| L1 skills | GOTO, EXPLORE, PUSH, OBSERVE | Each skill hardcoded, no planning inside |
| L0 reactive | Steering, stall detection, timeout stop | Built already |
| — world model | Places, edges, facts, agent state | A dict written by observations |

- The planner decides what. The skill decides how. No planning inside a skill. No driving inside the planner.
- The state controller interrupts the motion loop, it does not only set targets. A skill needs a cancel path, not just a setpoint. Retrofitting this is painful; design it in now.
- Waypoints are computed on the laptop. The robot receives track speeds only.
- The robot cannot accept a waypoint. It does not know where it is. Pose is detected on the laptop.
- A planner fault cannot drive a robot into a wall. L0 still owns the timeout stop.

## Build order

Rule from the team: work the fallbacks first, then raise the abstraction. Never break a working layer.

- **Design the top. Build the bottom.** Define every interface now so no layer needs a rewrite later.
- Fix the interfaces first. Swap the implementations upward.
- Refactor L0 now to accept a task rather than own its goals. This is the one early change that prevents a rewrite.
- Start with a hardcoded task list at L2. The full stack then runs end to end from the first hour.
- Replace the task list with the solver once the world model is trustworthy.

| Step | Replace | With | Demo still works |
| --- | --- | --- | --- |
| 1 | Reactive goals | Task interface | Yes |
| 2 | Nothing | World model written from marker pose | Yes |
| 3 | Hardcoded task list | Precondition checker | Yes |
| 4 | Fixed assignment | Promise-weighted allocation | Yes |
| 5 | Single goal | Goal set with biodiversity value | Yes |

## Guardrails

- Keep operators at or under five. Keep the graph under about ten places.
- Do not install a planning library. A precondition checker over a small graph is about 150 lines.
- The planner is only as good as the world model. The world model depends on marker pose. Pose is the real risk, not the planner.
- Failure mode to avoid: a correct planner attached to robots that cannot drive straight.
- If the solver takes longer than one frame to run, the graph is too big.

## Open points

- Arena shape and place count.
- Biodiversity score definition, which sets `value` in `promise`.
- `weight(t)` for the block: one crab or two.
- What the second crab reaches after the path clears.
