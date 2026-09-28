# The reach field: one mechanism for exploring and for navigating

> Design. Source: team chat. Deterministic; no model involved. Sibling of [push-planning.md](push-planning.md) — that one is a field over where the BLOCK could go, this one is a field over where a ROBOT can go.

## What we have now, and why it is not enough

- The world model is symbolic: places, and edges between them. `world.path()` is a breadth-first search over REACHABLE edges.
- Waypoints come from a hand-authored map of legs. Somebody types the coordinates.
- That works for an arena we designed. It does not work for an arena a robot discovers, and it cannot answer "is there a way through here" about ground nobody has named.

## The field

One occupancy grid over the arena, about 5 cm per cell, three states:

| State | Meaning |
| --- | --- |
| **UNKNOWN** | Nobody has seen it. Not the same as blocked. |
| **FREE** | Observed, and a chassis fits. |
| **BLOCKED** | Observed, and something is there. |

Then one **distance-to-goal field**: a wavefront expanded from the goal across FREE cells only. Each cell holds its distance to the goal, or infinity if no known-free route reaches it.

That single field answers both questions:

- **Can I get to the goal?** The robot's cell has a finite distance. Follow the gradient downhill.
- **If not, where should I look?** Go to the nearest **frontier** — a FREE cell touching an UNKNOWN one. That is where the map can still grow.

Exploration stops being a separate behaviour. It is what the field tells you to do when the goal is unreachable.

## It runs continuously

- Recompute the wavefront whenever the grid changes: a robot observes new ground, or a block moves.
- A full wavefront over a 7 x 2.6 m arena at 5 cm is about 7,000 cells. Sub-millisecond. There is no reason to be lazy about it.
- **This is the point.** The moment a block is shoved aside, the cells it occupied flip to FREE, the wavefront floods through, and a route to the goal exists for every robot at once — including robots that had nothing to do with the push. Nobody has to be told the way is open; the field already knows.

## Waypoints from the field

- Follow the gradient from the robot's cell downhill to the goal.
- That raw path is jagged, so simplify it: keep a point only where the direction changes by more than a few degrees, and drop any point that can be skipped because the straight line between its neighbours is clear.
- The survivors are the waypoints. Same `Waypoint` type, same `Navigator`, same motion controller.
- This replaces the hand-authored legs table. The legs exist because nobody had written this.

## How it meets the solver

The symbolic layer stays. It is what makes collaboration derivable, and the field does not replace it.

| Question | Answered by |
| --- | --- |
| What is the goal, and what must hold first | solver, on places and edges |
| Which robot does it | allocator |
| Is there physically a way through | **reach field** |
| What are the waypoints | **reach field** |
| Where should the block end up | push field |

So `reachable(place)` becomes a query against the field rather than a stored edge state, and `clear(edge)` is emitted when the field says the goal is unreachable but would be reachable with a removable blocker gone. That last test is cheap: recompute the wavefront with the blocker deleted and see if the distance becomes finite.

## Showing it

In the simulator, the field is the most legible thing on screen:

- **Dim the UNKNOWN cells. Saturate them as they are discovered.** Watching the dark recede is watching the robots learn.
- Shade FREE cells by distance to the goal, so the gradient the robots are descending is visible.
- Mark frontier cells. They are where the robots are about to go, and seeing that before they move makes the behaviour readable.
- When a block moves, the flood through the newly opened gap should be visible as it happens.

## Cost

| | |
| --- | --- |
| Grid update from an observation | trivial |
| Wavefront over the whole arena | sub-millisecond at 5 cm |
| Path extraction and simplification | negligible |
| Recompute on every change | fine, and simpler than deciding when to skip it |

A* is the usual name for the single-pair version. A full wavefront from the goal is cheaper here, because every robot reads the same field and only one expansion serves all of them.

## Not built

The grid, the wavefront, frontier selection and path extraction are all unbuilt. `world.path()` over symbolic edges and the hand-authored legs table are what exist today.
