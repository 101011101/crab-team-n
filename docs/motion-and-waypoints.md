# Motion and waypoints

Rules the path layer obeys. Each rule exists because its absence produced a measured failure in `viewers/sb_sim.js`.

## Chassis

- Robot is a rectangle, 0.20 m long by 0.15 m wide. It collides as one.
- Circumscribed radius `BOT_RC` = 0.125 m.
- Planning margin `PLAN_PAD` = `BOT_RC` + 0.045 = 0.17 m. No planned cell is nearer a wall than this.
- Arena gaps are 0.90 m. This leaves a 0.56 m planning band. Two robots pass.
- Collision tests: rectangle/rectangle by separating axis; rectangle/wall by Liang-Barsky clip in the rectangle frame. Both O(1).
- Sampling a wall segment every 3 cm instead cost 75 million distance calls per mission and ran the simulation at one third of real time.

## Field

- Occupancy grid, 5 cm cells, states UNKNOWN / FREE / BLOCKED.
- Wavefront is Dijkstra from the goal, 8-connected, binary heap, settled flag.
- A plain array queue re-pushes a cell on every relaxation. With diagonal moves it grows without bound and throws `Invalid array length`.
- Unknown is assumed open. Driving the route is therefore the act of exploring.
- Belief barriers are thickened to `PLAN_PAD` when a wall face is seen. A robot knows its own width, so seeing a wall implies its whole margin is closed.
- Without thickening the belief barrier is 1 cell and the margin is 3.4 cells. The optimistic field threads through the unobserved remainder. A robot in a sealed dead end keeps believing a route exists.
- A robot inside the margin has no field value. Read the field by walking outward over real floor, never by radius. A radius search reads cells through walls.

## Centre seeking

- Camera field of view is 30°. A robot with its nose to a wall sees almost nothing.
- Clearance grid: two-pass chamfer transform, metres to the nearest solid cell.
- Route cost is multiplied by `1 + 3.2 * (0.26 − clearance) / 0.26`, floored at 1.
- Hugging a wall is made expensive, never forbidden. A narrow gap stays usable.
- Applies to route SHAPE only. Every distance the robot reasons or reports with stays in true metres.
- Measured: clearance peaks at 0.30 m in the centre of a 0.90 m gap.

## Waypoints

- Two goal points: where the robot is, where it must arrive.
- Grid path is shortened by line-of-sight, then splined. One cubic Bezier per pair of key points.
- Start tangent is the robot's present heading. No jerk on the first tick.
- End tangent is CLAMPED to the direction the robot must face on arrival.
- Handles are scaled from 1.5 down to 0.12 until every sample of the cubic holds `PLAN_PAD` from all walls.
- If no scale clears, that leg falls back to the straight segment. The straight segment is known clear, because line-of-sight produced it.
- Do not route to a point set back along the approach axis. It produces hairpins. The curve does that job without the detour.

## Push

- The box moves only when the robot drives INTO it. Test: dot(motion this tick, vector to box centre) > 0.
- Contact alone moves nothing. Side-on contact moves nothing.
- If the box cannot move, neither can the robot.
- Rotation in place stays legal when translation is blocked. This frees a wedged corner.

## The field of placements

- Every movable thing carries a field of placements it could occupy. Score each placement directly. Do NOT re-run the wavefront per placement.
- Re-flooding for each of 32 placements cost 23.7 ms a call. A robot stood still waiting for it.
- Vacating is a geometric test against the gap cells. The leg after the shove is one lookup in the box-free field. Room on landing is one lookup in the clearance grid.
- Cost after: 4.3 ms. Same decisions.

## Certain and maybe

- GREEN: the robot has seen every cell the box would sweep through. It knows the box can go there.
- YELLOW: some swept cell is unseen. A wall could be hiding behind the box.
- A box that has been pushed along a line proves that line is clear. Those placements turn green.
- Prefer green. Take a yellow placement only when no green placement clears the gap.
- Among equals prefer room on landing, then price.

## Choosing a placement

- Sample placements along each of 4 push axes, 0.12 m steps, stopping at the first that hits a wall.
- A placement counts only if it VACATES every cell of the gap the box now denies that is otherwise open floor.
- Scoring a placement by field cost alone fails. With unknown space assumed open the route detours around the box through cells nobody has looked at, every placement scores alike, and the shortest useless push wins.
- Scoring by clearance from the box-free route also fails. Pushing the box forward keeps it near that route, because the route runs the same way.
- Rank surviving placements by the WHOLE plan: walk to the pushing position + shove + travel on.
- Ranking by the world after the push alone selects pushes that need the robot to stand on the far side of the doorway the box is sealing.
- Reachability of a pushing position is judged on ground actually seen, never on assumed-open space.
- Pushing position stands `half + BOT_RC + 0.02 + 0.06` from the box centre. Flush against the box pad leaves 5 mm; one nudge from a passing robot then discards every placement.
- Overshoot the chosen placement by 0.16 m. Stopping at the theoretical minimum leaves the gap shut after a centimetre of slip.
- Commit. Re-deriving the target each planning cycle makes a robot nudge the box a few centimetres and back off to reconsider.

## Holding the axis

- On the final leg, steer by lateral offset from the push axis, not by heading to the end point.
- Chasing the end point alone lets the robot creep sideways and walk the box off line.
- Measured drift, box pushed 0.90 m: 0.06 m before, 0.005 m after.

## Recovery

- Progress watchdog: no 0.14 m of travel in 3.5 s triggers reverse and turn for 1.1 s.
- Counting consecutive blocked ticks does not work. Two robots jostling never sit still long enough to trip it.
- A robot inside the margin backs out first, then plans.
- Count consecutive wall contacts. Above 5, reverse for 0.55 s and turn the way the wall repulsion points.
- Driving at a wall again never produces a better heading. Give ground, then approach at an angle.
- Do NOT skip a waypoint because the heading error is large. It looks like it breaks an orbit. It also throws away the one waypoint a lone robot needs to pass the doorway it just opened. Measured: 10 of 10 runs reach with no skipping, 9 of 10 with it.

## Goal board

- Three goals can be open at once. Every open goal must end each planning cycle with an owner.

| Goal | Open when | Owners |
| --- | --- | --- |
| UNBLOCK | the reward is out of reach, or the box costs more than moving it | 1 |
| EXPLORE | frontier remains | many |
| REACH | always | 1 |

- Ownership is LATCHED. An owner keeps its goal until the goal closes or the robot dies.
- A robot that has begun a shove is never released. Without this the job changed hands 14 to 26 times per run and the box never moved.
- UNBLOCK opens only if the best push PLAN beats the owner's own route. Judging against the team minimum fails: with unknown space assumed open, the team minimum is whatever the most optimistic robot believes, so the trade never looks worth taking.
- Opening UNBLOCK merely because the box sits on the SHORT route makes a robot shove a box nobody needed moved.

## Staging

- A robot holding REACH while the reward is walled off does not idle and does not scout.
- It drives as far along the route that WOULD exist once the box moves as it can presently stand, and waits.
- Staging point: descend the box-free field, walk it while the cell is still reachable in the CURRENT world, then back off. Setback grows per robot so they queue instead of piling up.
- When the owner of UNBLOCK clears the gap, the staged robot is already at the door.

## Keeping out of each other's way

- Lateral steering bias away from any robot within 0.55 m, the same term used for walls.
- Right of way by goal rank: UNBLOCK 3, REACH 2, STAGE 1, EXPLORE 0. Equal ranks break by serial.
- The robot clearing the gap must never be the one that gives ground. Otherwise the whole team waits on a robot that yielded to a scout.
- Two robots of equal rank must not both yield. Breaking by serial prevents that deadlock.
- Waiting robots must wait in DIFFERENT places. Backing each one further along the same line stacks them nose to tail in a 0.56 m band.
- A staging point at least 0.52 m from every other staging point. If the queue head is still crowded, sidestep perpendicular to the route in 0.30 m steps.
- Measured, before: 444 blocked ticks with 3 robots, 1243 with 5. After: 0 in every run. Closest approach 0.170 m to 0.196 m.

## Verified, goal board

- 10 runs, 1 to 5 robots, other way in open and sealed. All reach the reward.
- Other way in OPEN: detour costs 0.75 m (4.28 m against 3.53 m). No robot moves the box.
- Other way in SEALED: reward unreachable. Box pushed 0.86 to 0.89 m, every run.
- UNBLOCK changed hands 0 or 1 times per run.
- Staging observed in 4 of 5 sealed runs.

## Verified

- 10 runs, 1 to 5 robots, gap open and gap sealed. All reach the reward.
- Gap open: box never moved. Gap sealed: box pushed 0.87 to 0.93 m, every run.
- Same code decides both. Nothing selects the behaviour but whether a route exists.
- Cost 0.3 to 1.1 ms per frame.

## State lattice

- Plan over poses (x, y, heading), not over cells. 16 heading buckets. Visited set keyed on (cell, bucket).
- Six primitives from each pose: arc left, straight, arc right, forward and reverse. Arc radius `LAT_R` = 0.40 m (turning radius 0.34 m plus tracking margin), arc = one bucket (22.5°), straight = 0.15 m.
- Cost = length × centre-seeking weight. Reverse ×2.5. Gear change +0.30 m.
- Heuristic = max(wavefront from the target, `LAT_R` × heading still to turn) × 1.25.
- Without the heading term the search reached the pushing position 299 times facing the wrong way and ran out of budget at 4001 nodes. With it the same search closes in a few hundred.
- Budget 6000 nodes. Cost 0.2 to 1 ms mean, 8 to 15 ms at the budget.
- Measured cause of the 40 to 60 % reversing: a cell path laid directly behind the crab. The controller reversed along all of it at half speed with the heading error pinned at 180°, for 30 s. Nothing in the controller turns a crab round.
- Each waypoint carries its gear. The controller guesses the gear from heading error only on a path with no gear.
- The first move snaps the heading onto the lattice of the required end heading. Landing 11° off the push axis pushed the box 0.19 m sideways over a 0.90 m shove.
- If the constrained search fails, search again with no end heading. Never fall back to a cell path behind the crab.
- The spline stays as the last fallback. It subdivides once with a perpendicular offset. It returns null if no drivable curve exists. It never emits an unconstrained leg.
- Measured on the old code: the unconstrained straight-leg fallback fired on 44 to 55 % of legs, 90 % of them for curvature. The 30 s reverse traps were the separate first-leg-behind-the-crab case, 0 to 22 per run.
- Do NOT keep a path across planning cycles without re-checking it. A route laid before the box was seen ran through the box and the crab shoved it 2.26 m in the open arena.

## Stuck test

- Stuck = the crab's cell is a SEEN wall margin. Walls only.
- Judging it with the box pad counted every pushing crab as stuck: 556 of 557 triggers in a 3-crab sealed run were the box.
- Escaping a margin never pre-empts a shove in progress.
- The escape target is 0.18 m out, never the nearest open cell. The nearest cell sat inside the arrival tolerance: the crab arrived without moving, its centre still in the margin, and was stuck again 0.3 s later, for 150 s.
- Wall repulsion gain 3.0, above the crab-to-crab gain 2.2. A jostle in a 0.56 m band pushed a crab into the margin.
- Rotating in place is a move. Turning without the collision test swung an evading crab's corner into a neighbour: 55 to 266 overlap ticks a run. Now 0.
- Wall contact slides: keep the tangential part of the motion. Measured: the lattice keeps a full margin, so slides are 0 in every passing run and the wall-hit evade never fires.

## Goal board, corrections

- Give the job back when the UNBLOCK election releases its holder. Nulling it made a lone crab alternate REACH and EXPLORE every cycle and shuttle between two routes.
- A cleared shove hands the job back to the board. Falling through to a new shove skipped the "does it pay" test: the box went 3.31 m instead of 0.87 m.
- Placements are valid only for the box position they were scored at. A stale set sent the crab to a pushing position computed before the box moved.
- Shove completion is measured along the push axis. A radius round the target read a 3 cm sideways drift as "not there yet".
- The box is planned round with `PLAN_PAD`, the same margin as a wall. At `BOT_RC` + 0.02 a one-cell corridor beside a half-pushed box made the reward reachable in the field, UNBLOCK never re-opened, and a crab spent 100 s trying to thread it.
- Path commitment (re-plan only on finished, invalidated, job change, stall, 5 s safety) was measured and NOT adopted: 8 of 10, sealed runs 139 to 179 s, 36 to 556 plans per crab. The 0.45 s cycle stays.

## Reward belief

- `rewardBelief`: 1 = the cell can still hold the reward. Seed: disc of 1.2 m round the true reward, minus wall cells.
- Seeing a cell of the blob and not the star clears that cell. Seeing the star sets `rewardFound` and collapses the blob to one cell.
- Target while not found = centroid of the surviving cells, snapped to the nearest surviving cell believed passable. One seed, one flood. A nearest-to-each-crab target needs one flood per crab (100 ms).

## Field of view

- Cone of `FOV_DEG` about the heading, range `SIGHT`, occluded by walls. The crab's own cell is always known.
- A scout within 0.6 m of its frontier arrives facing the unknown neighbours.
- Facing the unknown at any range planned a turn-round at every frontier: reversing 26 to 42 % in the open arena.
- Measured: 30° passes 9 of 10 (4 crabs sealed fails: the pushing position is judged on seen floor and stays unseen). 45° passes 10 of 10. 60° and 90° each fail one different cell. `FOV_DEG` = 45.

## Placement scoring

- Score = total − 1.6 × freedom + 1.2 × unseen fraction of the sweep + 0.9 × |push · travel direction|.
- Travel direction = downhill in the box-free field at the box.
- Sealed arena, before and after: +x through the door, 1.00 m, total 3.77 m, sweep 100 % unseen, along 0.71. Only the +x placements vacate the throat; ±y hits the door posts. The perpendicular term cannot change the choice here.
- Count the sweep at stride 2. Counting every cell cost 13 ms a call.

## Verified, lattice

- 10 runs, 1 to 5 crabs, open and sealed. All reach inside 180 s. Overlap 0 in every run.
- Open: box 0.00 m. Sealed: box 0.97 to 1.77 m.
- Reversing, sealed: 40 to 61 % before, 19 to 37 % after. Open: 0 to 8 % before, 0 to 39 % after (planned turn-rounds at frontiers, cone).
- UNSTICK, 3 crabs sealed: 4.6 % of ticks before, 0 after.
- Frame 0.4 to 1.2 ms mean. Placement search 4 to 8 ms, 11 to 22 ms under load.
