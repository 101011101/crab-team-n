# Moving a block: potentials, not prediction

> Design for the PUSH operator. Source: team chat. Everything here is deterministic geometry; no model is involved.

## The idea

Do not predict where a shoved block ends up. Instead:

1. Build a **field of possible placements** — every position the block could legally occupy.
2. **Score** them and pick a desired one.
3. Derive the **push vector** that gets the block from where it is to where it should be.
4. Turn that vector into **waypoints** for a robot.
5. Emit those as a **goal**, and let the allocator send the nearest free robot.

The block moves under closed-loop correction, not open-loop prediction. See "Why not predict" below.

## 1. The field of possible placements

- Sample the floor on a grid, 5 cm is enough.
- A cell is **feasible** if the block's footprint placed there overlaps no wall and no other block.
- A cell is **reachable for the block** if a straight push from the current position gets there without the block crossing a wall. A block is not a robot: it does not turn corners on its own.
- The result is a mask, cheap to recompute whenever the world model changes.

## 2. Scoring a placement

Score every feasible cell and take the best. Terms, in order of weight:

| Term | Why |
| --- | --- |
| **Does it clear the blocked edge** | The whole point. A placement that leaves the edge blocked scores zero. |
| **Is the approach reachable** | The robot has to get behind the block. If the approach point is unreachable the placement is useless however good it looks. |
| **Push distance** | Shorter is better. Every centimetre is a chance to drift. |
| **Clearance at the destination** | Prefer somewhere the block is out of the way and will not become the next obstacle. |
| **Perpendicularity to the corridor** | A block shoved sideways out of a doorway stays out. A block shoved along a corridor is still in the corridor. |

That last one is the substantive one: **prefer pushing a block sideways out of the gap over pushing it down the gap.**

## 3. The push vector

- `dir = normalise(target - block_centre)`
- The robot must be **behind the block along that direction**, so the contact point is on the far side from the target.
- Approach point: `block_centre - (block_half_extent_along_dir + chassis_radius + margin) * dir`
- That distance is exactly the `push_standoff` already in `navigator.py`. It is not a constant: it depends on the blocker's size, which is why it is computed rather than typed.

## 4. Waypoints

Three, in order:

1. **Stage** — a point clear of the block, on the push axis, far enough back to turn in place. Roughly `approach - 0.35 * dir`.
2. **Contact** — the approach point. The robot arrives facing along `dir`.
3. **Follow through** — `target - standoff * dir`. Driving to this waypoint is what actually moves the block, because the block is in the way.

The robot never plans a path *through* the block. It plans to a point beyond it and the block comes along.

## 5. It becomes a goal

- The push emits as an ordinary task, so the existing allocator handles it: nearest free robot, latched, released on the usual conditions.
- Nothing new is needed in the solver. `clear(edge)` already has `PUSH` as its operator; this is how `PUSH` chooses its geometry.

## Why not predict where the block goes

- Where a pushed box ends up depends on the friction distribution across its contact patch. That is invisible, uneven, and changes as it slides. There is real theory for quasi-static pushing, and the practical finding is consistent: **open-loop pushing drifts and no model fixes it.**
- So do not model it. **Put a marker on the block.** The same detector and solver that localise a robot give the block's pose at camera rate, for free.
- Push in **short increments**, 10 to 15 cm. Re-measure. Correct the heading. Repeat until the edge reads clear.
- The win condition is then measured, not assumed: the edge becomes REACHABLE when the block has actually moved, not when the robot believes it has.

A vision-language model is the wrong tool here. It cannot produce metric predictions, it is slow, and it would sit inside the control loop, which is the one place nothing uncertain may live.

## What is easy and what is not

| | |
| --- | --- |
| Feasible-placement field | **Easy.** A grid and a collision test. Milliseconds. |
| Scoring and choosing | **Easy.** A weighted sum over the mask. |
| Push waypoints | **Easy.** The geometry above, already half written. |
| Getting the block to arrive exactly | **Hard, and avoided** by closing the loop on a marker instead of predicting. |
| Two robots pushing one block together | **Hard.** Needs synchronised contact and matched speeds. Out of scope unless the demo needs it. |

## Simulator

The simulated block must **drift slightly when pushed** — a small heading error proportional to the offset between the contact point and the block's centre. Without it the sim succeeds open-loop, the control loop never has to correct, and none of this transfers to hardware.

## Not built

The field, the scoring and the marker-based correction are all unbuilt. `navigator.py` currently pushes along a fixed axis with a computed standoff, which is the degenerate one-direction case of the above.
