"""Push planning — where to shove a box, and the three waypoints that do it.

Implements docs/push-planning.md, which was specified and marked "Not built".

The governing idea from that spec: do NOT predict where a pushed box ends up. Where
it lands depends on the friction distribution across its contact patch, which is
invisible and changes as it slides; open-loop pushing drifts and no model fixes it.
So push in SHORT INCREMENTS, re-measure the box, correct, repeat. The win condition
is measured, never assumed.
"""
import numpy as np

from grid import FREE, UNKNOWN
import reach_field as rf

CHASSIS = 0.09          # model units, robot radius
MARGIN = 0.04           # gap between chassis and box at the contact point
STAGE_BACK = 0.35       # how far behind the contact point to stage, per spec
INCREMENT = 0.13        # one push before re-measuring. The spec says 10-15 cm.


def half_extent(box, d):
    """Box half-width along direction d. The standoff depends on the blocker's
    size, which is why the spec computes it rather than typing a constant."""
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    ax = np.array([c, s]) * box["w"] / 2
    ay = np.array([-s, c]) * box["h"] / 2
    return abs(np.dot(ax, d)) + abs(np.dot(ay, d))


def standoff(box, d):
    return half_extent(box, d) + CHASSIS + MARGIN


def straight_push_clear(g, blocked, start, target, box):
    """A block is not a robot: it does not turn corners. So the straight line from
    here to there must be clear for the BOX, not just for a point."""
    d = target - start
    n = int(np.linalg.norm(d) / g.cell) + 1
    r = max(box["w"], box["h"]) / 2
    rc = max(1, int(r / g.cell))
    for t in np.linspace(0, 1, n * 2 + 1):
        p = start + d * t
        j, i = g.to_cell(*p)
        for dj in range(-rc, rc + 1):
            for di in range(-rc, rc + 1):
                a, b = j + dj, i + di
                if g.inside(a, b) and blocked[a, b]:
                    return False
    return True


def own_cells(g, box, blocked):
    """A box does not block itself. Planning must see the world without it, or a box
    standing in a run of other boxes has no feasible placement in any direction."""
    out = blocked.copy()
    r = max(box["w"], box["h"]) / 2 + g.cell
    j, i = g.to_cell(box["cx"], box["cz"])
    rc = int(np.ceil(r / g.cell))
    for dj in range(-rc, rc + 1):
        for di in range(-rc, rc + 1):
            a, b = j + dj, i + di
            if g.inside(a, b):
                out[a, b] = False
    return out


def free_ratio(g, box, blocked, r=6):
    """How boxed-in is this box? Used to pick one that can actually move."""
    j, i = g.to_cell(box["cx"], box["cz"])
    return _clearance(g, blocked, j, i, r)


def choose_target(g, box, blocked, open_goal, robot_reach, max_push=1.2):
    """Score every feasible placement and take the best.

    Terms, in the spec's order of weight:
      clears the goal cell   a placement that leaves it blocked scores zero
      approach reachable     the robot must be able to get behind the box
      push distance          shorter is better; every unit is a chance to drift
      clearance              do not make the box the next obstacle
      perpendicularity       shove it SIDEWAYS out of the gap, not along it
    """
    blocked = own_cells(g, box, blocked)
    c = np.array([box["cx"], box["cz"]])
    goal = np.array(open_goal, float)
    along = goal - c
    n = np.linalg.norm(along)
    if n < 1e-6:
        return None
    along /= n

    best, best_s = None, -1e9
    for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        d = np.array([np.cos(ang), np.sin(ang)])
        for dist in np.arange(3 * g.cell, max_push, 2 * g.cell):
            t = c + d * dist
            j, i = g.to_cell(*t)
            if not g.inside(j, i) or blocked[j, i]:
                continue
            if not straight_push_clear(g, blocked, c, t, box):
                continue
            # The robot stands BEHIND the box relative to the target, not behind
            # the target. Both the contact point and the staging point it turns in
            # must exist, be clear, and be reachable -- the spec is explicit that a
            # placement with an unreachable approach is useless however good it looks.
            so = standoff(box, d)
            contact = c - so * d
            stage = c - (so + STAGE_BACK) * d
            bad = False
            for q in (contact, stage):
                qj, qi = g.to_cell(*q)
                if not g.inside(qj, qi) or blocked[qj, qi] \
                   or not np.isfinite(robot_reach[qj, qi]):
                    bad = True
                    break
            if bad:
                continue
            clear = _clearance(g, blocked, j, i)
            perp = 1.0 - abs(float(np.dot(d, along)))   # sideways beats along
            s = 3.0 * perp + 1.5 * clear - 1.0 * (dist / max_push)
            if s > best_s:
                best, best_s = (t, d, dist), s
    return best


def _clearance(g, blocked, j, i, r=3):
    free = tot = 0
    for dj in range(-r, r + 1):
        for di in range(-r, r + 1):
            a, b = j + dj, i + di
            if g.inside(a, b):
                tot += 1
                free += not blocked[a, b]
    return free / max(tot, 1)


def waypoints(box, target, d):
    """Three, in order: stage, contact, follow-through.

    The robot never plans a path THROUGH the box. It plans to a point beyond it and
    the box comes along.
    """
    c = np.array([box["cx"], box["cz"]])
    so = standoff(box, d)
    contact = c - so * d
    return dict(
        stage=tuple(contact - STAGE_BACK * d),
        contact=tuple(contact),
        follow=tuple(target - so * d),
        heading=float(np.arctan2(d[1], d[0])),
        standoff=float(so))
