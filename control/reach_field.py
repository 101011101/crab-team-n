"""The reach field — one wavefront that answers both navigation questions.

    Can I get to the goal?      the robot's cell holds a finite distance; descend it.
    If not, where should I go?  the nearest FRONTIER, a free cell touching unknown.

Exploring stops being a separate behaviour. It is what the field tells you to do
when the goal is unreachable. Specified in docs/reach-field.md; this is that build.

Cost: one expansion serves every robot, because they all read the same field. A*
would solve one pair at a time and cost more in total.
"""
from collections import deque
import numpy as np

from grid import UNKNOWN, FREE

INF = np.inf
NEI = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def wavefront(passable, goals):
    """Distance in cells from the nearest goal, across passable cells only.

    Breadth-first with a diagonal step counted as 1.414, so the gradient points
    where a robot would actually drive rather than in staircases.
    """
    h, w = passable.shape
    dist = np.full((h, w), INF)
    q = deque()
    for j, i in goals:
        if 0 <= j < h and 0 <= i < w and passable[j, i]:
            dist[j, i] = 0.0
            q.append((j, i))
    while q:
        j, i = q.popleft()
        d = dist[j, i]
        for dj, di in NEI:
            a, b = j + dj, i + di
            if 0 <= a < h and 0 <= b < w and passable[a, b]:
                nd = d + (1.0 if dj == 0 or di == 0 else 1.41421)
                if nd < dist[a, b] - 1e-9:
                    dist[a, b] = nd
                    q.append((a, b))
    return dist


def frontier_cells(state, passable):
    """FREE cells that touch an UNKNOWN one. This is where the map can still grow."""
    free = (state == FREE) & passable
    unk = (state == UNKNOWN)
    touch = np.zeros_like(free)
    for dj, di in NEI[:4]:
        touch |= np.roll(np.roll(unk, dj, 0), di, 1)
    return np.argwhere(free & touch)


def descend(dist, start):
    """Walk downhill from start to the goal. Returns a list of cells, or None."""
    j, i = start
    if not np.isfinite(dist[j, i]):
        return None
    path = [(j, i)]
    seen = {(j, i)}
    while dist[j, i] > 0:
        best, bd = None, dist[j, i]
        for dj, di in NEI:
            a, b = j + dj, i + di
            if 0 <= a < dist.shape[0] and 0 <= b < dist.shape[1] \
               and (a, b) not in seen and dist[a, b] < bd - 1e-9:
                best, bd = (a, b), dist[a, b]
        if best is None:
            return None          # local minimum: no strictly-downhill neighbour
        j, i = best
        seen.add(best)
        path.append(best)
        if len(path) > dist.size:
            return None
    return path


def simplify(path, passable, max_skip=64):
    """Turn a jagged cell path into waypoints.

    Two passes, per the spec: keep a point only where the direction changes, then
    drop any point whose neighbours can see each other in a straight line.
    """
    if not path or len(path) < 3:
        return list(path or [])
    kept = [path[0]]
    for k in range(1, len(path) - 1):
        a, b, c = path[k - 1], path[k], path[k + 1]
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            kept.append(b)
    kept.append(path[-1])

    out = [kept[0]]
    k = 0
    while k < len(kept) - 1:
        nxt = k + 1
        for t in range(len(kept) - 1, k, -1):
            if t - k > max_skip:
                continue
            if clear_line(kept[k], kept[t], passable):
                nxt = t
                break
        out.append(kept[nxt])
        k = nxt
    return out


def clear_line(a, b, passable):
    """Is the straight cell line between a and b entirely passable?"""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    for t in np.linspace(0, 1, n * 2 + 1):
        j = int(round(a[0] + (b[0] - a[0]) * t))
        i = int(round(a[1] + (b[1] - a[1]) * t))
        if not passable[j, i]:
            return False
    return True


def route(g, passable, start_xy, goal_xy):
    """World-coordinate convenience: returns waypoints as (x, y) or None."""
    dist = wavefront(passable, [g.to_cell(*goal_xy)])
    path = descend(dist, g.to_cell(*start_xy))
    if path is None:
        return None, dist
    return [g.to_world(j, i) for j, i in simplify(path, passable)], dist
