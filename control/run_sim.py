"""Run the control loop on a real 2D map, with a couple of robots.

The map is the TEMPLATE: swap in any stage-4 map and this runs unchanged.

The scenario is chosen from the map itself, not hand-placed. It looks for a goal
that is genuinely GATED by a pushable carton — reachable if the carton were gone,
unreachable while it stands. On the IMG_7373/4/5 map, 645 cells are gated that way.
That is the case the reach field was specified for: the route opens for every robot
at once the moment the box moves, and nobody has to be told.

Writes a trace JSON (every pose, every box, every event) and a PNG.
"""
import argparse, json, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid as G
import loop as L
import reach_field as rf
import push as PUSH


def pick_scenario(g):
    """Start in the biggest observed free area; goal behind ONE pushable carton.

    Not the deepest gated cell: a pocket behind a continuous run of cartons needs
    several boxes moved and is not the case the spec describes. We want a goal that
    exactly ONE box gates, because "delete it, re-flood, is the goal finite now" is
    the test clear(edge) is supposed to make.
    """
    drive = g.drivable()
    lab = cv2.connectedComponentsWithStats(
        ((g.state == G.FREE) & drive).astype(np.uint8), 8)[1]
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    home = np.argwhere(lab == counts.argmax())
    start = tuple(home[len(home) // 2])

    # the SAME mask the control loop drives on, or the goal lands somewhere no
    # robot can stand and every wavefront comes back empty
    now = g.drivable()
    d_now = rf.wavefront(now, [start])
    reach = rf.wavefront(now, [start])
    best = None
    for b in g.boxes:
        d = rf.wavefront(_without(g, now, b), [start])
        gained = np.argwhere(np.isfinite(d) & ~np.isfinite(d_now))
        if not len(gained):
            continue
        # The box must gate the goal AND have somewhere to go. A carton wedged in
        # the middle of a run gates plenty and cannot be pushed anywhere, which is
        # a real dead end on this map, not a scenario.
        goal_try = g.to_world(*gained[np.argmax([d[j, i] for j, i in gained])])
        if PUSH.choose_target(g, b, ~now, goal_try, reach) is None:
            continue
        if best is None or len(gained) > len(best[1]):
            best = (b, gained, d)
    if best is None:
        raise SystemExit("no goal on this map is gated by a carton that can move")
    box, gained, d = best
    k = np.argmax([d[j, i] for j, i in gained])
    return start, tuple(gained[k]), box["id"], int(len(gained))


def _without(g, passable, b):
    ok = passable.copy()
    j, i = g.to_cell(b["cx"], b["cz"])
    rc = int(np.ceil((max(b["w"], b["h"]) / 2 + g.cell) / g.cell))
    for dj in range(-rc, rc + 1):
        for di in range(-rc, rc + 1):
            a, c = j + dj, i + di
            if g.inside(a, c):
                ok[a, c] = True
    return ok


def render(g, trace, path):
    pal = np.array([[28, 32, 36], [214, 226, 222], [92, 108, 116], [232, 168, 60]], np.uint8)
    base = np.where(g.pushable, 3, np.where(g.static, 2, g.state)).astype(np.uint8)
    img = cv2.resize(pal[base], (g.w * 6, g.h * 6), interpolation=cv2.INTER_NEAREST)
    px = lambda x, y: (int((x - g.origin[0]) / g.cell * 6), int((y - g.origin[1]) / g.cell * 6))
    for r, col in zip(trace["robots"], [(230, 90, 60), (60, 180, 230)]):
        pts = [px(*p[:2]) for p in r["path"]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, col, 2)
        if pts:
            cv2.circle(img, pts[0], 5, col, -1)
            cv2.circle(img, pts[-1], 7, col, 2)
    for b in trace["boxes_final"]:
        cv2.drawMarker(img, px(b["cx"], b["cz"]), (80, 80, 240), cv2.MARKER_TILTED_CROSS, 11, 2)
    cv2.circle(img, px(*trace["goal"]), 9, (90, 230, 140), 2)
    cv2.imwrite(path, cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", help="stage-4 map JSON")
    ap.add_argument("--robots", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--trace", default="sim_trace.json")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()

    g = G.Grid(a.map)
    start, goal_cell, gate_id, n_gated = pick_scenario(g)
    sx, sy = g.to_world(*start)
    goal = g.to_world(*goal_cell)
    print(f"map     {g.w} x {g.h} cells of {g.cell} model units, {len(g.boxes)} pushable")
    print(f"start   ({sx:.2f}, {sy:.2f})   goal ({goal[0]:.2f}, {goal[1]:.2f})")
    print(f"scenario: {gate_id} alone gates {n_gated} cells; the goal is the deepest\n")

    robots = [L.Robot(f"crab{i+1}", sx + 0.2 * i, sy + 0.2 * i, 0.0) for i in range(a.robots)]
    sim = L.Sim(g, robots, goal)
    paths = {r.name: [] for r in robots}

    steps = int(a.seconds / a.dt)
    for _ in range(steps):
        sim.tick(a.dt)
        for r in robots:
            paths[r.name].append([round(r.pose.x, 4), round(r.pose.y, 4), round(r.pose.theta, 3)])
        if all(np.hypot(r.pose.x - goal[0], r.pose.y - goal[1]) < 0.25 for r in robots):
            sim.note("all robots at the goal")
            break

    arrived = sum(np.hypot(r.pose.x - goal[0], r.pose.y - goal[1]) < 0.25 for r in robots)
    trace = dict(
        map=os.path.basename(a.map), cell=g.cell, origin=g.origin.tolist(),
        w=g.w, h=g.h, goal=list(goal), seconds=round(sim.t, 2),
        gated_cells=n_gated, gating_box=gate_id, arrived=int(arrived),
        boxes_moved={k: [round(v[0], 3), round(v[1], 3)] for k, v in sim.moved.items()},
        boxes_final=[dict(id=b["id"], cx=round(b["cx"], 3), cz=round(b["cz"], 3))
                     for b in sim.boxes.values()],
        events=sim.events,
        robots=[dict(name=r.name, path=paths[r.name]) for r in robots],
        units="model units; monocular SfM has no metre")
    json.dump(trace, open(a.trace, "w"), separators=(",", ":"))
    print(f"\n{arrived}/{len(robots)} robots reached the goal in {sim.t:.1f} s "
          f"({len(sim.moved)} box(es) moved)")
    print(f"wrote {a.trace}")
    if a.png:
        render(g, trace, a.png)
        print(f"wrote {a.png}")


if __name__ == "__main__":
    main()
