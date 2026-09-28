"""The control loop.

One tick, for every robot, in this order:

    1. FLOOD    if the world changed, recompute the reach field. One wavefront
                serves every robot, so this is never per-robot work.
    2. ASSIGN   a robot with no task gets one:
                  goal reachable            -> DRIVE to it
                  goal reachable if a box   -> PUSH that box. Which box is found by
                  were gone                    deleting each in turn and re-flooding;
                                               that is the whole test, per the spec.
                  neither                   -> EXPLORE the nearest frontier
    3. ACT      follow the current waypoint with the motion controller.

The push is deliberately NOT open-loop. Where a shoved box ends up depends on the
friction across its contact patch, which is invisible and changes as it slides, so
the box is moved one INCREMENT at a time and re-measured. The route is declared open
when the field says so, never when the robot believes it pushed far enough.

Reuses central/: Pose, Waypoint and WaypointController. The spec asks for that
explicitly — the field replaces the hand-authored legs table, not the controller.
"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "central"))
from geometry import Pose, Waypoint          # noqa: E402
from motion import wrap_angle                # noqa: E402

import grid as G                             # noqa: E402
import push as PUSH                          # noqa: E402
import reach_field as rf                     # noqa: E402

DRIVE_SPEED = 0.45          # model units / s
TURN_RATE = 1.8             # rad / s
ARRIVE = 0.10               # model units
TURN_FIRST = 0.55           # rad. Above this, turn in place before driving.
SENSE = 0.9                 # model units. How far a robot can see floor.


class Robot:
    """Differential drive, reduced to (v, omega). Turning radius, not a pivot."""

    def __init__(self, name, x, y, theta=0.0):
        self.name, self.pose = name, Pose(x, y, theta)
        self.task = None            # ("drive"|"push"|"explore", payload)
        self.route = []             # remaining waypoints, world coords
        self.log = []
        self.last = None
        self.pushing = None
        self.blocked_steps = 0
        self.best_d = None      # closest approach to the current waypoint
        self.stall = 0

    def steer(self, dt, target, free=None):
        """Returns True when the waypoint is reached.

        A refused move is not a no-op: the caller must replan, or the robot grinds
        against the obstacle forever. Without this check a robot drives straight
        into a carton and strands itself inside blocked space, where the reach field
        has no finite distance and it can never recover.
        """
        p = self.pose
        dx, dy = target[0] - p.x, target[1] - p.y
        d = math.hypot(dx, dy)
        if d < ARRIVE:
            return True
        err = wrap_angle(math.atan2(dy, dx) - p.theta)
        w = max(-TURN_RATE, min(TURN_RATE, err * 2.5))
        v = 0.0 if abs(err) > TURN_FIRST else DRIVE_SPEED * max(0.25, math.cos(err))
        th = wrap_angle(p.theta + w * dt)
        nx, ny = p.x + v * math.cos(th) * dt, p.y + v * math.sin(th) * dt
        if v > 0 and free is not None and not free(nx, ny):
            # Refused. Turning toward the waypoint will not help -- the waypoint is
            # what we are driving at. Bias the heading aside so the robot can slide
            # off the corner instead of grinding against it.
            self.blocked_steps += 1
            bias = TURN_RATE * dt * (1 if (self.blocked_steps // 8) % 2 == 0 else -1)
            self.pose = Pose(p.x, p.y, wrap_angle(th + bias))
            return False
        self.blocked_steps = 0
        self.pose = Pose(nx, ny, th)
        return False


class Sim:
    def __init__(self, g, robots, goal_xy):
        self.g, self.robots, self.goal = g, robots, goal_xy
        self.blocked_extra = np.zeros_like(g.state, bool)   # boxes as they move
        self.boxes = {b["id"]: dict(b) for b in g.boxes}
        self.moved = {}
        self.dirty = True
        self.gen = 0          # bumped on every re-flood; caches key off it
        self._blockers = {}
        self.events = []
        self.t = 0.0
        self.seen = 0

    # --- the field ---------------------------------------------------------
    def passable(self, without=None, inflate=True):
        """Free cells, SHRUNK by the chassis radius.

        Planning on un-inflated free space treats the robot as a point. It then
        plans routes that graze a carton corner, the body clips it, the move is
        refused, and the robot sits pointing straight at the obstacle with no reason
        to turn. Eroding the free mask once is what makes the plan drivable.
        """
        ok = self.g.passable() & ~self.blocked_extra
        if inflate:
            ok = self.g.inflate(ok, 1)      # see Grid.drivable: one shared mask
        if without is not None:
            b = self.boxes[without]
            r = max(b["w"], b["h"]) / 2 + self.g.cell
            j, i = self.g.to_cell(b["cx"], b["cz"])
            rc = int(np.ceil(r / self.g.cell))
            for dj in range(-rc, rc + 1):
                for di in range(-rc, rc + 1):
                    a, c = j + dj, i + di
                    if self.g.inside(a, c):
                        ok[a, c] = True
        return ok

    def flood(self):
        self.pass_now = self.passable()
        self.field = rf.wavefront(self.pass_now, [self.g.to_cell(*self.goal)])
        self.dirty = False
        self.gen += 1
        self._blockers.clear()

    def reachable(self, xy):
        j, i = self.g.to_cell(*xy)
        return self.g.inside(j, i) and np.isfinite(self.field[j, i])

    def blocking_boxes(self, frm):
        """Every box that, removed on its own, makes the goal reachable from here.

        Delete it, re-flood, see if the distance becomes finite. That is exactly how
        docs/reach-field.md says clear(edge) should be decided.

        Returns a LIST, least boxed-in first, because a box can gate the route and
        still have nowhere to go. The caller keeps trying until one has a feasible
        placement — a blocker with no legal push is not a plan.
        """
        gj, gi = self.g.to_cell(*frm)
        key = (gj, gi)
        if key in self._blockers:
            return self._blockers[key]        # one flood per box is not cheap; the
                                              # answer only changes when the grid does
        out = []
        for bid in sorted(self.boxes, key=lambda k: -PUSH.free_ratio(
                self.g, self.boxes[k], ~self.pass_now)):
            d = rf.wavefront(self.passable(without=bid), [self.g.to_cell(*self.goal)])
            if np.isfinite(d[gj, gi]):
                out.append(bid)
        self._blockers[key] = out
        return out

    # --- assignment --------------------------------------------------------
    def assign(self, r):
        if self.reachable((r.pose.x, r.pose.y)):
            wp, _ = rf.route(self.g, self.pass_now, (r.pose.x, r.pose.y), self.goal)
            if wp:
                r.task, r.route = ("drive", None), wp
                if r.last != "drive":
                    self.note(f"{r.name} DRIVE: goal reachable, {len(wp)} waypoints")
                    r.last = "drive"
                return
        cands = self.blocking_boxes((r.pose.x, r.pose.y))
        mine = {o.pushing for o in self.robots if o is not r and o.pushing}
        reach_r = self.field_from(r) if cands else None
        for bid in cands:
            if bid in mine:
                continue                      # another robot already has this one
            plan = PUSH.choose_target(self.g, self.boxes[bid], ~self.pass_now,
                                      self.goal, reach_r)
            if plan:
                t, d, dist = plan
                w = PUSH.waypoints(self.boxes[bid], t, d)
                r.task = ("push", dict(box=bid, target=t, dir=d, **w))
                # Get to the staging point the same way as anywhere else: through
                # the field. Only stage -> contact -> follow is driven straight,
                # and those three are collinear on the push axis by construction.
                approach, _ = rf.route(self.g, self.pass_now,
                                       (r.pose.x, r.pose.y), w["stage"])
                if approach is None:
                    continue              # cannot get behind it; try the next box
                r.route = approach + [w["contact"], w["follow"]]
                r.pushing = bid
                self.note(f"{r.name} PUSH {bid}: gates the route, "
                          f"{dist:.2f} units, standoff {w['standoff']:.2f}")
                r.last = "push"
                return
        if cands and r.last != "blocked":
            self.note(f"{r.name} sees {len(cands)} blocker(s) but none it may push "
                      f"has anywhere to go; exploring instead")
            r.last = "blocked"
        fr = rf.frontier_cells(self.g.state, self.pass_now)
        if len(fr):
            here = self.g.to_cell(r.pose.x, r.pose.y)
            k = np.argmin(np.hypot(fr[:, 0] - here[0], fr[:, 1] - here[1]))
            tgt = self.g.to_world(*fr[k])
            if r.last != "explore":
                self.note(f"{r.name} EXPLORE: goal unreachable, working the frontier")
                r.last = "explore"
            r.task, r.route = ("explore", None), [tgt]
        else:
            r.task = ("idle", None)

    def field_from(self, r):
        return rf.wavefront(self.pass_now, [self.g.to_cell(r.pose.x, r.pose.y)])

    def free_for(self, r):
        """Can this robot stand here?

        A robot that is pushing a box must be allowed into that box's cells: driving
        into where it stands is HOW it moves. Every other robot is refused.
        """
        if not r.pushing:
            return self.free_at
        b = self.boxes[r.pushing]
        rad = max(b["w"], b["h"]) / 2 + 2 * self.g.cell

        def ok(x, y):
            if abs(x - b["cx"]) < rad and abs(y - b["cz"]) < rad:
                return True
            return self.free_at(x, y)
        return ok

    def free_at(self, x, y):
        j, i = self.g.to_cell(x, y)
        if not self.g.inside(j, i):
            return False
        return bool(self.pass_now[j, i])

    # --- perception --------------------------------------------------------
    def observe(self, r):
        """UNKNOWN cells the robot can see become FREE.

        Without this the map never grows, every frontier stays a frontier, and a
        robot sent to explore arrives and is immediately sent to the same cell
        again. Exploration only terminates because the unknown shrinks.

        Line of sight is required: a robot does not see through a carton.
        """
        g = self.g
        j0, i0 = g.to_cell(r.pose.x, r.pose.y)
        if getattr(r, "_seen_from", None) == (j0, i0):
            return 0
        r._seen_from = (j0, i0)
        rc = int(SENSE / g.cell)
        blocked = ~g.passable() | self.blocked_extra
        gained = 0
        for dj in range(-rc, rc + 1):
            for di in range(-rc, rc + 1):
                if dj * dj + di * di > rc * rc:
                    continue
                a, b = j0 + dj, i0 + di
                if not g.inside(a, b) or g.state[a, b] != G.UNKNOWN:
                    continue
                if rf.clear_line((j0, i0), (a, b), ~blocked):
                    g.state[a, b] = G.FREE
                    gained += 1
        if gained:
            self.dirty = True
        return gained

    # --- one tick ----------------------------------------------------------
    def tick(self, dt):
        for r in self.robots:
            self.seen += self.observe(r)
        if self.dirty:
            self.flood()
        for r in self.robots:
            # The moment the field says the goal is reachable, drop whatever else
            # you were doing. This is the whole point of a shared field: the route
            # opens for EVERY robot at once, including the ones that had nothing to
            # do with the push, and nobody has to be told.
            if r.task and r.task[0] != "drive" and self.reachable((r.pose.x, r.pose.y)):
                if r.pushing:
                    self.note(f"{r.name} drops the push: the route is open anyway")
                r.task, r.route, r.pushing, r.last = None, [], None, None
                r.best_d, r.stall = None, 0
            if r.task is None:
                self.assign(r)
            if not r.route:
                if r.task and r.task[0] == "push" and r.pushing:
                    self.shove(r)
                else:
                    r.task = None
                continue
            wp = r.route[0]
            if r.steer(dt, wp, self.free_for(r)):
                r.route.pop(0)
                r.best_d, r.stall = None, 0
                continue

            # Stall guard. A robot turning in place has v = 0, so the collision
            # counter never fires: it can oscillate about a waypoint forever and
            # look busy. Watch the distance instead. central/motion.py watches for
            # the same thing on the real robot.
            d = math.hypot(wp[0] - r.pose.x, wp[1] - r.pose.y)
            if r.best_d is None or d < r.best_d - 1e-3:
                r.best_d, r.stall = d, 0
            else:
                r.stall += 1

            if r.stall > 40:
                r.route.pop(0)                 # give up on this one, take the next
                r.best_d, r.stall = None, 0
                if not r.route and r.task and r.task[0] != "push":
                    r.task, r.last = None, None
            elif r.blocked_steps > 25:
                r.blocked_steps = 0            # wedged. Drop the plan, think again.
                r.task, r.route, r.pushing, r.last = None, [], None, None
                r.best_d, r.stall = None, 0
        self.t += dt

    def shove(self, r):
        """Move the box one increment along the push axis, then re-measure.

        The box moves because the robot drove into where it was standing; this is the
        simulated consequence, not a prediction of where it will slide to.
        """
        b = self.boxes[r.pushing]
        d = r.task[1]["dir"]
        step = min(PUSH.INCREMENT, float(np.linalg.norm(r.task[1]["target"] -
                                                        np.array([b["cx"], b["cz"]]))))
        if step < self.g.cell * 0.5:
            self.note(f"{r.name} finished pushing {r.pushing}")
            r.pushing, r.task, r.route = None, None, []
            self.dirty = True
            return
        b["cx"] += float(d[0]) * step
        b["cz"] += float(d[1]) * step
        self.moved[r.pushing] = (b["cx"], b["cz"])
        self.restamp()
        self.dirty = True
        self.flood()
        w = PUSH.waypoints(b, r.task[1]["target"], d)
        r.task[1].update(w)
        r.route = [w["contact"], w["follow"]]
        if self.reachable((r.pose.x, r.pose.y)):
            self.note(f"{r.name} route OPEN after moving {r.pushing} "
                      f"{step:.2f} units — field re-flooded")
            r.pushing, r.task, r.route = None, None, []

    def restamp(self):
        """Redraw every box at its current position. Cells a box has left become
        free, which is what lets the wavefront flood through a newly opened gap."""
        self.blocked_extra[:] = False
        keep = self.g.pushable.copy()
        for bid, b in self.boxes.items():
            if bid not in self.moved:
                continue
            j, i = self.g.to_cell(*[v for v in (b["cx"], b["cz"])])
            rc = max(1, int(max(b["w"], b["h"]) / 2 / self.g.cell))
            for dj in range(-rc, rc + 1):
                for di in range(-rc, rc + 1):
                    a, c = j + dj, i + di
                    if self.g.inside(a, c):
                        self.blocked_extra[a, c] = True
        # original footprints of moved boxes are released
        for bid in self.moved:
            ob = next(x for x in self.g.boxes if x["id"] == bid)
            j, i = self.g.to_cell(ob["cx"], ob["cz"])
            rc = max(1, int(max(ob["w"], ob["h"]) / 2 / self.g.cell))
            for dj in range(-rc, rc + 1):
                for di in range(-rc, rc + 1):
                    a, c = j + dj, i + di
                    if self.g.inside(a, c):
                        keep[a, c] = False
        self.g.state = np.where(self.g.pushable & ~keep, G.FREE, self.g.state)
        self.g.pushable = keep

    def note(self, text):
        self.events.append(dict(t=round(self.t, 2), text=text))
        print(f"  t={self.t:6.2f}  {text}")
