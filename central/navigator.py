"""Turn one planner task into a sequence of waypoints, and drive it.

This is the seam between the planner and the wire, and it is deliberately
ignorant of where the pose came from. Give it a pose each tick and it
returns track speeds; the pose may come from the simulator or from marker
detection on a real camera, and nothing here can tell the difference. That
is what lets the same code run the simulated mission and the real one.

Two jobs, kept separate on purpose:

  route_for(task, at)  what the task means in metres -- a list of waypoints
  Navigator.step(pose) what to send the motors right now

The planner says "be at C". It never learns that reaching C means lining up
with a doorway first. That geometry lives in the arena map below, which is
the only place symbols become coordinates.
"""

from geometry import Pose, Waypoint
from motion import WaypointController
from solver import EXPLORE, GOTO, OBSERVE, PUSH


class ArenaMap:
    """Where each place and edge physically is.

    `legs` holds the intermediate points needed to cross an edge. A waypoint
    is a destination, not a route: the controller drives straight at it, and
    between two rooms joined by a narrow gap, straight is into the wall.
    """

    def __init__(self, places, edges, legs=None, sweep=None, push_standoff=0.24):
        self.places = places            # pid -> (x, y)
        self.edges = edges              # edge name -> (x, y)
        self.legs = legs or {}          # (from_pid, to_pid) -> [(x, y), ...]
        self.sweep = sweep or [(-0.45, 0.0), (0.45, 0.0), (0.45, 0.3), (-0.45, 0.3)]

        # How close the robot's CENTRE can get to the blocker's centre. It is
        # the blocker's half-extent plus the chassis radius, so it changes
        # with the obstacle: a wide box stops the robot further out than a
        # small circle. Get this wrong and the final push waypoint lands
        # inside the obstacle, where the robot can never arrive and simply
        # grinds against it until the stall timer fires.
        self.push_standoff = push_standoff

    def _chain(self, start, goal, via):
        """Intermediate points for start -> goal, following `via` if given."""
        if not via or len(via) < 2:
            return list(self.legs.get((start, goal), []))

        points = []
        for a, b in zip(via, via[1:]):
            points += list(self.legs.get((a, b), []))
            if b != goal and b in self.places:
                points.append(self.places[b])       # pass through each hop
        return points

    def route_for(self, task, at_place, via=None):
        """The waypoints a task means, in order.

        `via` is the sequence of places the PLANNER chose, from the world
        model. It matters: the planner knows which edges are passable and
        this map does not. Without it the robot is routed by geometry alone
        and will drive at a blocked corridor because it is the short way --
        which is exactly what happened the first time this ran.
        """
        if task.op in (GOTO, EXPLORE, OBSERVE):
            if task.target not in self.places:
                raise KeyError(f"no floor position for place {task.target!r}")

            cx, cy = self.places[task.target]
            points = self._chain(at_place, task.target, via) + [(cx, cy)]

            # Arriving at a room's centre is not surveying it. Anything
            # further from the centre than the robot can see stays unfound
            # unless the robot actually sweeps the space.
            if task.op == EXPLORE:
                points += [(cx + dx, cy + dy) for dx, dy in self.sweep]

            return [Waypoint(x, y, tolerance_m=0.12) for x, y in points]

        if task.op == PUSH:
            if task.target not in self.edges:
                raise KeyError(f"no floor position for edge {task.target!r}")

            x, y = self.edges[task.target]
            legs = self._chain(at_place, task.target, via)

            # Line up with the gap first, then stop at the standoff: the
            # crab pushes the blocker, it does not occupy the same space.
            approach = self.push_standoff + 0.22
            return [Waypoint(px, py, tolerance_m=0.12) for px, py in legs] + [
                Waypoint(x - approach, y, tolerance_m=0.10),
                Waypoint(x - self.push_standoff, y, tolerance_m=0.09)]

        raise KeyError(f"no waypoint rule for {task.op}")


class Navigator:
    """Drives one robot along the route for its current task.

    Holds no opinion about the robot. `step` takes the pose the caller
    believes and returns what to send; the caller decides whether that goes
    to a simulator or to an ESP32.
    """

    def __init__(self, arena: ArenaMap):
        self.arena = arena
        self.controller = WaypointController()
        self.task = None
        self.route = []
        self.leg = 0

    @property
    def waypoint(self):
        return self.route[self.leg] if self.leg < len(self.route) else None

    @property
    def route_done(self):
        return self.leg >= len(self.route)

    @property
    def stalled(self):
        return self.controller.stalled

    def take(self, task, at_place, via=None):
        """Accept a task. Re-taking the same task does not restart the route.

        `via` is the planner's chosen sequence of places. Pass it whenever
        the world model can supply one; without it the route is chosen by
        geometry and may cross an edge the planner knows is blocked.
        """
        if task is not None and str(task) == str(self.task):
            return

        self.task = task
        self.route = [] if task is None else self.arena.route_for(task, at_place, via)
        self.leg = 0
        self.controller.reset()

    def step(self, pose: Pose):
        """Return (left, right, event).

        `event` is None, "arrived" when the last waypoint is reached, or
        "stalled" when the robot is wedged. The caller applies the task's
        effect on "arrived" and raises the failure flag on "stalled" -- this
        class does not know what a task means, only where it goes.
        """
        if self.task is None or self.waypoint is None:
            return 0, 0, None

        left, right = self.controller.step(pose, self.waypoint)

        if self.controller.arrived:
            self.leg += 1
            self.controller.reset()
            return 0, 0, ("arrived" if self.route_done else None)

        if self.controller.stalled:
            return 0, 0, "stalled"

        return left, right, None
