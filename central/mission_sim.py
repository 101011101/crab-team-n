"""The whole stack, simulated: solver to allocator to waypoints to motion.

    ./.venv/bin/python mission_sim.py

Runs the demo scenario with no hardware, no network and no vision:

    two crabs start in room 1
    they explore, and discover the corridor is blocked
    the solver cannot satisfy reach(R2), so it emits clear(e2)
    a crab pushes the block clear
    room 2 opens and the species behind it are found

Nothing in that sequence is scripted. Each step is what falls out of a
precondition that could not be satisfied. The log prints the cause beside
every task so the derivation is visible rather than asserted.

The bridge worth understanding is `waypoint_for`: the planner speaks in
places and edges, the motion controller speaks in metres. That function is
the only place the two meet.
"""

import argparse
import math

from allocator import assign
from geometry import Pose, Waypoint
from motion import WaypointController
from sim import SimRobot, two_room_arena
from solver import EXPLORE, GOTO, OBSERVE, PUSH
from world import (AgentAt, EdgeSeen, PlaceSeen, Surveyed, ThingSeen, World,
                   Agent, BLOCK, BLOCKED, EXPLORED, FRONTIER, REACHABLE,
                   UNKNOWN)


DT = 0.05
PLAN_EVERY = 1.0            # the planner runs at 1 Hz; motion at 20 Hz
SIGHT_M = 0.55              # how far a crab can recognise something


# Where each place and edge physically is. The planner never sees these
# numbers; only this file maps a symbol to a point on the floor.
PLACE_XY = {
    "R1":  (1.00, 1.00),
    "C":   (2.30, 1.00),
    "R2N": (4.00, 1.50),
    "R2S": (4.00, 0.50),
}

# Sweeping a place, relative to its centre. Arriving at the middle of a
# room is not exploring it: the first run reached R2 and found none of the
# four species, because all of them sat further from the centre than a
# crab can see. EXPLORE covers the zone instead.
SWEEP = [(-0.45, 0.00), (+0.45, 0.00), (+0.45, +0.30), (-0.45, +0.30)]

EDGE_XY = {
    "e2": (2.50, 1.00),     # the blocked corridor mouth
}

# Intermediate points for crossing an edge, keyed by (from, to).
#
# A waypoint is a destination, not a route: the controller drives straight
# at it. Between two rooms joined by a narrow corridor, straight is into
# the wall -- the first run wedged a crab in the corridor mouth at
# (1.91, 0.93) doing exactly that. These points line the crab up with the
# opening before it commits to the gap.
#
# This is the GOTO skill deciding HOW. The planner only ever said "be at
# C", and it stays ignorant of the geometry.
_INTO_R2 = [(2.85, 1.00), (3.30, 1.00)]
_OUT_OF_R2 = [(3.30, 1.00), (2.85, 1.00)]

EDGE_ROUTE = {
    ("R1", "C"):    [(1.85, 1.00)],
    ("C", "R1"):    [(2.15, 1.00)],
    ("C", "R2N"):   _INTO_R2,
    ("C", "R2S"):   _INTO_R2,
    ("R2N", "C"):   _OUT_OF_R2,
    ("R2S", "C"):   _OUT_OF_R2,
    ("R1", "R2N"):  [(1.85, 1.00)] + _INTO_R2,
    ("R1", "R2S"):  [(1.85, 1.00)] + _INTO_R2,
    ("R2N", "R2S"): [(4.00, 1.00)],
    ("R2S", "R2N"): [(4.00, 1.00)],
}

SPECIES_PLACE = "R2"


def route_for(task, at_place):
    """The waypoints a task means, in order.

    The planner deals in places and edges. The motion controller deals in
    metres. This is the only place the two vocabularies meet, which is why
    a place with no floor position is a hard error rather than a silent
    drive to the origin.
    """
    if task.op in (GOTO, EXPLORE, OBSERVE):
        if task.target not in PLACE_XY:
            raise KeyError(f"no floor position for place {task.target!r}")

        cx, cy = PLACE_XY[task.target]
        points = list(EDGE_ROUTE.get((at_place, task.target), [])) + [(cx, cy)]

        if task.op == EXPLORE:
            points += [(cx + dx, cy + dy) for dx, dy in SWEEP]

        return [Waypoint(x, y, tolerance_m=0.12) for x, y in points]

    if task.op == PUSH:
        if task.target not in EDGE_XY:
            raise KeyError(f"no floor position for edge {task.target!r}")

        x, y = EDGE_XY[task.target]

        # Line up with the gap first, then stop short of the block: the
        # crab pushes it, it does not occupy the same space.
        return [Waypoint(x - 0.45, y, tolerance_m=0.10),
                Waypoint(x - 0.17, y, tolerance_m=0.06)]

    raise KeyError(f"no waypoint rule for {task.op}")


class Crab:
    """One simulated robot plus the controller driving it."""

    def __init__(self, serial, pose):
        self.serial = serial
        self.robot = SimRobot(serial, pose)
        self.controller = WaypointController()
        self.task = None
        self.route = []
        self.leg = 0

    @property
    def pose(self):
        return self.robot.pose

    @property
    def waypoint(self):
        return self.route[self.leg] if self.leg < len(self.route) else None

    @property
    def route_done(self):
        return self.leg >= len(self.route)

    def take(self, task, at_place):
        if task is not None and str(task) == str(self.task):
            return
        self.task = task
        self.route = [] if task is None else route_for(task, at_place)
        self.leg = 0
        self.controller.reset()

    def advance(self):
        """Move to the next leg. Returns True when the route is finished."""
        self.leg += 1
        self.controller.reset()
        return self.route_done


class Mission:
    def __init__(self, verbose=True):
        self.arena = two_room_arena()
        self.world = World("R1")
        self.verbose = verbose
        self.now = 0.0
        self.events = []

        # What the crabs know at the start: they are in room 1, and there
        # is something to the east. Everything else they must discover.
        self.world.observe(PlaceSeen("R1", FRONTIER))
        self.world.observe(PlaceSeen("C", FRONTIER))
        self.world.observe(EdgeSeen("R1", "C", "e1", REACHABLE))

        self.crabs = {
            1: Crab(1, Pose(0.60, 0.70, 0.0)),
            2: Crab(2, Pose(0.60, 1.30, 0.0)),
        }

        for serial, crab in self.crabs.items():
            agent = self.world.add_agent(Agent(serial=serial, place="R1"))
            agent.timeout_s = 45.0
            self.world.observe(AgentAt(serial, "R1"))

        self.species_found = {}

    # -----------------------------------------------------------

    def log(self, text):
        self.events.append((self.now, text))
        if self.verbose:
            print(f"  {self.now:6.2f}s  {text}")

    # -----------------------------------------------------------
    # PERCEPTION -- what a crab learns by being somewhere
    # -----------------------------------------------------------

    def perceive(self, crab):
        """Write what this crab can currently see into the world model."""
        pose = crab.pose

        # Arrived at a place: it is explored, and its neighbours appear.
        for pid, (px, py) in PLACE_XY.items():
            if math.hypot(pose.x - px, pose.y - py) > SIGHT_M:
                continue

            # Arriving is not surveying. Marking the place surveyed here
            # retired the EXPLORE task on arrival and cancelled the sweep
            # before it ran, so a crab reached room 2 and found none of the
            # species sitting further out than it could see. The survey fact
            # is written when the sweep completes, in `complete`.
            if pid == "C" and self.world.place("R2N") is None:
                self.discover_corridor(crab)

            self.world.observe(AgentAt(crab.serial, pid))

        self.find_species(crab)

    def discover_corridor(self, crab):
        """Reaching the corridor reveals room 2 and the block in the way."""
        if self.world.place("R2N") is not None:
            return

        self.world.observe(PlaceSeen("R2N", FRONTIER, found_by=crab.serial))
        self.world.observe(PlaceSeen("R2S", FRONTIER, found_by=crab.serial))
        self.world.observe(EdgeSeen("C", "R2N", "e2", BLOCKED, BLOCK))
        self.world.observe(EdgeSeen("C", "R2S", "e3", BLOCKED, BLOCK))
        self.world.observe(EdgeSeen("R2N", "R2S", "e4", REACHABLE))
        self.world.observe(ThingSeen("block", "C", "block", 1, "e2"))

        self.log(f"crab {crab.serial} sees two zones beyond the corridor, "
                 f"blocked by one block")

    def find_species(self, crab):
        for index, sp in enumerate(self.arena.species):
            if sp["found_by"] is not None:
                continue
            if math.hypot(crab.pose.x - sp["x"], crab.pose.y - sp["y"]) > SIGHT_M:
                continue

            sp["found_by"] = crab.serial
            self.species_found[index] = sp

            self.log(f"crab {crab.serial} found a {sp['colour']} species "
                     f"({len(self.species_found)} of {len(self.arena.species)})")

    # -----------------------------------------------------------
    # EXECUTION
    # -----------------------------------------------------------

    def execute(self, crab):
        """Drive one tick toward the current task's waypoint."""
        agent = self.world.agents[crab.serial]

        if crab.task is None or crab.waypoint is None:
            crab.robot.command(0, 0, self.now)
            return

        left, right = crab.controller.step(crab.pose, crab.waypoint)

        if crab.controller.arrived:
            if crab.advance():
                self.complete(crab)
            crab.robot.command(0, 0, self.now)
            return

        if crab.controller.stalled:
            # Wedged. Say so rather than pushing: the allocator needs the
            # failure to free the crab and remember what defeated it.
            self.log(f"crab {crab.serial} STALLED on {crab.task}")
            agent.failed = True
            crab.take(None, agent.place)
            crab.robot.command(0, 0, self.now)
            return

        crab.robot.command(left, right, self.now)

        before = crab.pose
        crab.robot.step(DT, self.now)

        if self.arena.blocked_at(crab.pose.x, crab.pose.y):
            crab.robot.pose = before

    def complete(self, crab):
        """The crab reached the waypoint. Apply the task's effect."""
        task = crab.task

        if task.op == EXPLORE:
            self.world.observe(PlaceSeen(task.target, EXPLORED,
                                         found_by=crab.serial))
            self.world.observe(Surveyed(task.target))
            self.log(f"crab {crab.serial} finished sweeping {task.target}")

        if task.op == PUSH:
            for block in self.arena.blocks:
                block["cleared"] = True

            # One block held both zones shut, so clearing it opens both.
            self.world.observe(EdgeSeen("C", "R2N", "e2", REACHABLE))
            self.world.observe(EdgeSeen("C", "R2S", "e3", REACHABLE))
            self.log(f"crab {crab.serial} CLEARED {task.target} -- both zones open")

        crab.take(None, self.world.agents[crab.serial].place)

    # -----------------------------------------------------------

    def run(self, limit_s=240.0):
        if self.verbose:
            print("=" * 74)
            print("  SIMULATED MISSION -- solver drives the crabs")
            print("=" * 74)

        next_plan = 0.0

        while self.now < limit_s:
            for crab in self.crabs.values():
                self.perceive(crab)

            if self.now >= next_plan:
                next_plan = self.now + PLAN_EVERY

                log = [] if self.verbose else None
                given = assign(self.world, self.world.live_agents(),
                               now=self.now, log=log)

                for line in (log or []):
                    self.log(line)

                for serial, crab in self.crabs.items():
                    crab.take(given.get(serial), self.world.agents[serial].place)

            for crab in self.crabs.values():
                self.execute(crab)

            if len(self.species_found) == len(self.arena.species):
                self.log("all species found")
                break

            self.now += DT

        return self.report()

    def report(self):
        done = len(self.species_found)
        total = len(self.arena.species)
        cleared = all(b["cleared"] for b in self.arena.blocks)

        print()
        print("=" * 74)
        print("  RESULT")
        print("=" * 74)
        print(f"  mission time       {self.now:.1f}s")
        print(f"  species found      {done}/{total}")
        print(f"  corridor cleared   {cleared}")
        for pid in ("R2N", "R2S"):
            place = self.world.place(pid)
            print(f"  {pid} explored       "
                  f"{place is not None and place.state == EXPLORED}")

        for serial, crab in self.crabs.items():
            print(f"  crab {serial}: travelled {crab.robot.distance_travelled:.2f} m")

        ok = done == total and cleared
        print()
        print("  MISSION COMPLETE" if ok else "  MISSION INCOMPLETE")
        return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    return Mission(verbose=not args.quiet).run()


if __name__ == "__main__":
    raise SystemExit(main())
