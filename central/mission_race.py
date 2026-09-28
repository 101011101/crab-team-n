"""Two routes, two crabs, one reward -- and nobody schedules the race.

    ./.venv/bin/python mission_race.py [--quiet] [--seal-south]

A crab finds the north corridor blocked. The solver cannot satisfy
reach(B) that way, but reach(B) has a second operator -- the south
corridor, still unexplored -- so the allocator hands one crab the push and
the other the unknown route. Whichever opens first satisfies the goal and
the loser's task is released as invalidated.

With --seal-south the alternative is blocked too, and the push becomes the
only way through. Same code, same world, different outcome: the branch is
chosen by what is true, not by what was written.
"""

import argparse, math, time

import race_arena
from allocator import assign
from geometry import Pose
from navigator import Navigator
from sim import SimRobot
from solver import EXPLORE, GOTO, OBSERVE, PUSH
from world import (Agent, AgentAt, BLOCK, BLOCKED, EdgeSeen, EXPLORED,
                   FRONTIER, PlaceSeen, REACHABLE, Surveyed, ThingSeen, World)

DT = 0.05
PLAN_EVERY = 1.0
SIGHT = 0.65


class Crab:
    def __init__(self, serial, pose, arena_map):
        self.serial = serial
        self.robot = SimRobot(serial, pose)
        self.nav = Navigator(arena_map)

    @property
    def pose(self):
        return self.robot.pose


class Race:
    def __init__(self, verbose=True, seal_south=False):
        self.arena = race_arena.build()
        self.seal_south = seal_south

        if seal_south:
            self.arena.add_box(4.60, 0.80, 0.30, 0.45)

        self.world = World("A")
        self.verbose = verbose
        self.now = 0.0
        self.events = []
        self.found = False
        self.winner = None

        self.world.observe(PlaceSeen("A", FRONTIER))

        self.crabs = {1: Crab(1, Pose(0.70, 1.65, 0.0), race_arena.ARENA_MAP),
                      2: Crab(2, Pose(0.70, 0.95, 0.0), race_arena.ARENA_MAP)}
        for s in self.crabs:
            ag = self.world.add_agent(Agent(serial=s, place="A"))
            ag.timeout_s = 60.0
            self.world.observe(AgentAt(s, "A"))

    def log(self, text):
        self.events.append((self.now, text))
        if self.verbose:
            print(f"  {self.now:6.2f}s  {text}")

    # -----------------------------------------------------------

    def perceive(self, crab):
        for pid, (px, py) in race_arena.PLACES.items():
            if math.hypot(crab.pose.x - px, crab.pose.y - py) > SIGHT:
                continue

            self.world.observe(AgentAt(crab.serial, pid))

            if pid == "A" and self.world.place("CN") is None:
                # Standing in the start room reveals both corridor mouths.
                # The rooms beyond them are not visible from here.
                self.world.observe(PlaceSeen("CN", FRONTIER, found_by=crab.serial))
                self.world.observe(PlaceSeen("CS", FRONTIER, found_by=crab.serial))
                self.world.observe(EdgeSeen("A", "CN", "e_an", REACHABLE))
                self.world.observe(EdgeSeen("A", "CS", "e_as", REACHABLE))
                self.log(f"crab {crab.serial} sees two corridors leaving A")

            if pid == "CN" and self.world.edge_named("e_nb") is None:
                self.world.observe(PlaceSeen("B", FRONTIER, found_by=crab.serial))
                self.world.observe(EdgeSeen("CN", "B", "e_nb", BLOCKED, BLOCK))
                self.world.observe(ThingSeen("block", "CN", "block", 1, "e_nb"))
                self.log(f"crab {crab.serial} reaches CN: the way to B is BLOCKED")

            if pid == "CS" and self.world.edge_named("e_sb") is None:
                if self.world.place("B") is None:
                    self.world.observe(PlaceSeen("B", FRONTIER, found_by=crab.serial))
                if self.seal_south:
                    self.world.observe(EdgeSeen("CS", "B", "e_sb", BLOCKED, BLOCK))
                    self.world.observe(ThingSeen("block2", "CS", "block", 1, "e_sb"))
                    self.log(f"crab {crab.serial} reaches CS: also BLOCKED")
                else:
                    self.world.observe(EdgeSeen("CS", "B", "e_sb", REACHABLE))
                    self.log(f"crab {crab.serial} reaches CS: the south route is OPEN")

        for sp in self.arena.species:
            if sp["found_by"] is not None:
                continue
            if math.hypot(crab.pose.x - sp["x"], crab.pose.y - sp["y"]) > SIGHT:
                continue
            sp["found_by"] = crab.serial
            self.found = True
            self.winner = crab.serial
            self.log(f"crab {crab.serial} REACHED THE REWARD")

    def execute(self, crab):
        agent = self.world.agents[crab.serial]
        left, right, event = crab.nav.step(crab.pose)

        if event == "stalled":
            self.log(f"crab {crab.serial} stalled on {crab.nav.task}")
            agent.failed = True
            crab.nav.take(None, agent.place)
            crab.robot.command(0, 0, self.now)
            return

        if event == "arrived":
            self.complete(crab)
            crab.robot.command(0, 0, self.now)
            return

        crab.robot.command(left, right, self.now)
        before = crab.pose
        crab.robot.step(DT, self.now)
        if self.arena.blocked_at(crab.pose.x, crab.pose.y):
            crab.robot.pose = before

    def complete(self, crab):
        task = crab.nav.task
        if task is None:
            return

        if task.op == PUSH:
            name = task.target
            # Both corridor mouths sit at the same x and differ only in y,
            # so matching on x alone cleared the south block when a crab
            # pushed the north one. Match the point, not one coordinate.
            ex, ey = race_arena.EDGE_XY.get(name, (99.0, 99.0))
            for b in self.arena.blocks:
                if math.hypot(b["x"] - ex, b["y"] - ey) < 0.3:
                    b["cleared"] = True
            edge = self.world.edge_named(name)
            if edge is not None:
                self.world.observe(EdgeSeen(edge.a, edge.b, name, REACHABLE))
            self.log(f"crab {crab.serial} CLEARED {name}")

        if task.op == EXPLORE:
            self.world.observe(PlaceSeen(task.target, EXPLORED, found_by=crab.serial))
            self.world.observe(Surveyed(task.target))
            self.log(f"crab {crab.serial} finished sweeping {task.target}")

        crab.nav.take(None, self.world.agents[crab.serial].place)

    # -----------------------------------------------------------

    def run(self, limit=200.0):
        if self.verbose:
            print("=" * 74)
            print("  RACE" + ("  (south sealed too)" if self.seal_south else ""))
            print("=" * 74)

        next_plan = 0.0
        while self.now < limit and not self.found:
            for c in self.crabs.values():
                self.perceive(c)

            if self.now >= next_plan:
                next_plan = self.now + PLAN_EVERY
                log = [] if self.verbose else None
                given = assign(self.world, self.world.live_agents(),
                               now=self.now, log=log)
                for line in (log or []):
                    self.log(line)
                for s, c in self.crabs.items():
                    task = given.get(s)
                    at = self.world.agents[s].place
                    # Ask the world which way is actually passable, rather
                    # than letting the arena map pick the short way.
                    via = None
                    if task is not None and task.op in (GOTO, EXPLORE, OBSERVE):
                        via = self.world.path(at, task.target)
                    elif task is not None and task.op == PUSH:
                        edge = self.world.edge_named(task.target)
                        if edge is not None:
                            via = self.world.path(at, edge.a) or self.world.path(at, edge.b)
                    c.nav.take(task, at, via)

            for c in self.crabs.values():
                self.execute(c)

            self.now += DT

        return self.report()

    def report(self):
        print()
        print("=" * 74)
        print(f"  reward reached   {self.found}" + (f" by crab {self.winner}" if self.winner else ""))
        print(f"  time             {self.now:.1f}s")
        cleared = [b for b in self.arena.blocks if b["cleared"]]
        print(f"  blocks cleared   {len(cleared)}/{len(self.arena.blocks)}")
        for s, c in self.crabs.items():
            print(f"  crab {s}: travelled {c.robot.distance_travelled:.2f} m")
        print()
        print("  RACE RESOLVED" if self.found else "  NO ROUTE FOUND")
        return 0 if self.found else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--seal-south", action="store_true",
                   help="block the alternative too, so pushing is the only way")
    a = p.parse_args()
    return Race(verbose=not a.quiet, seal_south=a.seal_south).run()


if __name__ == "__main__":
    raise SystemExit(main())
