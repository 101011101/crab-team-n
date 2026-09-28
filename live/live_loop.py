"""The live loop: read the newest map, decide, send track speeds. 10 Hz.

This is the state machine from control/loop.py, driving real hardware instead of a
simulated robot. Everything above it is unchanged: the reach field, the blocker test,
the push planner. The only difference is where the pose comes from and where the
command goes.

    MapService (slow thread)        this loop (10 Hz)
       newest map2d.json  ------->  reload when the file changes
       registered poses   ------->  correct the dead reckoner
                                       |
                                    control.loop.Sim decides
                                       |
                                    waypoint -> track speeds -> Fleet.drive

WHY THE LOOP NEVER WAITS FOR THE MAP: reconstruction takes tens of seconds. A robot
that stops moving while it waits for a better map is worse than a robot driving on a
map that is 30 seconds old. So the map is swapped in atomically and this loop reads
whatever is current.

SAFETY: the loop stops every robot on exit, on exception, and whenever it has not
had a usable pose for STALE_POSE_S. A robot driving on a pose it does not have is
the one failure mode that breaks hardware.
"""
import json, math, os, sys, threading, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "control"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "central"))
import grid as G                     # noqa: E402
import loop as CL                    # noqa: E402
import reach_field as rf             # noqa: E402
from motion import wrap_angle        # noqa: E402

from pose_source import DeadReckoner, tracks_for   # noqa: E402
from tag_pose import TagPose                        # noqa: E402

HZ = 10.0
STALE_POSE_S = 8.0        # no usable pose for this long -> stop the robot
CRUISE_MS = 0.22          # m/s along a waypoint leg
TURN_RPS = 1.4            # rad/s turning in place
TURN_FIRST = 0.55         # rad of bearing error above which we turn before driving


class LiveLoop(threading.Thread):
    def __init__(self, fleet, mapsvc, goal=None, tags=True):
        super().__init__(daemon=True)
        self.fleet, self.mapsvc, self.goal = fleet, mapsvc, goal
        self.stop_flag = threading.Event()
        self.dr = {n: DeadReckoner() for n in fleet.links}
        # AprilTags are the pose source. One detector, shared: detection is
        # stateless and constructing it per robot would just duplicate the
        # intrinsics. None disables tags and leaves dead reckoning alone.
        self.tags = TagPose() if tags else None
        self.box = None          # the pushable box, from BOX_TAG_ID. Measured.
        self.fixes = {n: 0 for n in fleet.links}
        self.sim = None
        self.grid = None
        self._map_stamp = None
        self.ticks = 0
        self.last_error = ""
        self.mode = {n: "wait" for n in fleet.links}

    # --- map handoff ---------------------------------------------------------
    def reload_map(self):
        p = self.mapsvc.map_path
        if not p or not os.path.exists(p):
            return False
        stamp = os.path.getmtime(p)
        if stamp == self._map_stamp:
            return False
        self.grid = G.Grid(p)
        self._map_stamp = stamp
        if self.goal is None:
            self.goal = self._pick_goal()
        robots = [CL.Robot(n, *self.dr[n].pose[:2]) for n in self.fleet.links]
        self.sim = CL.Sim(self.grid, robots, self.goal)
        return True

    def _pick_goal(self):
        """Farthest drivable cell from the fleet. A real mission sets this; without
        one, drive somewhere that is actually reachable rather than nowhere."""
        d = rf.wavefront(self.grid.drivable(),
                         [self.grid.to_cell(*self.dr[list(self.dr)[0]].pose[:2])])
        import numpy as np
        finite = np.isfinite(d)
        if not finite.any():
            return self.grid.to_world(self.grid.h // 2, self.grid.w // 2)
        j, i = np.unravel_index(np.where(finite, d, -1).argmax(), d.shape)
        x, y = self.grid.to_world(j, i)
        return (float(x), float(y))      # plain floats: this is printed and logged

    # --- the loop ------------------------------------------------------------
    def run(self):
        dt = 1.0 / HZ
        try:
            while not self.stop_flag.is_set():
                t0 = time.time()
                try:
                    self.tick(dt)
                except Exception as e:
                    self.last_error = f"{type(e).__name__}: {e}"
                    self.fleet.stop_all()
                self.ticks += 1
                time.sleep(max(0.0, dt - (time.time() - t0)))
        finally:
            self.fleet.stop_all()

    def sense(self):
        """Look for tags in each robot's newest frame and correct its pose.

        A fix is the only measurement in this system. Everything else -- the
        waypoints, the push plan, what the dashboard draws -- is downstream of it.
        """
        if not self.tags:
            return
        for name, (frame, age) in self.fleet.frames(max_age=1.0).items():
            dr = self.dr.get(name)
            if dr is None:
                continue
            fix = self.tags.fix(frame)
            if fix is None:
                continue
            dr.correct(fix["x"], fix["y"], fix["theta"])
            self.fixes[name] += 1
            # The box carries its own tag, so its pose is measured rather than
            # predicted. docs/push-planning.md: where a shoved box ends up depends
            # on friction across its contact patch, which no model recovers.
            b = self.tags.box(frame, dr.pose)
            if b:
                self.box = b

    def tick(self, dt):
        self.reload_map()

        self.sense()

        if self.sim is None:
            self.fleet.stop_all()
            return

        for r in self.sim.robots:
            dr = self.dr[r.name]
            r.pose = CL.Pose(dr.x, dr.y, dr.theta)

        self.sim.tick(dt)

        for r in self.sim.robots:
            dr = self.dr[r.name]
            link = self.fleet.links.get(r.name)
            if link is None:
                continue
            if not link.online:
                self.mode[r.name] = "offline"
                continue
            if dr.fixed and time.time() - dr.fixed_at > STALE_POSE_S \
               and dr.confidence() <= 0.0:
                self.mode[r.name] = "stale-pose"
                self.fleet.drive(r.name, 0, 0)
                continue
            left, right = self._tracks(r)
            self.mode[r.name] = r.task[0] if r.task else "idle"
            self.fleet.drive(r.name, left, right)
            dr.command(left, right)

    def _tracks(self, r):
        """Waypoint -> left and right duty.

        Two regimes, as in central/motion.py: turn in place when the bearing error
        is large, otherwise drive with proportional steering. Arcing toward a
        waypoint you are not facing sweeps past it and the robot circles instead of
        arriving.

        The twist is converted to duty through the measured motor model, so
        CRUISE_MS really is metres per second rather than a duty number that
        happened to look right.
        """
        if not r.route:
            return 0, 0
        wx, wy = r.route[0]
        err = wrap_angle(math.atan2(wy - r.pose.y, wx - r.pose.x) - r.pose.theta)
        if abs(err) > TURN_FIRST:
            return tracks_for(0.0, TURN_RPS if err > 0 else -TURN_RPS)
        return tracks_for(CRUISE_MS * max(0.3, math.cos(err)),
                          max(-TURN_RPS, min(TURN_RPS, err * 2.0)))

    def status(self):
        return dict(ticks=self.ticks, goal=self.goal, mode=dict(self.mode),
                    error=self.last_error, box=self.box,
                    tags=(self.tags.stats() if self.tags else None),
                    fixes=dict(self.fixes),
                    poses={n: [round(d.x, 2), round(d.y, 2), round(d.theta, 2),
                               round(d.confidence(), 2)] for n, d in self.dr.items()})
