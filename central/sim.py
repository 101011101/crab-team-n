"""Simulated arena and robots.

Exists so the planner, the allocator and the motion controller can be run
end to end with no hardware, no network and no vision. Everything above
this file sees the same interfaces the real system exposes: a Pose in, two
track speeds out.

The simulated robot is deliberately only as good as the real one. It
accepts track speeds, it stops itself when commands go stale, and it has
no idea where it is. If a bug needs the robot to know its own position to
reproduce, it is not a bug the real system can have.
"""

import math

from config import DRIVE, MOTION
from geometry import Pose


class SimRobot:
    """Differential drive integrated from track speeds.

    Track speeds are the same -4095..4095 wire units the ESP32 takes, so a
    controller cannot tell this apart from the real robot by its interface.
    """

    def __init__(self, serial: int, pose: Pose, motion=MOTION):
        self.serial = serial
        self.pose = pose
        self.motion = motion

        self.left = 0
        self.right = 0

        self.alive = True
        self.distance_travelled = 0.0

        # Mirrors the firmware watchdog. A controller that forgets to
        # resend must fail here exactly as it would on the bench.
        self.last_command_at = 0.0
        self.timeout_s = 0.5

    def command(self, left: int, right: int, now: float):
        self.left = int(left)
        self.right = int(right)
        self.last_command_at = now

    def step(self, dt: float, now: float):
        if not self.alive:
            self.left = self.right = 0
            return

        if now - self.last_command_at > self.timeout_s:
            self.left = self.right = 0
            return

        # Wire units are a fraction of full scale, scaled by top speed.
        vl = (self.left / 4095.0) * self.motion.max_speed_mps
        vr = (self.right / 4095.0) * self.motion.max_speed_mps

        v = (vl + vr) / 2.0
        w = (vr - vl) / self.motion.track_width_m

        theta = self.pose.theta + w * dt

        # Integrate at the midpoint heading. Using the start heading makes
        # a turning robot cut the corner, which flatters the controller.
        mid = self.pose.theta + w * dt / 2.0

        x = self.pose.x + v * math.cos(mid) * dt
        y = self.pose.y + v * math.sin(mid) * dt

        self.distance_travelled += abs(v) * dt
        self.pose = Pose(x, y, math.atan2(math.sin(theta), math.cos(theta)))

    def kill(self):
        """Simulate a dead battery or a lost link."""
        self.alive = False
        self.left = self.right = 0


class Arena:
    """Static geometry plus the things in it.

    Walls are segments. Collision is a stop, not a bounce: a toy car that
    hits a wall stalls against it, and a controller that relies on
    bouncing off geometry is relying on something the real arena will not
    do.
    """

    def __init__(self):
        self.walls = []          # (x1, y1, x2, y2)
        self.blocks = []         # dicts: x, y, cleared, plus either radius or w/h/angle
        self.species = []        # dicts: x, y, colour, found_by

    def add_wall(self, x1, y1, x2, y2):
        self.walls.append((x1, y1, x2, y2))

    def add_block(self, x, y, radius=0.08):
        self.blocks.append({"shape": "circle", "x": x, "y": y,
                            "radius": radius, "cleared": False})
        return self.blocks[-1]

    def add_box(self, x, y, w, h=None, angle=0.0):
        """A rectangular block, centred at (x, y), `angle` radians CCW.

        A real obstacle is a box, and the difference is not cosmetic: a
        circle of the radius needed to seal a corridor is far larger than
        the corridor is wide, so it cannot physically be in there. A box
        seals the same gap at its true width.
        """
        self.blocks.append({"shape": "box", "x": x, "y": y,
                            "w": w, "h": w if h is None else h,
                            "angle": angle, "cleared": False})
        return self.blocks[-1]

    def add_species(self, x, y, colour):
        self.species.append({"x": x, "y": y, "colour": colour, "found_by": None})
        return self.species[-1]

    def blocked_at(self, x, y, radius=0.09):
        """True when a robot body at (x, y) overlaps a wall or a live block."""
        for block in self.blocks:
            if block["cleared"]:
                continue
            if _block_distance(x, y, block) < radius:
                return True

        for wall in self.walls:
            if _segment_distance(x, y, *wall) < radius:
                return True

        return False


def _block_distance(px, py, block):
    """Distance from a point to a block's surface, 0 inside it."""
    if block.get("shape", "circle") == "circle":
        return max(0.0, math.hypot(px - block["x"], py - block["y"]) - block["radius"])

    # Rotate the point into the box's own frame, where the box is axis
    # aligned and the nearest-point test is a clamp.
    ca, sa = math.cos(-block["angle"]), math.sin(-block["angle"])
    dx, dy = px - block["x"], py - block["y"]
    lx, ly = dx * ca - dy * sa, dx * sa + dy * ca

    ox = abs(lx) - block["w"] / 2.0
    oy = abs(ly) - block["h"] / 2.0

    if ox <= 0 and oy <= 0:
        return 0.0
    return math.hypot(max(ox, 0.0), max(oy, 0.0))


def _segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy

    if length_sq < 1e-12:
        return math.hypot(px - x1, py - y1)

    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))

    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def two_room_arena():
    """The demo arena: two rooms, one corridor, one block, species behind it.

    Coordinates in metres. Room 1 is where the robots start. The corridor
    is blocked, so room 2 and every species in it is unreachable until a
    robot clears the block.
    """
    arena = Arena()

    # The corridor is 0.40 m wide. An earlier 0.20 m corridor left a 0.18 m
    # robot one centimetre of clearance per side and wedged both crabs, which
    # is a property of the arena, not of the controller. Build the real one
    # at least twice the chassis width.

    # Room 1: 0..2 x 0..2, opening on the right at y 0.8..1.2
    arena.add_wall(0.0, 0.0, 2.0, 0.0)
    arena.add_wall(0.0, 2.0, 2.0, 2.0)
    arena.add_wall(0.0, 0.0, 0.0, 2.0)
    arena.add_wall(2.0, 0.0, 2.0, 0.8)
    arena.add_wall(2.0, 1.2, 2.0, 2.0)

    # Corridor: 2..3 x 0.8..1.2
    arena.add_wall(2.0, 0.8, 3.0, 0.8)
    arena.add_wall(2.0, 1.2, 3.0, 1.2)

    # Room 2: 3..5 x 0..2, opening on the left at y 0.8..1.2
    arena.add_wall(3.0, 0.0, 5.0, 0.0)
    arena.add_wall(3.0, 2.0, 5.0, 2.0)
    arena.add_wall(5.0, 0.0, 5.0, 2.0)
    arena.add_wall(3.0, 0.0, 3.0, 0.8)
    arena.add_wall(3.0, 1.2, 3.0, 2.0)

    arena.add_block(2.5, 1.0)

    # Species live behind the blockage, so exploration, collaboration and
    # recruitment become one chain rather than three separate set pieces.
    arena.add_species(3.6, 1.6, "green")
    arena.add_species(4.4, 0.5, "red")
    arena.add_species(4.6, 1.5, "blue")
    arena.add_species(3.8, 0.4, "green")

    return arena
