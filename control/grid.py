"""The occupancy grid, loaded from a stage-4 map.

Three states, exactly as docs/reach-field.md specifies:

    UNKNOWN   nobody has seen it. NOT the same as blocked.
    FREE      observed, and a chassis fits.
    BLOCKED   observed, and something is there.

Plus one extra bit the map carries: a BLOCKED cell may be PUSHABLE. That is the
whole point of the white-carton rule in stage 4 — a pushable blocker is one the
solver can emit a clear(edge) for, because deleting it from the grid and re-flooding
the field is a cheap test of "would this route open if the box moved".

Coordinates: world (x, y) in MODEL UNITS, matching central/geometry.py's Pose, where
y is the second ground-plane axis. The map calls that axis z; it is the same axis.
"""
import base64, json
import numpy as np

UNKNOWN, FREE, BLOCKED = 0, 1, 2


class Grid:
    def __init__(self, path):
        m = json.load(open(path))
        raw = np.frombuffer(base64.b64decode(m["grid"]), np.uint8).reshape(m["h"], m["w"])
        # stage 4 codes 0 unknown / 1 free / 2 static / 3 box
        self.state = np.where(raw == 0, UNKNOWN, np.where(raw == 1, FREE, BLOCKED)).astype(np.uint8)
        self.pushable = (raw == 3)
        self.static = (raw == 2)
        self.cell = m["cell"]
        self.origin = np.array(m["origin"], float)
        self.h, self.w = self.state.shape
        self.R = m["R"]
        self.boxes = m["boxes"]
        self.meta = {k: m[k] for k in ("units", "rule", "counts", "floor_level")}

    # --- coordinate helpers -------------------------------------------------
    def to_cell(self, x, y):
        return (int((y - self.origin[1]) / self.cell), int((x - self.origin[0]) / self.cell))

    def to_world(self, j, i):
        return (self.origin[0] + (i + .5) * self.cell, self.origin[1] + (j + .5) * self.cell)

    def inside(self, j, i):
        return 0 <= j < self.h and 0 <= i < self.w

    def at(self, x, y):
        j, i = self.to_cell(x, y)
        return self.state[j, i] if self.inside(j, i) else UNKNOWN

    # --- what a chassis can stand on ---------------------------------------
    def passable(self, optimistic=True):
        """Cells a robot may occupy.

        optimistic=True treats UNKNOWN as passable. That is deliberate: UNKNOWN is
        not blocked, and a sparse reconstruction leaves most of the floor unseen. A
        pessimistic grid would strand every robot in the few cells that happened to
        get points.
        """
        ok = self.state == FREE
        return ok | (self.state == UNKNOWN) if optimistic else ok

    def drivable(self, chassis_cells=1):
        """Cells whose CENTRE a robot may occupy: passable, eroded by the chassis.

        Everything -- planning, the wavefront, frontier choice and scenario setup --
        must agree on this one mask. When they disagree you get goals that no robot
        can stand on and routes that graze corners the body will not clear.
        """
        return self.inflate(self.passable(), chassis_cells)

    def inflate(self, mask, radius_cells):
        """Grow blocked cells by the chassis radius so the robot is a point."""
        if radius_cells <= 0:
            return mask
        out = mask.copy()
        k = int(np.ceil(radius_cells))
        for dj in range(-k, k + 1):
            for di in range(-k, k + 1):
                if dj * dj + di * di > radius_cells ** 2:
                    continue
                out &= np.roll(np.roll(mask, dj, 0), di, 1)
        return out
