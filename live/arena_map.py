"""The arena as a metric occupancy grid, straight from pipeline/common/constants.py.

WHY THIS EXISTS, AND WHY IT IS NOT THE SfM MAP

Pose now comes from AprilTags, which are solved in the ROOM frame: metres, origin at
the room's bottom-left corner. The SfM map from pipeline/ is in MODEL UNITS with an
arbitrary origin, scale and orientation.

Those two frames can only be related through something visible in both. There is
nothing: all 183 frames of IMG_7373/4/5 were scanned for tag36h11 and **not one tag
appears**. They are a kitchen full of cartons; the tag map describes the 6 x 4 m
arena. Two different rooms.

So a pose in metres and a map in model units cannot be used together, and pretending
otherwise would put robots at confident, wrong positions. This module supplies the
other half of a coherent metric system: the arena, whose geometry is already known
and already written down.

WHAT THIS BUYS, BEYOND CONSISTENCY

Every threshold in control/push.py was written as a bare number in "model units":
chassis 0.09, margin 0.04, push increment 0.13. In metres those are a 9 cm chassis
radius and a 13 cm push -- and docs/push-planning.md specifies pushes of "10 to 15
cm". The constants were always metric; they just had no frame to be metric in.

TO USE THE SfM MAP INSTEAD: shoot the reconstruction footage in the arena with the
wall tags in shot. Tags then appear in the model, and the 7-DoF similarity that
aligns SfM to the room is a lookup rather than a fit -- exactly the argument in
docs/robot1-reconstruction.md section 3.
"""
import argparse, base64, json, math, os, sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from pipeline.common import constants as C          # noqa: E402

UNKNOWN, FREE, STATIC, BOX = 0, 1, 2, 3
CELL_M = 0.05          # docs/reach-field.md: about 5 cm per cell
WALL_M = 0.10          # drawn thickness of the room boundary


def build(cell=CELL_M, known=True):
    """The arena grid.

    known=True marks the floor FREE, because the room's shape is given, not
    discovered. Set it False to make the robots explore a room they have been told
    nothing about -- the reach field then drives exploration off the frontier, which
    is the behaviour docs/reach-field.md describes.
    """
    W = int(round(C.ROOM_W / cell)) + 2
    H = int(round(C.ROOM_H / cell)) + 2
    grid = np.full((H, W), UNKNOWN, np.uint8)
    origin = (-cell, -cell)        # one cell of wall margin outside the room

    def cells(x, y, w, h):
        i0 = max(0, int(math.floor((x - origin[0]) / cell)))
        j0 = max(0, int(math.floor((y - origin[1]) / cell)))
        i1 = min(W, int(math.ceil((x + w - origin[0]) / cell)))
        j1 = min(H, int(math.ceil((y + h - origin[1]) / cell)))
        return slice(j0, j1), slice(i0, i1)

    if known:
        grid[cells(0, 0, C.ROOM_W, C.ROOM_H)] = FREE

    # room boundary
    for x, y, w, h in ((-WALL_M, -WALL_M, C.ROOM_W + 2 * WALL_M, WALL_M),
                       (-WALL_M, C.ROOM_H, C.ROOM_W + 2 * WALL_M, WALL_M),
                       (-WALL_M, -WALL_M, WALL_M, C.ROOM_H + 2 * WALL_M),
                       (C.ROOM_W, -WALL_M, WALL_M, C.ROOM_H + 2 * WALL_M)):
        grid[cells(x, y, w, h)] = STATIC

    for x, y, w, h, _label in C.OBSTACLES:
        grid[cells(x, y, w, h)] = STATIC

    return grid, cell, origin, W, H


def to_map(grid, cell, origin, W, H, box=None):
    """Serialise in the stage-4 map schema, so control/ reads it unchanged."""
    boxes = []
    if box is not None:
        s = box.get("size", C.BOX_SIZE)
        boxes.append(dict(id="box", cx=round(float(box["x"]), 4),
                          cz=round(float(box["y"]), 4),
                          w=round(s, 4), h=round(s, 4), yaw=0.0,
                          cells=int(round((s / cell) ** 2)),
                          value=1.0, sat=0.0, source="tag%d" % C.BOX_TAG_ID))
        g = grid.copy()
        i = int((box["x"] - origin[0]) / cell)
        j = int((box["y"] - origin[1]) / cell)
        r = max(1, int(s / 2 / cell))
        g[max(0, j - r):j + r + 1, max(0, i - r):i + r + 1] = BOX
        grid = g

    return dict(
        cell=round(cell, 5),
        R=round(C.ROOM_W / 2, 4),
        origin=[round(origin[0], 4), round(origin[1], 4)],
        w=int(W), h=int(H), floor_level=0.0,
        grid=base64.b64encode(grid.tobytes()).decode(),
        boxes=boxes,
        counts={n: int((grid == v).sum()) for n, v in
                (("unknown", UNKNOWN), ("free", FREE), ("static", STATIC), ("box", BOX))},
        units="METRES. Room frame: x right, y up, origin at the room's bottom-left "
              "corner. Unlike a stage-4 SfM map, these are real metres -- the scale "
              "comes from the printed tag size, not from a reconstruction.",
        frame="room",
        rule=dict(source="pipeline/common/constants.py",
                  room=[C.ROOM_W, C.ROOM_H], cell_m=cell,
                  obstacles=len(C.OBSTACLES), box_tag=C.BOX_TAG_ID))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="arena_map.json")
    ap.add_argument("--cell", type=float, default=CELL_M)
    ap.add_argument("--unknown", action="store_true",
                    help="start with the floor UNKNOWN so the robots must explore it")
    a = ap.parse_args()
    grid, cell, origin, W, H = build(a.cell, known=not a.unknown)
    m = to_map(grid, cell, origin, W, H)
    json.dump(m, open(a.out, "w"), separators=(",", ":"))
    print(f"arena {C.ROOM_W} x {C.ROOM_H} m -> {W} x {H} cells of {cell} m")
    print("cells  " + "  ".join(f"{k}={v}" for k, v in m["counts"].items()))
    print(f"obstacles from constants.py: {len(C.OBSTACLES)}")
    for x, y, w, h, lab in C.OBSTACLES:
        print(f"   {lab:16} at ({x}, {y})  {w} x {h} m")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
