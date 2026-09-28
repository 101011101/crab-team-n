"""Stage 4 — flatten the 3D world into a 2D map the control loop can drive on.

    3D point cloud  ->  occupancy grid  +  a list of movable boxes

WHAT EACH CELL MEANS
    0 UNKNOWN   nothing was reconstructed here. Not free; just unseen.
    1 FREE      floor points landed here, so the robot has actually seen ground.
    2 STATIC    something stands here that the robot must go around: walls,
                counters, the fridge.
    3 BOX       something stands here that the robot may push.

HOW A BOX IS TOLD FROM A WALL
    Two tests, both needed, because either alone is wrong.
      colour  the boxes in this footage are white/cream shipping cartons against a
              green-and-grey lino floor and yellow walls. Above-floor points that are
              bright and unsaturated are box candidates. Measured on the joint cloud:
              34.7% of above-floor points pass, so this is a real population and not
              a sliver.
      size    a wall is also white. A wall is not a box. So a connected blob of white
              cells only becomes a box if its footprint fits inside MAX_BOX_CELLS on
              its shorter side. Anything longer is left STATIC.

    This colour rule is tuned to THIS room and is the weakest link in the chain. A
    white wall in a different room reads as a box until the rule is replaced. See
    the README's "what would break this" section.

HEIGHT BANDS
    Everything is relative to the modal height, which is the floor: it is the single
    most-sampled surface in the room.
      below floor + FLOOR_BAND      floor, marks the cell FREE
      up to floor + CEILING_CUT     obstacle, the band a robot can collide with
      above that                    ignored: ceiling, lights, tops of shelving

Distances are MODEL UNITS. Monocular SfM has no metre. Cell size is set as a
fraction of the camera-path radius R so the grid scales with the room.
"""
import argparse, base64, json, os
import numpy as np
import cv2
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import world_io

FLOOR_BAND = 0.08     # x R. floor points lie within this of the modal height
CEILING_CUT = 0.75    # x R. above this we stop caring; it cannot be driven into
CELL = 1 / 20         # x R. one grid cell. A SPARSE cloud cannot support a fine
                      # grid: at R/30 most cells hold no point at all and the map
                      # reads as noise. R/12 is the finest that stays connected.
MAP_RADII = 3.0       # x R. points past this are far-field outliers, not room
MIN_HITS = 1          # points needed before a cell counts as occupied
CLOSE = 2             # morphological close, in cells. Sparse points fragment one
                      # box into several blobs; closing puts them back together.
WHITE_VAL = 0.70      # HSV value at or above this is "bright"
WHITE_SAT = 0.18      # HSV saturation at or below this is "not a colour"
TILE_CELLS = 5        # pushable area is cut into tiles this many cells on a side
MIN_BOX_CELLS = 4     # a tile needs this many observed white cells to be a unit

UNKNOWN, FREE, STATIC, BOX = 0, 1, 2, 3


def build(w, cell_frac=CELL, floor_band=FLOOR_BAND, ceiling=CEILING_CUT,
          white_val=WHITE_VAL, white_sat=WHITE_SAT,
          min_box=MIN_BOX_CELLS, close=CLOSE, tile=TILE_CELLS):
    cell = cell_frac * w.R
    floor = w.floor_level()
    # keep the room, drop the far field: the 5R point trim upstream is deliberately
    # generous, and everything past 3R here is stray triangulation, not geometry.
    near = np.linalg.norm(w.P[:, [0, 2]] - w.cam[:, [0, 2]].mean(0), axis=1) < MAP_RADII * w.R
    P, Cc = w.P[near], w.C[near]
    Y = P[:, 1]
    is_floor = Y < floor + floor_band * w.R
    is_obst = (~is_floor) & (Y < floor + ceiling * w.R)

    XZ = P[:, [0, 2]]
    lo = XZ.min(0) - cell
    ij = np.floor((XZ - lo) / cell).astype(int)
    W, H = ij[:, 0].max() + 2, ij[:, 1].max() + 2

    grid = np.zeros((H, W), np.uint8)
    hits = np.zeros((H, W), np.int32)

    # floor cells -> FREE
    fj, fi = ij[is_floor, 1], ij[is_floor, 0]
    np.add.at(hits, (fj, fi), 1)
    grid[hits >= MIN_HITS] = FREE

    # obstacle cells -> STATIC, and accumulate colour so we can ask "is it white"
    hits[:] = 0
    oj, oi = ij[is_obst, 1], ij[is_obst, 0]
    np.add.at(hits, (oj, oi), 1)
    rgb = Cc[is_obst] / 255.0
    mx, mn = rgb.max(1), rgb.min(1)
    val = mx
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    sv = np.zeros((H, W)); ss = np.zeros((H, W))
    np.add.at(sv, (oj, oi), val)
    np.add.at(ss, (oj, oi), sat)
    occ_raw = hits >= MIN_HITS
    occ = occ_raw
    if close:
        k = np.ones((close, close), np.uint8)
        occ = cv2.morphologyEx(occ.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    grid[occ] = STATIC
    with np.errstate(invalid="ignore"):
        mval = np.where(occ, sv / np.maximum(hits, 1), 0.0)
        msat = np.where(occ, ss / np.maximum(hits, 1), 1.0)

    # WHITE CELLS ARE PUSHABLE.
    # Colour is judged only on cells that actually hold points; closing bridges the
    # gaps a sparse cloud leaves inside one carton but never votes on colour.
    white_raw = occ_raw & (mval >= white_val) & (msat <= white_sat)
    blob = white_raw.astype(np.uint8)
    if close:
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    blob = blob.astype(bool) & occ
    grid[blob] = BOX

    # The cartons in this room are stacked shoulder to shoulder, so the pushable
    # region is ONE connected run 58 x 48 cells across, not a scatter of separate
    # objects. Connected components cannot split it and neither can a human eye from
    # a sparse cloud. So the region is TILED into box-sized units: each tile is a
    # thing the simulator can push. A tile is a unit of pushable area, NOT a detected
    # carton, and the README says so.
    boxes = []
    for ty in range(0, H, tile):
        for tx in range(0, W, tile):
            sub = white_raw[ty:ty + tile, tx:tx + tile]
            if sub.sum() < min_box:
                continue
            ys, xs = np.where(sub)
            cyc, cxc = ys.mean() + ty, xs.mean() + tx
            boxes.append(dict(
                id=f"box{len(boxes)+1}",
                cx=round(float(lo[0] + (cxc + .5) * cell), 4),
                cz=round(float(lo[1] + (cyc + .5) * cell), 4),
                w=round(float((xs.max() - xs.min() + 1) * cell), 4),
                h=round(float((ys.max() - ys.min() + 1) * cell), 4),
                yaw=0.0,
                cells=int(sub.sum()),
                value=round(float(mval[ty:ty + tile, tx:tx + tile][sub].mean()), 3),
                sat=round(float(msat[ty:ty + tile, tx:tx + tile][sub].mean()), 3)))

    # crop to what was actually mapped: an all-unknown border is not information
    nz = np.argwhere(grid != UNKNOWN)
    (y0, x0), (y1, x1) = nz.min(0) - 2, nz.max(0) + 3
    y0, x0 = max(y0, 0), max(x0, 0)
    grid = grid[y0:y1, x0:x1]
    lo = lo + np.array([x0, y0]) * cell
    H, W = grid.shape

    return dict(
        cell=round(float(cell), 5), R=round(w.R, 4),
        origin=[round(float(lo[0]), 4), round(float(lo[1]), 4)],
        w=int(W), h=int(H),
        floor_level=round(float(floor), 4),
        grid=base64.b64encode(grid.tobytes()).decode(),
        boxes=boxes,
        counts={n: int((grid == v).sum()) for n, v in
                (("unknown", UNKNOWN), ("free", FREE), ("static", STATIC), ("box", BOX))},
        units="model units; monocular SfM has no metre. cell is R/30.",
        rule=dict(white_val=white_val, white_sat=white_sat,
                  tile_cells=tile, min_box_cells=min_box,
                  floor_band_R=floor_band, ceiling_R=ceiling)), grid


def preview(grid, boxes, m, path):
    """PNG so a teammate can see the map without running anything."""
    pal = np.array([[28, 32, 36], [214, 226, 222], [92, 108, 116], [232, 168, 60]], np.uint8)
    img = pal[grid]
    img = cv2.resize(img, (grid.shape[1] * 4, grid.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    for b in boxes:
        x = int((b["cx"] - m["origin"][0]) / m["cell"] * 4)
        y = int((b["cz"] - m["origin"][1]) / m["cell"] * 4)
        cv2.drawMarker(img, (x, y), (60, 90, 230), cv2.MARKER_CROSS, 11, 2)
    cv2.imwrite(path, cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sparse", help="COLMAP sparse/ directory holding the text model")
    ap.add_argument("--out", default="map2d.json")
    ap.add_argument("--png", default=None, help="also write a preview image")
    ap.add_argument("--cell", type=float, default=CELL, help="cell size as a fraction of R")
    ap.add_argument("--tile", type=int, default=TILE_CELLS,
                    help="pushable tiles this many cells on a side")
    a = ap.parse_args()

    w = world_io.load(a.sparse)
    m, grid = build(w, cell_frac=a.cell, tile=a.tile)
    json.dump(m, open(a.out, "w"), separators=(",", ":"))
    print(f"model       {w.model}")
    print(f"grid        {m['w']} x {m['h']} cells of {m['cell']} model units")
    print(f"cells       " + "  ".join(f"{k}={v}" for k, v in m["counts"].items()))
    print(f"boxes       {len(m['boxes'])} movable")
    for b in m["boxes"][:6]:
        print(f"  {b['id']:6} at ({b['cx']:7.2f},{b['cz']:7.2f})  "
              f"{b['w']:.2f} x {b['h']:.2f}  {b['cells']:3} cells  value={b['value']}")
    if len(m["boxes"]) > 6:
        print(f"  ... and {len(m['boxes'])-6} more")
    print(f"wrote       {a.out}")
    if a.png:
        preview(grid, m["boxes"], m, a.png)
        print(f"wrote       {a.png}")


if __name__ == "__main__":
    main()
