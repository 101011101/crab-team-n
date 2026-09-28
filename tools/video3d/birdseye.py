"""Collapse the reconstructed corridor into a bird's-eye map a robot can plan on.

A point cloud is not a plan. A planner wants to know three things about a
square of floor: can I stand here, is something in the way, or do I simply
not know. That last state is the one a naive projection throws away -- a
cell with no points in it is not free, it is unobserved, and driving into
it is a gamble.

The chain is:

    ground plane  ->  height above it  ->  class  ->  grid cell  ->  label

Everything hangs off the ground plane, so it is fitted twice and the two
fits are compared. The cloud says where the flat dominant surface is. The
camera trajectory says the same thing independently: a phone carried down
a corridor stays a roughly fixed distance above the floor, so the plane
that makes the camera height constant is the floor. When the two agree,
the plane is real; when they do not, everything downstream is suspect and
the script says so rather than drawing a confident wrong map.

Free space comes from two sources that answer different questions. Floor
points say "something flat was seen here". The camera path says "a person
physically stood here", which is stronger evidence than any amount of
geometry, and ray casting from each pose spreads that certainty out to
the first thing that blocked the view.

Reads the cloud and the COLMAP model, writes a PNG, a JSON, and the JS
data file the viewer page loads.
"""
import base64
import json
import math
import os
import struct
import sys
import time

import cv2
import numpy as np


# --- tuning ------------------------------------------------------------
CELL = 0.04                # metres per grid cell
FLOOR_BAND = (-0.08, 0.10)  # height range counted as floor
OBSTACLE_BAND = (0.10, 1.80)  # above this is ceiling/signage, not in the way
MIN_PTS = 3                # points needed before a cell asserts anything
OCC_MIN_HEIGHT = 0.20      # tallest point in a cell before it blocks the robot
CARVE_RANGE = 6.0          # metres of camera ray trusted for free space
DOOR_WIDTH = (0.60, 2.00)  # a gap in a wall run this wide is an opening


def qR(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                     [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])


# --- input -------------------------------------------------------------
PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
             "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
             "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
             "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}


def read_ply(path):
    """Vertex positions and colours from a binary PLY.

    Written out rather than pulled from open3d so the map can be built in
    the light venv: the cloud and the Poisson mesh declare their vertices
    with different property types, and only the header knows which.
    """
    f = open(path, "rb")
    hdr = b""
    while not hdr.endswith(b"end_header\n"):
        c = f.read(1)
        if not c:
            raise ValueError(f"{path}: no PLY header")
        hdr += c
    lines = hdr.decode("ascii", "replace").splitlines()
    if "binary_little_endian" not in lines[1]:
        raise ValueError(f"{path}: only binary_little_endian is handled")

    fields, count, in_vertex = [], 0, False
    for ln in lines:
        w = ln.split()
        if not w:
            continue
        if w[0] == "element":
            in_vertex = w[1] == "vertex"
            if in_vertex:
                count = int(w[2])
        elif w[0] == "property" and in_vertex:
            if w[1] == "list":
                raise ValueError("list property inside vertex element")
            fields.append((w[2], PLY_TYPES[w[1]]))

    raw = np.frombuffer(f.read(np.dtype(fields).itemsize * count),
                        dtype=np.dtype(fields), count=count)
    f.close()
    P = np.stack([raw["x"], raw["y"], raw["z"]], 1).astype(np.float64)
    if "red" in raw.dtype.names:
        C = np.stack([raw["red"], raw["green"], raw["blue"]], 1)
        if C.dtype != np.uint8:  # mesh colours may be float 0..1
            C = (np.clip(C, 0, 1) * 255).astype(np.uint8)
    else:
        C = np.full((len(P), 3), 160, np.uint8)
    return P, C


def read_colmap(sfm):
    """Camera centres and rotations, ordered along the walk.

    Same binary layout as fuse_colmap_depth.py reads; the 2D observations
    are skipped because the map only needs where the camera was and which
    way it faced.
    """
    with open(f"{sfm}/cameras.bin", "rb") as f:
        struct.unpack("<Q", f.read(8))
        _, _, W, H = struct.unpack("<iiQQ", f.read(24))
        fpx, cx, cy, _ = struct.unpack("<4d", f.read(32))

    views = []
    with open(f"{sfm}/images.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)
            q = np.array(struct.unpack("<4d", f.read(32)))
            t = np.array(struct.unpack("<3d", f.read(24)))
            f.read(4)
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            m = struct.unpack("<Q", f.read(8))[0]
            f.read(m * 24)
            R = qR(q)
            views.append((name.decode(), R, -R.T @ t))
    views.sort(key=lambda v: v[0])
    C = np.array([v[2] for v in views])
    R = np.array([v[1] for v in views])
    hfov = 2.0 * math.atan(0.5 * W / fpx)
    return C, R, hfov


# --- ground plane ------------------------------------------------------
def trajectory_up(C, R):
    """The up direction implied by the walk, before the cloud is consulted.

    The camera's own down axis is not gravity: this phone was carried
    pitched about 20 deg toward the floor, so using it directly tilts the
    whole map. What is reliable is that the operator walked on the level.
    Taking the camera down axis and removing whatever part of it points
    along the direction of travel leaves a vector perpendicular to the
    walk -- gravity, if the walk was level.
    """
    down = np.array([Ri.T @ np.array([0.0, 1.0, 0.0]) for Ri in R])
    down = down.mean(0)
    down /= np.linalg.norm(down)

    step = np.diff(C, axis=0)
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    travel = step.mean(0)
    travel /= np.linalg.norm(travel)

    g = down - (down @ travel) * travel
    g /= np.linalg.norm(g)
    return -g, travel, math.degrees(math.acos(abs(down @ g)))


def fit_plane(pts, thresh, iters=3000, rng=None):
    """Plain RANSAC plane. Returns normal, offset, inlier mask."""
    rng = rng or np.random.default_rng(0)
    sample = pts if len(pts) < 200_000 else pts[rng.choice(len(pts), 200_000, replace=False)]
    best_n, best_d, best_k = None, 0.0, -1
    for _ in range(iters):
        a, b, c = sample[rng.integers(0, len(sample), 3)]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm /= ln
        d = -nrm @ a
        k = int((np.abs(sample @ nrm + d) < thresh).sum())
        if k > best_k:
            best_n, best_d, best_k = nrm, d, k

    # One least-squares pass over the winning inliers: three random points
    # fix the plane's orientation only as well as their own noise allows.
    for _ in range(2):
        m = np.abs(pts @ best_n + best_d) < thresh
        q = pts[m] - pts[m].mean(0)
        best_n = np.linalg.svd(q, full_matrices=False)[2][-1]
        best_d = -best_n @ pts[m].mean(0)
    return best_n, best_d, np.abs(pts @ best_n + best_d) < thresh


def ground_plane(P, C, R, thresh=0.03):
    up_prior, travel, pitch = trajectory_up(C, R)
    h = P @ up_prior
    cam_h = C @ up_prior

    # Fit only below the cameras: the walls and ceiling are also large flat
    # surfaces and a plane fitted to everything can prefer one of them.
    low = P[h < cam_h.mean() - 0.6]
    n, d, inl = fit_plane(low, thresh)
    if n @ up_prior < 0:
        n, d = -n, -d

    height_above = C @ n + d
    tilt = math.degrees(math.acos(np.clip(n @ up_prior, -1, 1)))

    # Horizontal frame: +x down the corridor, +y across it.
    ex = travel - (travel @ n) * n
    ex /= np.linalg.norm(ex)
    ey = np.cross(n, ex)
    return dict(normal=n, offset=d, ex=ex, ey=ey,
                inlier_frac=float(inl.mean()), n_low=len(low),
                cam_pitch_deg=pitch, tilt_vs_traj_deg=tilt,
                cam_height=float(height_above.mean()),
                cam_height_std=float(height_above.std()))


# --- rasterising -------------------------------------------------------
UNKNOWN, FREE, OCCUPIED = 0, 1, 2


def rasterise(X, Y, H, COL, cls, path_xy, cell):
    x0 = min(X.min(), path_xy[:, 0].min()) - 0.6
    y0 = min(Y.min(), path_xy[:, 1].min()) - 0.6
    x1 = max(X.max(), path_xy[:, 0].max()) + 0.6
    y1 = max(Y.max(), path_xy[:, 1].max()) + 0.6
    w = int(math.ceil((x1 - x0) / cell))
    h = int(math.ceil((y1 - y0) / cell))

    ix = np.clip(((X - x0) / cell).astype(np.int32), 0, w - 1)
    iy = np.clip(((Y - y0) / cell).astype(np.int32), 0, h - 1)
    lin = iy * w + ix

    n_floor = np.bincount(lin[cls == 1], minlength=w * h).reshape(h, w)
    n_obst = np.bincount(lin[cls == 2], minlength=w * h).reshape(h, w)

    hmax = np.zeros(w * h)
    np.maximum.at(hmax, lin[cls == 2], H[cls == 2])

    # Colour from everything at or below head height: the ceiling is
    # excluded from the map, so it must not tint the map either.
    vis = cls != 3
    csum = np.zeros((w * h, 3))
    np.add.at(csum, lin[vis], COL[vis].astype(np.float64))
    cnt = np.bincount(lin[vis], minlength=w * h)
    rgb = np.zeros((w * h, 3), np.uint8)
    nz = cnt > 0
    rgb[nz] = np.clip(csum[nz] / cnt[nz, None], 0, 255).astype(np.uint8)

    return dict(w=w, h=h, x0=x0, y0=y0, cell=cell, lin=lin,
                n_floor=n_floor, n_obst=n_obst,
                hmax=hmax.reshape(h, w), rgb=rgb.reshape(h, w, 3))


def tag_fraction(g, cls, tag):
    """Per cell, what share of its obstacle points carry `tag`."""
    sel = cls == 2
    num = np.bincount(g["lin"][sel & tag], minlength=g["w"] * g["h"]).astype(np.float32)
    den = np.bincount(g["lin"][sel], minlength=g["w"] * g["h"]).astype(np.float32)
    return (num / np.maximum(den, 1)).reshape(g["h"], g["w"])


def occupancy(g, path_ij, heading, hfov):
    """Three-state occupancy: unknown is a state, not a default.

    Points alone leave holes wherever the floor was too plain to give the
    depth model anything to lock onto. The camera fills them: every pose
    is a place a person stood, and everything between the camera and the
    first thing that stopped the ray was empty air at the time.
    """
    occ = np.full((g["h"], g["w"]), UNKNOWN, np.uint8)

    # Height gate, not just a point count. Two frames whose floors did not
    # register to each other leave a sheet of carpet a hand's width above
    # the fitted plane, and by class alone that sheet is an obstacle. It
    # runs the length of the corridor and squeezes the measured width to
    # half of what the operator walked through.
    occ[(g["n_obst"] >= MIN_PTS) & (g["hmax"] >= OCC_MIN_HEIGHT)] = OCCUPIED
    occ[(occ == UNKNOWN) & (g["n_floor"] >= MIN_PTS)] = FREE

    blocked = occ == OCCUPIED
    reach = int(CARVE_RANGE / g["cell"])
    carved = 0
    for (ci, cj), th in zip(path_ij, heading):
        for a in np.linspace(th - hfov / 2, th + hfov / 2, 61):
            dx, dy = math.cos(a), math.sin(a)
            for s in range(1, reach):
                i = int(ci + dx * s)
                j = int(cj + dy * s)
                if not (0 <= i < g["w"] and 0 <= j < g["h"]):
                    break
                if blocked[j, i]:
                    break
                if occ[j, i] == UNKNOWN:
                    occ[j, i] = FREE
                    carved += 1

    # The operator's own footprint is the one cell class nothing can argue
    # with, so it is written last and overrides the ray result.
    r = max(1, int(0.18 / g["cell"]))
    for ci, cj in path_ij:
        occ[max(0, cj-r):cj+r+1, max(0, ci-r):ci+r+1] = FREE
    return occ, carved


# --- structure ---------------------------------------------------------
def cross2(u, v):
    return u[0] * v[1] - u[1] * v[0]


def hough_segments(occ, g, min_len):
    img = (occ == OCCUPIED).astype(np.uint8) * 255
    img = cv2.dilate(img, np.ones((3, 3), np.uint8))
    lines = cv2.HoughLinesP(img, 1, np.pi / 360,
                            threshold=int(0.6 * min_len / g["cell"]),
                            minLineLength=int(min_len / g["cell"]),
                            maxLineGap=int(0.20 / g["cell"]))
    if lines is None:
        return []
    out = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        a = np.array([g["x0"] + (x1 + .5) * g["cell"], g["y0"] + (y1 + .5) * g["cell"]])
        b = np.array([g["x0"] + (x2 + .5) * g["cell"], g["y0"] + (y2 + .5) * g["cell"]])
        out.append((a, b))
    return out


def probe_run(occ, g, a, u, t, half=0.22):
    """What lies in the band `half` metres either side of a + u*t.

    Hough draws an ideal line; the wall it came from is a ragged band of
    cells a few centimetres either side of it. Asking only the cell under
    the line would read a solid wall as full of holes.

    Returns (solid, free): solid means a wall cell is there, free means
    the band is open floor. A line that is mostly free is not a wall at
    all -- Hough will happily join two real walls straight across the
    corridor between them, and that line is the one this catches.
    """
    p = a + u * t
    nrm = np.array([-u[1], u[0]])
    k = int(half / g["cell"])
    solid = free = False
    for s in range(-k, k + 1):
        q = p + nrm * (s * g["cell"])
        i = int((q[0] - g["x0"]) / g["cell"])
        j = int((q[1] - g["y0"]) / g["cell"])
        if not (0 <= i < g["w"] and 0 <= j < g["h"]):
            continue
        if occ[j, i] == OCCUPIED:
            solid = True
        elif occ[j, i] == FREE:
            free = True
    return solid, free and not solid


def extract_walls(occ, g, path_xy, min_len=1.2, min_solid=0.55, max_free=0.18):
    """Straight runs of occupied cells, merged into one wall per surface.

    Hough reports a real wall a dozen times over -- once per subset of its
    cells that happened to vote together -- and breaks it at every doorway.
    Both are fixed the same way: segments that share a direction and a
    perpendicular offset are one surface, so they are pooled and refitted,
    and where the pooled run has no occupied cells under it, that is the
    doorway Hough was avoiding.
    """
    segs = hough_segments(occ, g, min_len)
    if not segs:
        return []

    # Cluster on (direction, perpendicular distance from origin). Direction
    # is folded to a half turn because a wall has no preferred end.
    items = []
    for a, b in segs:
        v = b - a
        ln = float(np.linalg.norm(v))
        u = v / ln
        if u[0] < 0 or (u[0] == 0 and u[1] < 0):
            u = -u
        items.append((a, b, ln, u, cross2(u, a)))
    items.sort(key=lambda s: -s[2])

    clusters = []
    for a, b, ln, u, rho in items:
        for c in clusters:
            if u @ c["u"] > 0.990 and abs(rho - c["rho"]) < 0.25:
                c["pts"] += [a, b]
                break
        else:
            clusters.append(dict(pts=[a, b], u=u, rho=rho))

    walls = []
    for c in clusters:
        pts = np.array(c["pts"])
        mid = pts.mean(0)
        u = np.linalg.svd(pts - mid, full_matrices=False)[2][0]
        t = (pts - mid) @ u

        # Walk the pooled run and record what actually sits under it.
        step = g["cell"]
        ts = np.arange(t.min(), t.max() + step, step)
        probe = np.array([probe_run(occ, g, mid, u, tt) for tt in ts])
        solid, free = probe[:, 0], probe[:, 1]
        if not solid.any():
            continue
        lo, hi = np.argmax(solid), len(solid) - 1 - np.argmax(solid[::-1])
        ts, solid, free = ts[lo:hi + 1], solid[lo:hi + 1], free[lo:hi + 1]
        if ts[-1] - ts[0] < min_len or solid.mean() < min_solid or free.mean() > max_free:
            continue

        gaps = []
        k = 0
        while k < len(solid):
            if solid[k]:
                k += 1
                continue
            s = k
            while k < len(solid) and not solid[k]:
                k += 1
            gaps.append((ts[s], ts[k - 1] + step))

        # How far the operator ever got from this wall. Geometry the camera
        # only ever saw from 6 m down a corridor is extrapolation, and the
        # map should not present it with the same confidence as a wall the
        # operator walked past at arm's length.
        seg = np.stack([mid + u * ts[0], mid + u * ts[-1]])
        view = float(np.min(np.linalg.norm(
            path_xy[:, None, :] - seg[None, :, :], axis=2)))

        walls.append(dict(a=mid + u * ts[0], b=mid + u * ts[-1], u=u, mid=mid,
                          length=float(ts[-1] - ts[0]), view_dist=view,
                          solid=float(solid.mean()), gaps=gaps))
    walls.sort(key=lambda w: -w["length"])
    return walls


def wall_openings(walls, path_xy):
    """Door-sized gaps inside a wall run, with wall on both sides.

    A wall that simply stops is the edge of the reconstruction, not a
    door. Only an interior gap counts, and only at door width.
    """
    out = []
    for w in walls:
        for t0, t1 in w["gaps"]:
            width = t1 - t0
            if not DOOR_WIDTH[0] <= width <= DOOR_WIDTH[1]:
                continue
            a = w["mid"] + w["u"] * t0
            b = w["mid"] + w["u"] * t1
            c = (a + b) / 2
            # Two parallel fits of the same wall face find the same door
            # twice; one door is one door.
            if any(np.linalg.norm(c - o["centre"]) < 0.5 for o in out):
                continue
            view = float(np.min(np.linalg.norm(path_xy - c, axis=1)))
            out.append(dict(a=[round(float(v), 3) for v in a],
                            b=[round(float(v), 3) for v in b], centre=c,
                            width=round(float(width), 2),
                            view_dist=round(view, 2)))
    for o in out:
        o["centre"] = [round(float(v), 3) for v in o["centre"]]
    return out


def path_tangent(path_xy, span=6):
    """Smoothed direction of travel at each station.

    The camera heading is not the direction of travel here: partway down
    the corridor the operator swung the phone 40 deg to look at a door and
    swung back. Measuring corridor width along that heading would measure
    a diagonal. The path itself does not lie about where the walk went.
    """
    n = len(path_xy)
    t = np.zeros((n, 2))
    for i in range(n):
        a, b = max(0, i - span), min(n - 1, i + span)
        v = path_xy[b] - path_xy[a]
        t[i] = v / max(np.linalg.norm(v), 1e-9)
    return t


def corridor_width(occ, g, path_xy, tang):
    """Measured clearance at each station: cast left and right, sum the hits."""
    rows = []
    reach = int(4.0 / g["cell"])
    for (px, py), u in zip(path_xy, tang):
        d = []
        for side in (+1, -1):
            dx, dy = -u[1] * side, u[0] * side
            hit = None
            for s in range(1, reach):
                i = int((px - g["x0"]) / g["cell"] + dx * s)
                j = int((py - g["y0"]) / g["cell"] + dy * s)
                if not (0 <= i < g["w"] and 0 <= j < g["h"]):
                    break
                if occ[j, i] == OCCUPIED:
                    hit = s * g["cell"]
                    break
            d.append(hit)
        if d[0] and d[1]:
            rows.append((px, py, d[0], d[1], d[0] + d[1]))
    return rows


def colour_regions(g, occ, frac, min_frac=0.35, min_area=0.20):
    """Connected occupied cells where most of the points share one colour.

    Colour is the only handle on identity a geometric map has. It is used
    for one thing here -- the painted door surrounds in this corridor are
    saturated orange against cream block -- and each region is reported
    with the colour and the fraction that earned the label, so the claim
    can be checked rather than believed.

    Averaging the cell first would not work: one orange point beside one
    cream point averages to a colour that is neither, and the whole map
    goes beige. The vote has to happen per point.
    """
    mask = (frac >= min_frac) & (occ == OCCUPIED)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA] * g["cell"] ** 2
        if area < min_area:
            continue
        sel = lab == k
        out.append(dict(
            centre=[round(g["x0"] + cent[k][0] * g["cell"], 2),
                    round(g["y0"] + cent[k][1] * g["cell"], 2)],
            extent=[round(stats[k, cv2.CC_STAT_WIDTH] * g["cell"], 2),
                    round(stats[k, cv2.CC_STAT_HEIGHT] * g["cell"], 2)],
            area=round(area, 2),
            height=round(float(g["hmax"][sel].max()), 2),
            frac=round(float(frac[sel].mean()), 3),
            rgb=[int(v) for v in g["rgb"][sel].mean(0).round()]))
    out.sort(key=lambda r: -r["area"])
    return out


# --- output ------------------------------------------------------------
def render_png(g, occ, walls, openings, path_ij, regions, out):
    img = np.zeros((g["h"], g["w"], 3), np.uint8)
    img[:] = (23, 19, 10)
    img[occ == FREE] = (58, 78, 52)
    hm = np.clip(g["hmax"] / 1.8, 0, 1)
    ob = occ == OCCUPIED
    img[ob] = np.stack([(60 + 60 * hm[ob]), (110 * hm[ob] + 40), (60 + 180 * hm[ob])], 1).astype(np.uint8)

    def pt(p):
        return (int((p[0] - g["x0"]) / g["cell"]), int((p[1] - g["y0"]) / g["cell"]))

    # BGR, from the viewer's palette: walls cyan, openings green,
    # colour-labelled regions crab orange, camera path amber.
    for w in walls:
        cv2.line(img, pt(w["a"]), pt(w["b"]), (196, 196, 79), 1)
    for o in openings:
        cv2.line(img, pt(o["a"]), pt(o["b"]), (140, 192, 100), 2)
    for r in regions:
        c = pt(r["centre"])
        cv2.circle(img, c, max(3, int(0.20 / g["cell"])), (60, 112, 240), 2)
    for k in range(1, len(path_ij)):
        cv2.line(img, tuple(path_ij[k-1]), tuple(path_ij[k]), (60, 178, 230), 1)

    img = cv2.flip(img, 0)                 # +y up, the way a map is read
    scale = max(1, int(1400 / g["w"]))
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    bar_m = 1.0
    px = int(bar_m / g["cell"] * scale)
    y = img.shape[0] - 24
    cv2.line(img, (24, y), (24 + px, y), (234, 231, 220), 2)
    cv2.putText(img, f"{bar_m:.0f} m", (24, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (234, 231, 220), 1, cv2.LINE_AA)
    cv2.imwrite(out, img)
    return img.shape


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a)).decode("ascii")


def main():
    cloud = sys.argv[1] if len(sys.argv) > 1 else "out/world_tri.ply"
    sfm = sys.argv[2] if len(sys.argv) > 2 else "sfm/sparse_tri/1"
    stem = sys.argv[3] if len(sys.argv) > 3 else "out/birdseye"
    js_out = sys.argv[4] if len(sys.argv) > 4 else "crab-team-n/viewers/birdseye.js"
    cell = float(sys.argv[5]) if len(sys.argv) > 5 else CELL

    t0 = time.time()
    P, COL = read_ply(cloud)
    C, R, hfov = read_colmap(sfm)
    print(f"  {len(P)} points, {len(C)} poses, hfov {math.degrees(hfov):.1f} deg")

    gp = ground_plane(P, C, R)
    print(f"  ground plane: normal {np.round(gp['normal'], 4)}  "
          f"inliers {100*gp['inlier_frac']:.1f}% of {gp['n_low']} sub-camera points")
    print(f"    camera pitch below level {gp['cam_pitch_deg']:.1f} deg, "
          f"plane vs trajectory prior {gp['tilt_vs_traj_deg']:.2f} deg")
    print(f"    camera height above plane {gp['cam_height']:.3f} m "
          f"+/- {gp['cam_height_std']:.3f} (independent check on scale)")

    n, d, ex, ey = gp["normal"], gp["offset"], gp["ex"], gp["ey"]
    H = P @ n + d
    X, Y = P @ ex, P @ ey

    cls = np.zeros(len(P), np.uint8)
    cls[(H > FLOOR_BAND[0]) & (H < FLOOR_BAND[1])] = 1
    cls[(H >= OBSTACLE_BAND[0]) & (H < OBSTACLE_BAND[1])] = 2
    cls[H >= OBSTACLE_BAND[1]] = 3
    counts = {k: int((cls == v).sum()) for k, v in
              (("below", 0), ("floor", 1), ("obstacle", 2), ("overhead", 3))}
    for k, v in counts.items():
        print(f"    {k:9s} {v:9d}  {100*v/len(P):5.1f}%")

    path_xy = np.stack([C @ ex, C @ ey], 1)
    fwd = np.array([Ri.T @ np.array([0.0, 0.0, 1.0]) for Ri in R])
    fwd -= np.outer(fwd @ n, n)
    heading = np.arctan2(fwd @ ey, fwd @ ex)

    g = rasterise(X, Y, H, COL, cls, path_xy, cell)
    path_ij = np.stack([((path_xy[:, 0] - g["x0"]) / cell).astype(int),
                        ((path_xy[:, 1] - g["y0"]) / cell).astype(int)], 1)
    occ, carved = occupancy(g, path_ij, heading, hfov)
    print(f"  grid {g['w']} x {g['h']} @ {cell*100:.0f} cm  "
          f"({g['w']*cell:.1f} x {g['h']*cell:.1f} m)")
    print(f"    free {int((occ==FREE).sum())} ({100*(occ==FREE).mean():.1f}%)  "
          f"occupied {int((occ==OCCUPIED).sum())} ({100*(occ==OCCUPIED).mean():.1f}%)  "
          f"unknown {int((occ==UNKNOWN).sum())} ({100*(occ==UNKNOWN).mean():.1f}%)")
    print(f"    {carved} cells free only because a camera ray crossed them")

    walls = extract_walls(occ, g, path_xy)
    openings = wall_openings(walls, path_xy)
    tang = path_tangent(path_xy)
    widths = corridor_width(occ, g, path_xy, tang)
    wv = np.array([r[4] for r in widths]) if widths else np.zeros(0)

    r_, g_, b_ = COL[:, 0].astype(np.int32), COL[:, 1].astype(np.int32), COL[:, 2].astype(np.int32)
    is_orange = (r_ > 110) & (r_ > g_ * 1.30) & (g_ > b_ * 1.05) & (r_ - b_ > 55)
    orange_frac = tag_fraction(g, cls, is_orange)
    orange = colour_regions(g, occ, orange_frac, min_frac=0.30)

    plen = float(np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum())
    span = float(np.linalg.norm(path_xy[-1] - path_xy[0]))

    # How far the recovered walk bows away from the straight line between
    # its own ends, and how much of the map the operator ever got near.
    # Both are stated because the map looks equally confident everywhere
    # and is not.
    chord = (path_xy[-1] - path_xy[0]) / span
    rel = path_xy - path_xy[0]
    bow = float(np.abs(chord[0] * rel[:, 1] - chord[1] * rel[:, 0]).max())
    yy, xx = np.mgrid[0:g["h"], 0:g["w"]]
    cx = g["x0"] + (xx + .5) * cell
    cy = g["y0"] + (yy + .5) * cell
    near = np.zeros((g["h"], g["w"]), bool)
    for px, py in path_xy:
        near |= (cx - px) ** 2 + (cy - py) ** 2 < 2.5 ** 2
    known = occ != UNKNOWN
    near_frac = float((known & near).sum() / max(known.sum(), 1))
    print(f"  path bows {bow:.2f} m off its own chord; "
          f"{100*near_frac:.0f}% of mapped cells lie within 2.5 m of a camera")
    print(f"  walls {len(walls)} (longest {walls[0]['length']:.2f} m, "
          f"total {sum(w['length'] for w in walls):.1f} m)  "
          f"openings {len(openings)}  orange regions {len(orange)}")
    for w in walls[:8]:
        print(f"    wall {w['length']:5.2f} m  solid {100*w['solid']:3.0f}%  "
              f"closest camera {w['view_dist']:.2f} m  "
              f"gaps {[round(float(t1 - t0), 2) for t0, t1 in w['gaps']]}")
    for o in openings:
        print(f"    opening {o['width']:.2f} m at "
              f"({o['a'][0]:.2f},{o['a'][1]:.2f})-({o['b'][0]:.2f},{o['b'][1]:.2f})  "
              f"closest camera {o['view_dist']:.2f} m")
    for rr in orange:
        print(f"    orange region {rr['area']:.2f} m2 at "
              f"({rr['centre'][0]:.2f},{rr['centre'][1]:.2f}), {100*rr['frac']:.0f}% orange, "
              f"top {rr['height']:.2f} m, rgb {tuple(rr['rgb'])}")
    if len(wv):
        print(f"  corridor width at {len(wv)}/{len(path_xy)} stations: "
              f"median {np.median(wv):.2f} m  "
              f"IQR {np.percentile(wv,25):.2f}-{np.percentile(wv,75):.2f}")
    print(f"  walked path {plen:.2f} m, end-to-end {span:.2f} m, "
          f"mapped extent {g['w']*cell:.1f} m along corridor")

    shape = render_png(g, occ, walls, openings, path_ij, orange, stem + ".png")
    print(f"  wrote {stem}.png  {shape[1]}x{shape[0]}")

    hq = np.clip(g["hmax"] / 2.0 * 255, 0, 255).astype(np.uint8)
    data = dict(
        meta=dict(source=cloud, points=len(P), poses=len(C),
                  cell=cell, hfov_deg=round(math.degrees(hfov), 2),
                  generated=time.strftime("%Y-%m-%dT%H:%M:%S")),
        plane=dict(normal=[round(float(v), 5) for v in n], offset=round(float(d), 5),
                   inlier_frac=round(gp["inlier_frac"], 4),
                   inlier_of=gp["n_low"],
                   cam_height=round(gp["cam_height"], 3),
                   cam_height_std=round(gp["cam_height_std"], 3),
                   cam_pitch_deg=round(gp["cam_pitch_deg"], 2),
                   tilt_vs_traj_deg=round(gp["tilt_vs_traj_deg"], 3)),
        counts=counts,
        grid=dict(w=g["w"], h=g["h"], cell=cell,
                  x0=round(g["x0"], 4), y0=round(g["y0"], 4),
                  free=int((occ == FREE).sum()),
                  occupied=int((occ == OCCUPIED).sum()),
                  unknown=int((occ == UNKNOWN).sum()),
                  carved=carved, height_scale=2.0),
        layers=dict(occ=b64(occ), height=b64(hq), colour=b64(g["rgb"])),
        path=[[round(float(a), 3), round(float(b), 3)] for a, b in path_xy],
        heading=[round(float(a), 3) for a in heading],
        walls=[dict(a=[round(float(v), 3) for v in w["a"]],
                    b=[round(float(v), 3) for v in w["b"]],
                    length=round(w["length"], 2),
                    solid=round(w["solid"], 3),
                    view_dist=round(w["view_dist"], 2),
                    label="wall",
                    evidence=f"{w['length']:.2f} m straight run, occupied cells under "
                             f"{100*w['solid']:.0f}% of it "
                             f"(points {OBSTACLE_BAND[0]:.2f}-{OBSTACLE_BAND[1]:.2f} m "
                             f"above the fitted floor); nearest camera "
                             f"{w['view_dist']:.2f} m")
               for w in walls],
        openings=[dict(**o, label="opening",
                       evidence=f"{o['width']:.2f} m run with no occupied cell inside a "
                                f"wall that is solid on both sides of it; nearest "
                                f"camera {o['view_dist']:.2f} m")
                  for o in openings],
        regions=[dict(**r, label="orange door surround",
                      evidence=f"{100*r['frac']:.0f}% of the points in this "
                               f"{r['area']:.2f} m2 occupied region are saturated orange "
                               f"(mean rgb {tuple(r['rgb'])}), rising to {r['height']:.2f} m")
                 for r in orange],
        widths=[[round(float(v), 3) for v in row] for row in widths],
        metrics=dict(
            width_median=round(float(np.median(wv)), 2) if len(wv) else None,
            width_p25=round(float(np.percentile(wv, 25)), 2) if len(wv) else None,
            width_p75=round(float(np.percentile(wv, 75)), 2) if len(wv) else None,
            path_length=round(plen, 2), path_span=round(span, 2),
            extent_x=round(g["w"] * cell, 2), extent_y=round(g["h"] * cell, 2),
            path_bow=round(bow, 2), near_path_frac=round(near_frac, 3),
            width_stations=len(wv), stations=len(path_xy),
            walls=len(walls), openings=len(openings),
            wall_length_total=round(sum(w["length"] for w in walls), 2)))

    with open(stem + ".json", "w") as f:
        json.dump(data, f)
    with open(js_out, "w") as f:
        f.write("// generated by crab-team-n/tools/video3d/birdseye.py -- do not edit\n")
        f.write("window.BIRDSEYE = ")
        json.dump(data, f)
        f.write(";\n")
    print(f"  wrote {stem}.json ({os.path.getsize(stem+'.json')/1e6:.2f} MB)")
    print(f"  wrote {js_out} ({os.path.getsize(js_out)/1e6:.2f} MB)")
    print(f"  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
