"""Stage 3 — did the clips actually fuse? Numbers, not vibes.

Four questions, each with a measurement:

1. Did every clip get into ONE model?          per_video registration
2. Are the clips linked, or merely next to     cross-clip verified pairs, read
   each other?                                 straight out of the COLMAP database
3. Do the clips agree on where things are?     points observed by 2+ clips
4. Is the fused world more complete than       floor coverage of each clip alone
   the best single clip?                       against all of them, in ONE frame

(4) is the only one that answers "was layering worth it", and it has to be measured
inside the joint model: three separate solves have three arbitrary scales, so their
coverage numbers cannot be compared to each other at all.
"""
import argparse, collections, itertools, json, os, sqlite3
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import world_io
from colmap_io import write_ply


def cross_clip_matches(db, clip_of=lambda n: n[0]):
    """Verified pairs per clip pair, read from COLMAP's own geometry table.

    'Verified' is the real test: a pair survives only if a consistent two-view
    geometry explains the matches. Raw match counts prove nothing.
    """
    c = sqlite3.connect(db)
    names = dict(c.execute("select image_id, name from images").fetchall())
    raw, ver, inl = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    for pid, n in c.execute("select pair_id, rows from matches"):
        a, b = pid // 2147483647, pid % 2147483647
        raw["".join(sorted(clip_of(names[a]) + clip_of(names[b])))] += 1
    for pid, n in c.execute("select pair_id, rows from two_view_geometries"):
        if n <= 0:
            continue
        a, b = pid // 2147483647, pid % 2147483647
        k = "".join(sorted(clip_of(names[a]) + clip_of(names[b])))
        ver[k] += 1
        inl[k].append(n)
    return {k: dict(tried=raw[k], verified=ver[k],
                    rate=round(100 * ver[k] / raw[k], 1) if raw[k] else 0.0,
                    median_inliers=int(np.median(inl[k])) if inl[k] else 0)
            for k in sorted(raw)}


def coverage(w, cells_per_R=30.0):
    """Occupied floor cells, per clip and for the union. One frame, one scale."""
    X = w.P - w.P.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    F = X @ Vt.T[:, :2]
    cell = w.R / cells_per_R
    grid = np.floor(F / cell).astype(int)
    out = {}
    for tag, bit in w.clip_bits.items():
        m = (w.mask & bit) > 0
        out[tag] = len(set(map(tuple, grid[m]))) if m.any() else 0
    out["all"] = len(set(map(tuple, grid)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", help="stage 2 output directory (holds db.db and sparse/)")
    ap.add_argument("--out", default=None, help="report JSON (default <work>/report.json)")
    ap.add_argument("--ply", default=None, help="also write the cleaned cloud")
    ap.add_argument("--names", default="", help="comma list mapping clip tags to video names")
    a = ap.parse_args()

    w = world_io.load(os.path.join(a.work, "sparse"))
    label = dict(zip(sorted(w.clip_bits), a.names.split(","))) if a.names else {}
    nm = lambda t: label.get(t, t)

    per = {}
    for tag in sorted(w.clip_bits):
        reg = int((w.cam_clip == tag).sum())
        drop = sum(1 for d in w.dropped if d[0] == tag)
        per[nm(tag)] = dict(registered=reg, dropped_diverged=drop)

    nvid = np.unpackbits(w.mask[:, None], axis=1, bitorder="little")[:, :len(w.clip_bits)].sum(1)
    shared = collections.Counter()
    for x, y in itertools.combinations(sorted(w.clip_bits), 2):
        shared[f"{nm(x)}+{nm(y)}"] = int((((w.mask & w.clip_bits[x]) > 0) &
                                          ((w.mask & w.clip_bits[y]) > 0)).sum())

    cov = coverage(w)
    best = max(v for k, v in cov.items() if k != "all")

    rep = dict(
        model=w.model,
        cameras_used=int(len(w.cam)), cameras_dropped=w.dropped,
        points=int(len(w.P)), camera_path_radius=round(w.R, 4),
        per_clip=per,
        points_by_clips_observing={str(k): int(v) for k, v in
                                   sorted(collections.Counter(nvid).items())},
        points_corroborated=int((nvid > 1).sum()),
        points_shared_between=dict(shared),
        cross_clip_matches=cross_clip_matches(os.path.join(a.work, "db.db")),
        floor_coverage_cells={nm(k) if k != "all" else "all clips": v for k, v in cov.items()},
        coverage_gain_pct=round(100 * (cov["all"] / best - 1), 1),
        units="model units; monocular SfM has no metre")

    out = a.out or os.path.join(a.work, "report.json")
    json.dump(rep, open(out, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print(f"\nwrote {out}")
    if a.ply:
        write_ply(a.ply, w.P, w.C)
        print(f"wrote {a.ply}")


if __name__ == "__main__":
    main()
