"""Stage 5 — pack the world into the blob viewers/world3.html reads.

Quantises positions to 16 bits over the scene bounding box. At a scene span of ~29
model units that is a 0.45 mm step, far below the reconstruction's own error, and it
makes the page a third of the size it would otherwise be.

The per-point clip bitfield is what lets the viewer colour by "which clip saw this"
and "how many clips agree", so it ships alongside the colours.
"""
import argparse, base64, json, os
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import world_io


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sparse")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None, help="stage 3 report.json to embed")
    ap.add_argument("--names", default="")
    a = ap.parse_args()

    w = world_io.load(a.sparse)
    names = a.names.split(",") if a.names else sorted(w.clip_bits)
    label = dict(zip(sorted(w.clip_bits), names))

    allp = np.vstack([w.P, w.cam])
    lo, hi = allp.min(0), allp.max(0)
    span = float((hi - lo).max())
    q = lambda X: base64.b64encode(
        np.clip((X - lo) / span * 65534, 0, 65534).astype("<u2").tobytes()).decode()
    b = lambda X: base64.b64encode(np.asarray(X, "u1").tobytes()).decode()

    data = dict(
        lo=[round(float(x), 6) for x in lo], span=round(span, 6), R=round(w.R, 4),
        pts=dict(n=int(len(w.P)), pos=q(w.P), col=b(w.C), mask=b(w.mask),
                 tl=b(np.clip(w.track, 0, 255))),
        cam=dict(n=int(len(w.cam)), pos=q(w.cam), vid=list(w.cam_clip)),
        dropped=w.dropped, videos=label,
        report=json.load(open(a.report)) if a.report else None)
    json.dump(data, open(a.out, "w"), separators=(",", ":"))
    n2 = int((np.unpackbits(w.mask[:, None], axis=1, bitorder="little")[:, :len(names)]
              .sum(1) > 1).sum())
    print(f"wrote {a.out}  {os.path.getsize(a.out)//1024} KB | {len(w.P)} points | "
          f"{len(w.cam)} cameras | {n2} corroborated by 2+ clips")


if __name__ == "__main__":
    main()
