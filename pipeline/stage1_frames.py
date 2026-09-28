"""Stage 1 — video in, the frames worth solving out.

Sampling on a fixed stride is the obvious thing and it is wrong. A handheld clip
held near the floor is full of motion blur, and a blurred frame contributes no
usable SIFT features but still costs a full extraction. So: decode at a higher rate
than needed, score every candidate by variance of Laplacian, and keep the sharpest
one in each bucket.

Measured on IMG_7373/4/5: this lifted the 10th-percentile sharpness by 1.4x to 3.0x
against a fixed stride, for the same number of frames.

Each clip gets a one-letter prefix (a_, b_, c_ ...). Everything downstream reads
that prefix to know which clip a frame came from, so do not rename the files.
"""
import argparse, glob, json, os, shutil, subprocess, sys
import numpy as np
import cv2

TAGS = "abcdefghijklmnopqrstuvwxyz"


def decode(video, work, fps):
    d = os.path.join(work, os.path.basename(video).split(".")[0])
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{d}/f_*.jpg"):
        os.remove(f)
    subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"fps={fps}",
                    "-q:v", "2", "-y", f"{d}/f_%04d.jpg"], check=True)
    return sorted(glob.glob(f"{d}/f_*.jpg"))


def sharpness(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True, help="image directory the solver will read")
    ap.add_argument("--work", default=None, help="scratch for decoded candidates")
    ap.add_argument("--candidate-fps", type=float, default=12.0)
    ap.add_argument("--keep-fps", type=float, default=3.0)
    a = ap.parse_args()

    work = a.work or os.path.join(a.out, "_candidates")
    os.makedirs(a.out, exist_ok=True)
    bucket = max(1, round(a.candidate_fps / a.keep_fps))
    report = {}

    for i, v in enumerate(a.videos):
        tag = TAGS[i]
        files = decode(v, work, a.candidate_fps)
        sharp = [sharpness(f) for f in files]
        picked = [s + int(np.argmax(sharp[s:s + bucket])) for s in range(0, len(files), bucket)]
        for n, idx in enumerate(picked):
            shutil.copyfile(files[idx], f"{a.out}/{tag}_{n:04d}.jpg")
        ps = [sharp[i] for i in picked]
        report[tag] = dict(video=os.path.basename(v), candidates=len(files), kept=len(picked),
                           sharp_p10_all=round(float(np.percentile(sharp, 10)), 1),
                           sharp_p10_kept=round(float(np.percentile(ps, 10)), 1),
                           gain=round(float(np.percentile(ps, 10) / max(np.percentile(sharp, 10), 1e-9)), 2))
        print(f"{tag}  {os.path.basename(v):16} {len(files):4} candidates -> {len(picked):3} kept"
              f"   p10 sharpness {report[tag]['sharp_p10_all']:7.1f} -> "
              f"{report[tag]['sharp_p10_kept']:7.1f}  ({report[tag]['gain']}x)")

    json.dump(report, open(os.path.join(a.out, "..", "frame_selection.json"), "w"), indent=2)
    print(f"\n{sum(r['kept'] for r in report.values())} frames in {a.out}")


if __name__ == "__main__":
    main()
