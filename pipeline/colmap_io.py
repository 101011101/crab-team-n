"""Read a COLMAP text model. No dependencies beyond numpy.

COLMAP writes .bin by default; every stage here runs `model_converter` first, so
these readers only handle the text form.
"""
import os
import numpy as np


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def largest_model(sparse_dir):
    """COLMAP numbers sub-models by discovery order, NOT by size.

    A run can leave a 2-image fragment in sparse/0 and the real 179-image model in
    sparse/1. Reading sparse/0 unconditionally returns almost nothing and still
    looks like a success, so always pick by size.
    """
    best, best_n = None, -1
    for d in sorted(os.listdir(sparse_dir)):
        f = os.path.join(sparse_dir, d, "images.txt")
        if not os.path.exists(f):
            continue
        n = sum(1 for l in open(f) if not l.startswith("#")) // 2
        if n > best_n:
            best, best_n = os.path.join(sparse_dir, d), n
    if best is None:
        raise SystemExit(f"no text model under {sparse_dir} (run model_converter first)")
    return best, best_n


def read_images(path):
    """name -> (R, t, C). C = -R^T t is the camera centre in world coordinates."""
    out = {}
    lines = [l for l in open(os.path.join(path, "images.txt")) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        q = np.array(list(map(float, p[1:5])))
        t = np.array(list(map(float, p[5:8])))
        R = qvec2rotmat(q)
        out[p[9]] = (R, t, -R.T @ t)
    return out


def read_image_ids(path):
    """COLMAP image id -> image name. Needed to tell which clip observed a point."""
    out = {}
    lines = [l for l in open(os.path.join(path, "images.txt")) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) >= 10:
            out[int(p[0])] = p[9]
    return out


def read_points(path, with_tracks=False):
    """xyz, rgb, reprojection error, track length (and the observing image ids)."""
    P, C, E, T, K = [], [], [], [], []
    for l in open(os.path.join(path, "points3D.txt")):
        if l.startswith("#"):
            continue
        p = l.split()
        if len(p) < 8:
            continue
        P.append(list(map(float, p[1:4])))
        C.append(list(map(int, p[4:7])))
        E.append(float(p[7]))
        T.append((len(p) - 8) // 2)
        if with_tracks:
            K.append([int(x) for x in p[8::2]])
    out = (np.array(P), np.array(C), np.array(E), np.array(T))
    return out + (K,) if with_tracks else out


def read_camera(path):
    for l in open(os.path.join(path, "cameras.txt")):
        if l.startswith("#"):
            continue
        p = l.split()
        return dict(model=p[1], w=int(p[2]), h=int(p[3]),
                    params=list(map(float, p[4:])))


def write_ply(fn, xyz, rgb=None):
    with open(fn, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(xyz)):
            if rgb is not None:
                f.write("%f %f %f %d %d %d\n" % (*xyz[i], *rgb[i]))
            else:
                f.write("%f %f %f\n" % tuple(xyz[i]))
