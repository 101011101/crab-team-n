"""Score a point cloud against what a corridor is known to be.

Needs only numpy, so reconstruct.py and any lighter tool can both import it.

WHY THIS EXISTS

Everything this pipeline produces is up to an unknown similarity, so no
distance in the cloud can be checked against a tape measure. A corridor
nevertheless supplies three facts that survive an unknown scale:

    the walls are FLAT
    the two walls are PARALLEL
    the floor meets each wall at 90 DEGREES

Those turn "the cloud looks wrong" into numbers, and give the two quantities
the pipeline used to guess -- focal length and the affine offset of the depth
model -- something to be measured against.

ONE OF THE THREE IS NOT USABLE AS AN OBJECTIVE

Wall parallelism improves monotonically as the focal length rises, because a
longer focal length approaches an orthographic projection, under which every
surface flattens toward parallel. It therefore cannot select a focal length,
and is reported as a diagnostic only. Flatness and perpendicularity both have
genuine interior minima and can.
"""

import math

import numpy as np


def fit_plane(points):
    """Total-least-squares plane. Returns (normal, centroid)."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)

    return vt[-1], centroid


def ransac_plane(points, tolerance, iterations=400, rng=None):
    rng = rng or np.random.default_rng(0)

    # Sampling a subset keeps the scoring loop cheap; the final refit still
    # sees every point, so nothing is lost but a little consensus resolution.
    sample = (points if len(points) <= 40_000
              else points[rng.choice(len(points), 40_000, replace=False)])

    best, best_count = None, 0

    for _ in range(iterations):
        trio = sample[rng.integers(0, len(sample), 3)]
        normal = np.cross(trio[1] - trio[0], trio[2] - trio[0])
        length = np.linalg.norm(normal)

        if length < 1e-9:
            continue

        normal /= length
        count = int((np.abs((sample - trio[0]) @ normal) < tolerance).sum())

        if count > best_count:
            best, best_count = (normal, trio[0]), count

    if best is None:
        return None

    normal, origin = best
    inliers = np.abs((points - origin) @ normal) < tolerance

    if inliers.sum() < 50:
        return None

    normal, centroid = fit_plane(points[inliers])

    return normal, centroid, np.abs((points - centroid) @ normal) < tolerance


def extract_planes(points, tolerance, count=4):
    """The largest planes, strongest first, each consuming its own inliers."""
    remaining = np.ones(len(points), bool)
    planes = []

    for _ in range(count):
        pool = np.flatnonzero(remaining)

        if len(pool) < 2000:
            break

        found = ransac_plane(points[pool], tolerance)

        if found is None:
            break

        normal, centroid, inliers = found
        index = pool[inliers]
        planes.append({"normal": normal, "centroid": centroid,
                       "index": index, "count": len(index)})
        remaining[index] = False

    return planes


def classify(planes):
    """Name planes by which way they face. Walls face sideways, floors up."""
    named = {}

    for plane in planes:
        normal, centroid = plane["normal"], plane["centroid"]

        if abs(normal[0]) > abs(normal[1]):
            key = "wall_left" if centroid[0] < 0 else "wall_right"
        else:
            key = "floor" if centroid[1] > 0 else "ceiling"

        if key not in named or plane["count"] > named[key]["count"]:
            named[key] = plane

    return named


def flatness(points, normal, centroid):
    """Return (bow_pct, rms_pct) for one plane's inliers.

    An RMS residual on its own cannot tell a bowed wall from a noisy flat
    one, and it also falls as the focal length rises for reasons that have
    nothing to do with the wall. A bow is smooth and low order, so fit the
    residual with a quadratic across the wall and report the range that
    quadratic covers. Both numbers are percentages of the wall's own extent,
    which keeps them meaningful under an arbitrary scale.
    """
    axis_u = np.cross(normal, [0.0, 1.0, 0.0])

    if np.linalg.norm(axis_u) < 1e-6:
        axis_u = np.cross(normal, [1.0, 0.0, 0.0])

    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)

    offset = points - centroid
    u, v = offset @ axis_u, offset @ axis_v
    residual = offset @ normal

    design = np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])
    coefficients, *_ = np.linalg.lstsq(design, residual, rcond=None)

    extent = max(np.ptp(u), np.ptp(v), 1e-9)
    rms = float(np.sqrt((residual ** 2).mean()))

    return 100.0 * np.ptp(design @ coefficients) / extent, 100.0 * rms / extent


def angle_between(first, second):
    return math.degrees(math.acos(min(1.0, abs(float(first @ second)))))


def measure(points, tolerance_frac=0.02, plane_count=4, min_plane=1500):
    """Corridor metrics for one cloud, or None when no walls were found.

    bow_pct and rms_pct are averaged over whichever walls were found.
    """
    if len(points) < 5000:
        return None

    radius = float(np.percentile(np.linalg.norm(points, axis=1), 90)) or 1.0
    named = classify(extract_planes(points, tolerance_frac * radius, plane_count))

    walls, result = {}, {}

    for key in ("wall_left", "wall_right", "floor"):
        plane = named.get(key)

        if plane is None or plane["count"] < min_plane:
            continue

        bow, rms = flatness(points[plane["index"]], plane["normal"], plane["centroid"])
        walls[key] = (bow, rms, plane["normal"], plane["count"])

    found = [k for k in ("wall_left", "wall_right") if k in walls]

    if not found:
        return None

    result["walls_found"] = len(found)
    result["wall_points"] = int(sum(walls[k][3] for k in found))
    result["bow_pct"] = float(np.mean([walls[k][0] for k in found]))
    result["rms_pct"] = float(np.mean([walls[k][1] for k in found]))

    if len(found) == 2:
        result["parallel_err_deg"] = angle_between(walls["wall_left"][2],
                                                   walls["wall_right"][2])

    if "floor" in walls:
        result["perp_err_deg"] = float(np.mean(
            [abs(90.0 - angle_between(walls["floor"][2], walls[k][2])) for k in found]))

    return result


def format_report(metrics, indent="  "):
    if metrics is None:
        return f"{indent}corridor  no wall plane found; nothing to score"

    parts = [f"{metrics['walls_found']} wall(s), "
             f"{metrics['wall_points']:,} pts",
             f"bow {metrics['bow_pct']:.2f}%",
             f"rms {metrics['rms_pct']:.2f}%"]

    if "parallel_err_deg" in metrics:
        parts.append(f"wall-wall {metrics['parallel_err_deg']:.1f} deg off parallel")

    if "perp_err_deg" in metrics:
        parts.append(f"floor-wall {metrics['perp_err_deg']:.1f} deg off square")

    return f"{indent}corridor  " + ", ".join(parts)
