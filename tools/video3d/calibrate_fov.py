"""Measure the camera's focal length from the footage itself.

    python3 calibrate_fov.py ../../../IMG_0300.MOV

Needs only opencv-python and numpy, so it runs in central/.venv alongside
feasibility.py. Run it before reconstruct.py and pass the number it prints.

WHY THIS EXISTS

reconstruct.py used to take the field of view as an assertion (--hfov-deg,
default 60). A focal length that is wrong by a factor k scales every
back-projected X and Y by k while leaving Z alone, so the scene comes out
stretched or squashed along the view axis. On corridor footage that failure
mode looks exactly like the corridor collapsing into a tube, which is why
guessing here is not a small sin.

THE METHOD

Two orthogonal directions in the world project to two vanishing points v1
and v2. Writing c for the principal point, their back-projected rays are
orthogonal only when

    f^2 = -(v1 - c) . (v2 - c)

so any pair of perpendicular scene directions measures the focal length. A
corridor is close to the ideal case: the long axis, the vertical door jambs
and wall corners, and the horizontal courses of the far wall give three
mutually orthogonal families of parallel lines, all of them long and
straight.

WHAT IT ASSUMES

  principal point   at the image centre. Never measured. A phone lens is
                    close enough that the error this contributes is smaller
                    than the spread across frames.
  square pixels     fx = fy, no skew.
  no distortion     the lines are treated as straight. A phone's wide lens is
                    mildly barrel distorted, which bends long segments and
                    biases f slightly low.

The vertical vanishing point is only finite because the operator walked with
the camera pitched down. A perfectly level camera puts it at infinity, where
the constraint above degenerates and this tool returns nothing usable.
"""

import argparse
import math
import sys

import cv2
import numpy as np

from feasibility import sample_frames


# ---------------------------------------------------------------
# LINES
# ---------------------------------------------------------------

def detect_segments(gray, min_length):
    """Line segments, long ones only.

    Short segments carry almost no direction information but dominate the
    count, so an unfiltered set lets texture noise outvote the wall edges
    that actually define the vanishing points.
    """
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    found = detector.detect(gray)[0]

    if found is None:
        return np.zeros((0, 4))

    segments = found.reshape(-1, 4).astype(np.float64)
    length = np.hypot(segments[:, 2] - segments[:, 0],
                      segments[:, 3] - segments[:, 1])

    return segments[length >= min_length]


def segment_lines(segments):
    """Homogeneous line through each segment, normalised on its (a, b) part."""
    start = np.column_stack([segments[:, 0], segments[:, 1], np.ones(len(segments))])
    end = np.column_stack([segments[:, 2], segments[:, 3], np.ones(len(segments))])

    lines = np.cross(start, end)
    scale = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)

    return lines / np.maximum(scale, 1e-12)


def segment_lengths(segments):
    return np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])


def consensus(segments, vanishing, tolerance_deg):
    """Segments whose direction points at the vanishing point.

    Scored on the angle at the segment MIDPOINT rather than on the algebraic
    residual l.v, because that residual grows with distance to the vanishing
    point and would quietly prefer distant ones.
    """
    mid = np.column_stack([(segments[:, 0] + segments[:, 2]) * 0.5,
                           (segments[:, 1] + segments[:, 3]) * 0.5])

    direction = np.column_stack([segments[:, 2] - segments[:, 0],
                                 segments[:, 3] - segments[:, 1]])
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)

    if abs(vanishing[2]) < 1e-9:
        # A vanishing point at infinity is a direction, the same for every pixel.
        ray = np.broadcast_to(vanishing[:2] / np.linalg.norm(vanishing[:2]),
                              direction.shape).copy()
    else:
        ray = vanishing[:2] / vanishing[2] - mid
        ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12)

    cosine = np.abs(np.sum(direction * ray, axis=1))

    return np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))) < tolerance_deg


def refit(lines, weights):
    """Vanishing point minimising sum w (l.v)^2: smallest right singular vector."""
    _, _, vt = np.linalg.svd(lines * weights[:, None], full_matrices=False)

    return vt[-1]


def find_vanishing_points(segments, tolerance_deg, iterations, min_support, rng):
    """Up to three vanishing points by sequential RANSAC, strongest first."""
    lines = segment_lines(segments)
    lengths = segment_lengths(segments)

    remaining = np.ones(len(segments), bool)
    found = []

    for _ in range(3):
        pool = np.flatnonzero(remaining)

        if len(pool) < min_support:
            break

        best_inliers, best_score = None, 0.0

        for _ in range(iterations):
            a, b = rng.choice(pool, 2, replace=False)
            candidate = np.cross(lines[a], lines[b])

            if np.linalg.norm(candidate) < 1e-9:
                continue

            inliers = np.zeros(len(segments), bool)
            inliers[pool] = consensus(segments[pool], candidate, tolerance_deg)

            # Total inlier LENGTH, not inlier count: one 600 px wall edge is
            # far better evidence than ten 40 px scratches on the floor.
            score = float(lengths[inliers].sum())

            if score > best_score:
                best_inliers, best_score = inliers, score

        if best_inliers is None or best_inliers.sum() < min_support:
            break

        vanishing = refit(lines[best_inliers], lengths[best_inliers])

        inliers = np.zeros(len(segments), bool)
        inliers[remaining] = consensus(segments[remaining], vanishing, tolerance_deg)

        if inliers.sum() >= min_support:
            vanishing = refit(lines[inliers], lengths[inliers])

        found.append((vanishing, float(lengths[inliers].sum()), int(inliers.sum())))
        remaining &= ~inliers

    return found


# ---------------------------------------------------------------
# FOCAL LENGTH
# ---------------------------------------------------------------

def focal_from_pair(first, second, centre):
    """f from two vanishing points, or None when the pair cannot supply it."""
    if abs(first[2]) < 1e-9 or abs(second[2]) < 1e-9:
        return None

    offset_a = first[:2] / first[2] - centre
    offset_b = second[:2] / second[2] - centre
    dot = float(offset_a @ offset_b)

    # A positive dot product means the two rays cannot be made orthogonal by
    # any real focal length: the pair is not an orthogonal pair, or one
    # vanishing point is junk.
    if dot >= -1.0:
        return None

    return math.sqrt(-dot)


def weighted_median(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])

    return float(values[order][np.searchsorted(cumulative, 0.5 * weights.sum())])


def estimate_focal(images, tolerance_deg=1.5, iterations=2000, min_support=12,
                   min_length_frac=0.04, seed=0):
    """Focal length in pixels from every orthogonal vanishing-point pair found.

    Returns (focal_px, low, high, samples). The interval is a 95 per cent
    bootstrap over the pooled estimates, so it reports the spread of this
    measurement and NOT the lens distortion or principal-point error, which
    are systematic and sit outside it.
    """
    rng = np.random.default_rng(seed)
    focals, weights, contributing = [], [], 0

    for image in images:
        height, width = image.shape[:2]
        centre = np.array([width * 0.5, height * 0.5])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        segments = detect_segments(gray, min_length_frac * math.hypot(width, height))

        if len(segments) < 2 * min_support:
            continue

        points = find_vanishing_points(segments, tolerance_deg, iterations,
                                       min_support, rng)
        before = len(focals)

        for a in range(len(points)):
            for b in range(a + 1, len(points)):
                focal = focal_from_pair(points[a][0], points[b][0], centre)

                # Reject focals outside a range no phone camera can occupy,
                # rather than let one degenerate pair drag the median.
                if focal is None or not 0.2 * width < focal < 4.0 * height:
                    continue

                focals.append(focal)
                weights.append(min(points[a][1], points[b][1]))

        contributing += len(focals) > before

    if not focals:
        return None

    focals = np.asarray(focals)
    weights = np.asarray(weights)
    median = weighted_median(focals, weights)

    draws = rng.integers(0, len(focals), (2000, len(focals)))
    boot = np.array([weighted_median(focals[d], weights[d]) for d in draws])
    low, high = np.percentile(boot, (2.5, 97.5))

    return {
        "focal_px": median,
        "low_px": float(low),
        "high_px": float(high),
        "samples": len(focals),
        "frames": contributing,
        "spread_px": float(np.median(np.abs(focals - median))),
    }


def fov_deg(focal_px, extent_px):
    return math.degrees(2.0 * math.atan(0.5 * extent_px / focal_px))


# ---------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--stride-s", type=float, default=0.2,
                        help="seconds between sampled frames (default 0.2)")
    parser.add_argument("--max-frames", type=int, default=60,
                        help="frames to measure (default 60)")
    parser.add_argument("--tolerance-deg", type=float, default=1.5,
                        help="angle a segment may miss a vanishing point by")
    parser.add_argument("--assumed-hfov-deg", type=float, default=60.0,
                        help="value to compare the measurement against")

    args = parser.parse_args(argv)

    frames, fps, stride, total = sample_frames(args.video, args.stride_s,
                                               args.max_frames)

    if not frames:
        raise SystemExit(f"no frames read from {args.video}")

    images = [frame for _, frame in frames]
    height, width = images[0].shape[:2]

    print(f"  video    {args.video}  {width}x{height}  {total} frames")
    print(f"  sampled  {len(images)} frames, every {stride}")

    result = estimate_focal(images, tolerance_deg=args.tolerance_deg)

    if result is None:
        print("  FAILED   no orthogonal vanishing-point pair found. This scene "
              "has no two perpendicular")
        print("           families of parallel lines, or the camera is too "
              "level for the vertical one.")
        return 1

    focal = result["focal_px"]
    assumed = 0.5 * width / math.tan(math.radians(args.assumed_hfov_deg) * 0.5)

    print(f"  focal    {focal:.0f} px  "
          f"95% CI [{result['low_px']:.0f}, {result['high_px']:.0f}]  "
          f"MAD {result['spread_px']:.0f}")
    print(f"           from {result['samples']} orthogonal pairs over "
          f"{result['frames']} frames")
    print(f"  fov      {fov_deg(focal, width):.1f} deg across {width} px "
          f"(image width)")
    print(f"           {fov_deg(focal, height):.1f} deg across {height} px "
          f"(image height)")
    print(f"           {focal / max(width, height) * 36:.0f} mm "
          f"35 mm-equivalent on the long axis")
    print(f"  assumed  --hfov-deg {args.assumed_hfov_deg:.0f} implies "
          f"{assumed:.0f} px, which is {focal / assumed:.2f}x off")
    print()
    print(f"  Pass  --focal-px {focal:.0f}  to reconstruct.py.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
