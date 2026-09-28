"""Check calibrate_fov.py against a corridor whose focal length is known.

    ../../central/.venv/bin/python synth_corridor.py

Runs in central/.venv. No model, no video, no download.

WHY THIS EXISTS

calibrate_fov.py contradicts a COLMAP run on the same clip by a factor of
1.67, and a measurement that disagrees with another measurement is worth
nothing until one of them has been shown to work on a case where the answer
is known. make_test_video.py cannot serve: its planes face the camera, so
its vanishing points are at infinity and the orthogonality constraint
degenerates.

So draw a corridor instead -- two walls, a floor, a ceiling, block courses on
each -- project it through a chosen K at a chosen pitch and roll, and ask the
estimator to recover the focal length it was drawn with.

WHAT THIS DOES NOT TEST

The render is a perfect pinhole: no lens distortion, no motion blur, no JPEG,
and the principal point really is at the image centre. Real footage breaks
all four. This bounds the estimator's error from BELOW. It answers "is the
method sound and unbiased", not "how well does it do on a phone".
"""

import argparse
import math
import sys

import cv2
import numpy as np

from calibrate_fov import estimate_focal


def corridor_segments(half_width=1.0, floor_y=1.5, ceiling_y=-1.2, far_z=25.0):
    """A corridor as 3D line segments, in a camera-like frame: y down, z forward.

    Dimensions are a real corridor: 2 m wide, camera 1.5 m above the floor.
    The exact numbers do not matter to the test, only that the three families
    of parallel edges are mutually orthogonal.
    """
    segments = []

    for x in (-half_width, half_width):
        for y in np.arange(ceiling_y, floor_y + 0.05, 0.2):        # block courses
            segments.append(((x, y, 0.3), (x, y, far_z)))

        for z in np.arange(0.5, far_z, 0.4):                       # vertical joints
            segments.append(((x, ceiling_y, z), (x, floor_y, z)))

    for x in np.arange(-half_width, half_width + 0.01, 0.25):      # floor, ceiling seams
        segments.append(((x, floor_y, 0.3), (x, floor_y, far_z)))
        segments.append(((x, ceiling_y, 0.3), (x, ceiling_y, far_z)))

    for z in np.arange(0.5, far_z, 0.5):
        segments.append(((-half_width, floor_y, z), (half_width, floor_y, z)))
        segments.append(((-half_width, ceiling_y, z), (half_width, ceiling_y, z)))

    return segments


def render(focal, width, height, pitch_deg, roll_deg=0.0):
    K = np.array([[focal, 0.0, width * 0.5],
                  [0.0, focal, height * 0.5],
                  [0.0, 0.0, 1.0]])

    pitch, roll = math.radians(pitch_deg), math.radians(roll_deg)
    rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll), math.cos(roll), 0],
                   [0, 0, 1.0]])
    rotation = rx @ rz

    image = np.full((height, width, 3), 30, np.uint8)

    for start, end in corridor_segments():
        a, b = rotation @ np.array(start), rotation @ np.array(end)

        # Anything at or behind the image plane projects to nonsense; a
        # proper near-plane clip is not worth it for a test pattern.
        if a[2] < 0.2 or b[2] < 0.2:
            continue

        pa, pb = K @ a, K @ b
        pa, pb = pa[:2] / pa[2], pb[:2] / pb[2]

        if not (np.all(np.abs(pa) < 1e5) and np.all(np.abs(pb) < 1e5)):
            continue

        cv2.line(image, tuple(pa.astype(int)), tuple(pb.astype(int)),
                 (220, 220, 210), 2, cv2.LINE_AA)

    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--focals", type=float, nargs="+",
                        default=[935.0, 1200.0, 1655.0, 2000.0],
                        help="focal lengths to render and try to recover")
    parser.add_argument("--pitches", type=float, nargs="+", default=[10.0, 18.0],
                        help="camera pitch in degrees. A level camera puts "
                             "the vertical vanishing point at infinity and "
                             "the method degenerates")
    parser.add_argument("--save", help="write one rendered frame here")

    args = parser.parse_args(argv)

    print(f"  render   {args.width}x{args.height} pinhole corridor, no "
          f"distortion, no blur")
    print(f"  {'true f':>8} {'pitch':>6} {'recovered':>10} {'error':>8} "
          f"{'95% CI':>16} {'pairs':>6}")

    worst = 0.0

    for focal in args.focals:
        for pitch in args.pitches:
            # Vary pose across the batch. One frame repeated would understate
            # the spread, since every estimate would share its geometry.
            images = [render(focal, args.width, args.height, pitch + step,
                             roll_deg=1.5 * step)
                      for step in (-2, 0, 2, 4)]

            result = estimate_focal(images)

            if result is None:
                print(f"  {focal:8.0f} {pitch:6.0f} {'FAILED':>10}")
                continue

            error = 100.0 * (result["focal_px"] - focal) / focal
            worst = max(worst, abs(error))
            interval = f"[{result['low_px']:.0f}, {result['high_px']:.0f}]"

            print(f"  {focal:8.0f} {pitch:6.0f} {result['focal_px']:10.0f} "
                  f"{error:+7.1f}% {interval:>16} {result['samples']:6d}")

    print(f"  worst    {worst:.1f}% error over "
          f"{len(args.focals) * len(args.pitches)} cases")
    print("  A method biased toward long focal lengths would fail the "
          "smallest case here.")

    if args.save:
        cv2.imwrite(args.save, render(args.focals[-1], args.width,
                                      args.height, args.pitches[-1]))
        print(f"  wrote    {args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
