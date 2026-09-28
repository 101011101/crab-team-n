"""Turn a video into a coloured point cloud with a monocular depth model.

    python3 reconstruct.py ../../../IMG_0300.MOV --out out/

Run feasibility.py first. This tool needs torch and transformers (see
requirements.txt) and will not run in central/.venv. That separation is the
point: the control loop must not carry a 1 GB dependency tree.

WHAT THIS IS

Depth Anything V2 Small predicts depth from ONE image. It is not
structure-from-motion: it never compares two frames to triangulate anything.
That has one large advantage and two large limits.

  advantage   it works on video that SfM refuses, including pure rotation and
              textureless walls, because it is guessing from learned priors
              rather than solving for geometry.
  limit 1     the output is affine-invariant inverse depth. Both the scale
              and the offset are unknown, so metres are not recoverable from
              one frame. Everything here is up to an unknown similarity.
  limit 2     it is guessing. A confident wrong surface looks exactly like a
              confident right one. Nothing in this file can tell them apart.

MERGING FRAMES

Each frame is back-projected in its own camera frame, then placed in a common
frame by a pose chained from the essential matrix. Rotation is recoverable
from correspondence alone; translation is recoverable only up to a scale that
no monocular method supplies, so the per-pair baseline is a command line
number and defaults to zero. Zero is the honest default for a panning camera:
it stitches the frames into one panoramic shell, which is exactly what a
camera that does not translate can support.

CALIBRATION

The focal length is no longer a guess if you do not want it to be.
calibrate_fov.py measures it from the footage; pass the result as
--focal-px, or let this tool measure it with --solve-fov. The old
--hfov-deg still works and still defaults to 60, byte for byte, but that
default is contested on the footage this was built for, so leaving it alone
now prints a warning. The README carries the evidence on both sides.

--beta-frac exposes the affine offset that to_depth used to hard-code, and
corridor.py scores the result against surfaces a corridor is known to have.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Shares the sampler and the ESP32 degradation with the feasibility gate so
# that both tools see byte-identical frames. Divergence there would make the
# gate's verdict meaningless as a prediction of what this tool receives.
from feasibility import Degrader, sample_frames

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


# ---------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------

def load_model(name, device):
    """Import torch lazily so --help still works outside this tool's venv."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForDepthEstimation.from_pretrained(name)
    model.to(device).eval()

    return torch, processor, model


def pick_device(requested):
    import torch

    if requested != "auto":
        return requested

    return "mps" if torch.backends.mps.is_available() else "cpu"


def synchronise(torch, device):
    """MPS queues work asynchronously, so a forward pass returns before it has
    run. Timing without this measures the enqueue, not the inference."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def predict_disparity(torch, processor, model, device, frame):
    """Return raw model output resampled to the frame size.

    The values are inverse depth: LARGER IS NEARER, in arbitrary units with an
    unknown offset. The name says disparity and not depth on purpose.
    """
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    inputs = processor(images=rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        prediction = model(**inputs).predicted_depth

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1), size=(height, width),
        mode="bicubic", align_corners=False)

    return prediction.squeeze().float().cpu().numpy()


# ---------------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------------

def focal_from_hfov(width, hfov_deg):
    return 0.5 * width / math.tan(math.radians(hfov_deg) * 0.5)


def intrinsics(width, height, focal_px):
    """Pinhole K from a focal length in pixels.

    A wrong focal length does not bend the cloud arbitrarily: it scales X and
    Y against Z, so the scene comes out stretched or squashed along the view
    axis. On corridor footage that reads as the corridor collapsing into a
    tube, which is why calibrate_fov.py exists and why --hfov-deg alone is
    not good enough.
    """
    return np.array([[focal_px, 0.0, width * 0.5],
                     [0.0, focal_px, height * 0.5],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def to_depth(disparity, near_m, far_m, beta_frac=0.0):
    """Map affine-invariant inverse depth to a depth in fake metres.

    The model returns d = alpha/Z + beta for unknown alpha and beta, so
    Z = alpha/(d - beta) and the OFFSET beta has to come from somewhere.
    beta_frac = 0 places it at the 2nd percentile of d, which is what this
    tool has always done; larger values push it further below, flattening the
    near field and stretching the far one. corridor.py can pick it by
    measuring wall flatness, since a corridor's walls are known to be flat.

    Robust percentiles, not min and max: one speculative pixel on a window
    frame otherwise sets the scale for the whole image.
    """
    low, high = np.percentile(disparity, (2.0, 98.0))
    span = high - low

    if span < 1e-6:
        return np.full_like(disparity, far_m)

    beta = low - beta_frac * span
    depth = near_m * (high - beta) / np.maximum(disparity - beta, 1e-3 * span)

    return np.clip(depth, near_m, far_m)


def back_project(depth, colour, K, stride):
    """Lift pixels to camera-frame points. Returns (xyz, rgb)."""
    height, width = depth.shape

    rows = np.arange(0, height, stride)
    cols = np.arange(0, width, stride)
    grid_u, grid_v = np.meshgrid(cols, rows)

    z = depth[rows][:, cols].astype(np.float32)
    x = (grid_u - K[0, 2]) * z / K[0, 0]
    y = (grid_v - K[1, 2]) * z / K[1, 1]

    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    rgb = colour[rows][:, cols].reshape(-1, 3)[:, ::-1]      # BGR to RGB

    return xyz, rgb


def trim_tail(xyz, rgb, depth, keep_pct):
    """Drop the sparse far tail of the cloud.

    The depth model's far field is a guess stretched over very few disparity
    values, so a handful of pixels land metres beyond where the dense body
    ends and set the extent of the whole cloud. Cutting at a percentile of
    DEPTH rather than at a fixed distance keeps this independent of the
    arbitrary scale.
    """
    if keep_pct >= 100.0:
        return xyz, rgb, None

    cut = float(np.percentile(depth, keep_pct))
    keep = depth <= cut

    return xyz[keep], rgb[keep], cut


def voxel_denoise(xyz, rgb, voxel_frac, min_count):
    """Drop points sitting in near-empty voxels.

    A k-nearest-neighbour outlier test would be the usual tool, but scipy is
    deliberately not in either venv. Occupancy of a grid cell is the same
    idea one order of approximation coarser, costs one np.unique, and removes
    what actually hurts here: speckle flung off depth discontinuities at
    doorways and light fittings.
    """
    if min_count <= 1 or len(xyz) == 0:
        return xyz, rgb, 0

    extent = np.percentile(xyz, 99.0, axis=0) - np.percentile(xyz, 1.0, axis=0)
    size = voxel_frac * float(np.max(extent))

    if size <= 0:
        return xyz, rgb, 0

    keys = np.floor(xyz / size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                   return_counts=True)
    keep = counts[inverse.ravel()] >= min_count

    return xyz[keep], rgb[keep], int((~keep).sum())


def relative_pose(frame_a, frame_b, K, detector):
    """Recover (R, unit_t) taking points from camera A's frame into camera B's.

    Returns None when the pair does not support a pose at all, which is a
    normal outcome on a blurred or featureless pair and must not be silently
    turned into an identity pose.
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    kp_a, desc_a = detector.detectAndCompute(gray_a, None)
    kp_b, desc_b = detector.detectAndCompute(gray_b, None)

    if desc_a is None or desc_b is None or len(desc_a) < 8 or len(desc_b) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)

    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < 0.75 * n.distance]

    if len(good) < 16:
        return None

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

    essential, mask = cv2.findEssentialMat(pts_a, pts_b, K, cv2.RANSAC,
                                           0.999, 1.0)

    if essential is None or essential.shape != (3, 3):
        return None

    _, rotation, translation, _ = cv2.recoverPose(essential, pts_a, pts_b, K,
                                                  mask=mask)

    return rotation, translation.reshape(3)


def chain_poses(frames, K, baseline_m, detector):
    """World pose per frame, as (R_cam_to_world, camera_centre).

    The first camera defines the world frame. A pair that fails to give a pose
    inherits the previous one: better a duplicated viewpoint than a random
    rotation thrown into the middle of the cloud.
    """
    poses = [(np.eye(3), np.zeros(3))]
    failures = 0

    for previous, current in zip(frames, frames[1:]):
        rotation_world, centre = poses[-1]
        estimate = relative_pose(previous, current, K, detector)

        if estimate is None:
            failures += 1
            poses.append((rotation_world, centre))
            continue

        rotation, translation = estimate

        # recoverPose gives X_b = R X_a + t, so camera B's orientation in the
        # world is R_a R^T and its centre moves by -R^T t, scaled by the
        # baseline the user asserts. Monocular vision cannot supply that
        # scale; only the user knows how far they walked.
        next_rotation = rotation_world @ rotation.T
        step = -baseline_m * (rotation.T @ translation)
        next_centre = centre + rotation_world @ step

        poses.append((next_rotation, next_centre))

    return poses, failures


# ---------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------

PLY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])


def write_ply(path, xyz, rgb):
    """Binary little-endian PLY. ASCII triples the size for no reader gain."""
    records = np.empty(len(xyz), dtype=PLY_DTYPE)
    records["x"], records["y"], records["z"] = xyz.T
    records["red"], records["green"], records["blue"] = rgb.T

    header = ("ply\n"
              "format binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")

    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        records.tofile(handle)


def render_cloud(xyz, rgb, size, yaw_deg, pitch_deg):
    """Painter's-algorithm splat render. No OpenGL, no extra dependency."""
    canvas = np.zeros((size, size, 3), np.uint8)

    if len(xyz) == 0:
        return canvas

    centre = np.median(xyz, axis=0)
    centred = xyz - centre
    radius = float(np.percentile(np.linalg.norm(centred, axis=1), 90)) or 1.0

    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                   [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]])
    rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])

    viewed = centred @ (rx @ ry).T
    viewed[:, 2] += 1.6 * radius

    keep = viewed[:, 2] > 1e-3
    viewed, colours = viewed[keep], rgb[keep]

    if len(viewed) == 0:
        return canvas

    focal = 0.9 * size
    u = (focal * viewed[:, 0] / viewed[:, 2] + size * 0.5).astype(np.int32)
    v = (focal * viewed[:, 1] / viewed[:, 2] + size * 0.5).astype(np.int32)

    inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    u, v, colours, depth = u[inside], v[inside], colours[inside], viewed[inside, 2]

    # Far points written first, so near points overwrite them. A real z-buffer
    # would be correct for splats larger than a pixel; these are one pixel.
    order = np.argsort(-depth)
    canvas[v[order], u[order]] = colours[order][:, ::-1]

    return canvas


def depth_panel(frame, disparity, width):
    """Frame beside its depth, at a common width, for eyeballing plausibility."""
    low, high = np.percentile(disparity, (2.0, 98.0))
    scaled = np.clip((disparity - low) / max(high - low, 1e-6), 0.0, 1.0)
    coloured = cv2.applyColorMap((scaled * 255).astype(np.uint8),
                                 cv2.COLORMAP_INFERNO)

    stacked = np.hstack([frame, coloured])
    height = int(stacked.shape[0] * width / stacked.shape[1])

    return cv2.resize(stacked, (width, height))


# ---------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--out", default="out", help="output directory")
    parser.add_argument("--stride-s", type=float, default=0.5,
                        help="seconds between sampled frames (default 0.5)")
    parser.add_argument("--max-frames", type=int, default=12,
                        help="frames to run the model on (default 12)")
    parser.add_argument("--point-stride", type=int, default=3,
                        help="keep every Nth pixel per axis (default 3)")
    parser.add_argument("--max-points", type=int, default=1_500_000)
    parser.add_argument("--hfov-deg", type=float, default=60.0,
                        help="ASSUMED horizontal field of view, used only "
                             "when neither --focal-px nor --solve-fov is "
                             "given. Disputed on IMG_0300.MOV: see README")
    parser.add_argument("--focal-px", type=float,
                        help="focal length in pixels, as calibrate_fov.py "
                             "reports it. Overrides --hfov-deg")
    parser.add_argument("--solve-fov", action="store_true",
                        help="measure the focal length from the sampled "
                             "frames instead of assuming it")
    parser.add_argument("--near-m", type=float, default=0.5,
                        help="depth assigned to the nearest surface. Sets the "
                             "arbitrary scale of the whole cloud")
    parser.add_argument("--far-m", type=float, default=12.0,
                        help="depth clamp. Points beyond it are dropped")
    parser.add_argument("--baseline-m", type=float, default=0.0,
                        help="ASSUMED camera translation between sampled "
                             "frames. 0 stitches by rotation only, which is "
                             "correct for a panning camera (default 0)")
    parser.add_argument("--beta-frac", type=float, default=0.0,
                        help="affine offset of the depth model, in units of "
                             "the robust disparity range. 0 is the old "
                             "assumption. Measured best on IMG_0300.MOV: see "
                             "README (default 0)")
    parser.add_argument("--keep-depth-pct", type=float, default=100.0,
                        help="drop points beyond this percentile of depth, "
                             "cutting the sparse far tail (default 100, off)")
    parser.add_argument("--denoise", type=int, default=0, metavar="N",
                        help="drop points whose voxel holds fewer than N "
                             "points (default 0, off)")
    parser.add_argument("--voxel-frac", type=float, default=0.004,
                        help="denoise voxel size, as a fraction of cloud "
                             "extent (default 0.004)")
    parser.add_argument("--report-geometry", action="store_true",
                        help="score the cloud against a corridor's known "
                             "flat, parallel, square surfaces")
    parser.add_argument("--single", action="store_true",
                        help="reconstruct only the middle frame, skipping "
                             "all multi-frame alignment")
    parser.add_argument("--degrade", action="store_true",
                        help="push frames through the ESP32-CAM losses first, "
                             "to test depth on the feed the robot will send")
    parser.add_argument("--device", default="auto",
                        choices=("auto", "mps", "cpu", "cuda"))
    parser.add_argument("--model", default=MODEL)

    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem

    frames, fps, stride, total = sample_frames(args.video, args.stride_s,
                                               args.max_frames)

    if not frames:
        raise SystemExit(f"no frames read from {args.video}")

    images = [frame for _, frame in frames]

    if args.degrade:
        degrader = Degrader()
        images = [degrader(frame) for frame in images]

    if args.single:
        images = [images[len(images) // 2]]

    height, width = images[0].shape[:2]

    print(f"  video    {args.video}  {width}x{height}  {total} frames")
    print(f"  sampled  {len(images)} frames, every {stride} "
          f"({stride / fps:.2f} s apart)")

    focal, focal_source = focal_from_hfov(width, args.hfov_deg), "assumed"

    if args.solve_fov:
        from calibrate_fov import estimate_focal

        solved = estimate_focal(images)

        if solved is None:
            print("  fov      MEASUREMENT FAILED, falling back to "
                  f"--hfov-deg {args.hfov_deg:.0f}")
        else:
            focal, focal_source = solved["focal_px"], "measured"
            print(f"  fov      {focal:.0f} px measured from {solved['samples']} "
                  f"vanishing-point pairs, 95% CI "
                  f"[{solved['low_px']:.0f}, {solved['high_px']:.0f}]")

    if args.focal_px:
        focal, focal_source = args.focal_px, "given"

    print(f"  focal    {focal:.0f} px ({focal_source}), "
          f"{math.degrees(2 * math.atan(0.5 * width / focal)):.1f} deg across "
          f"the {width} px axis")

    K = intrinsics(width, height, focal)

    device = pick_device(args.device)
    load_start = time.perf_counter()
    torch, processor, model = load_model(args.model, device)
    print(f"  model    {args.model} on {device}, "
          f"loaded in {time.perf_counter() - load_start:.1f} s")

    # The first call compiles kernels and allocates the MPS graph. Timing it
    # with the rest would roughly double the reported per-frame cost and
    # misinform the decision this number exists to support.
    warm_start = time.perf_counter()
    predict_disparity(torch, processor, model, device, images[0])
    synchronise(torch, device)
    warmup_s = time.perf_counter() - warm_start

    disparities, timings = [], []

    for image in images:
        start = time.perf_counter()
        disparity = predict_disparity(torch, processor, model, device, image)
        synchronise(torch, device)
        timings.append(time.perf_counter() - start)
        disparities.append(disparity)

    print(f"  depth    {np.median(timings) * 1000:.0f} ms/frame median, "
          f"{min(timings) * 1000:.0f} ms best, {max(timings) * 1000:.0f} ms "
          f"worst, {warmup_s * 1000:.0f} ms first call (excluded)")
    print(f"           {1.0 / np.median(timings):.1f} fps sustained at "
          f"{width}x{height}")

    detector = cv2.ORB_create(nfeatures=2000, fastThreshold=12)
    failures = 0

    if len(images) > 1:
        poses, failures = chain_poses(images, K, args.baseline_m, detector)
    else:
        poses = [(np.eye(3), np.zeros(3))]

    clouds, colours, depths = [], [], []

    for image, disparity, (rotation, centre) in zip(images, disparities, poses):
        depth = to_depth(disparity, args.near_m, args.far_m, args.beta_frac)
        xyz, rgb = back_project(depth, image, K, args.point_stride)

        # The clamp value is not a measurement, it is "we gave up". Keeping
        # those points would paste a solid wall across the back of the cloud.
        # A non-positive z is the same admission from the other end.
        keep = (xyz[:, 2] < args.far_m * 0.999) & (xyz[:, 2] > 0.0)
        xyz, rgb = xyz[keep], rgb[keep]

        depths.append(xyz[:, 2].copy())          # depth in ITS OWN camera, pre-pose
        clouds.append(xyz @ rotation.T + centre)
        colours.append(rgb)

    xyz = np.concatenate(clouds).astype(np.float32)
    rgb = np.concatenate(colours).astype(np.uint8)
    depth_axis = np.concatenate(depths)
    raw_count = len(xyz)

    xyz, rgb, cut = trim_tail(xyz, rgb, depth_axis, args.keep_depth_pct)

    if cut is not None:
        print(f"  trim     kept depth <= {cut:.2f} m "
              f"(p{args.keep_depth_pct:g}), dropped {raw_count - len(xyz):,} "
              f"of {raw_count:,} points")

    xyz, rgb, dropped = voxel_denoise(xyz, rgb, args.voxel_frac, args.denoise)

    if dropped:
        print(f"  denoise  dropped {dropped:,} points in voxels holding "
              f"fewer than {args.denoise}")

    if len(xyz) > args.max_points:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(xyz), args.max_points, replace=False)
        xyz, rgb = xyz[pick], rgb[pick]

    ply_path = out / f"{stem}_cloud.ply"
    write_ply(ply_path, xyz, rgb)

    middle = len(images) // 2
    panel = depth_panel(images[middle], disparities[middle], 960)
    views = np.hstack([render_cloud(xyz, rgb, 480, 0.0, 0.0),
                       render_cloud(xyz, rgb, 480, 35.0, 15.0)])

    preview = np.vstack([panel, views])
    preview_path = out / f"{stem}_preview.png"
    cv2.imwrite(str(preview_path), preview)

    print(f"  cloud    {len(xyz):,} points -> {ply_path}")
    print(f"  preview  {preview_path}  "
          f"(top: frame and depth. bottom: cloud from two angles)")

    if args.report_geometry:
        import corridor

        print(corridor.format_report(corridor.measure(xyz.astype(np.float64))))
        print("           bow and rms are percentages of the wall's own "
              "extent, so they survive")
        print("           the unknown scale. Lower is flatter. Parallelism "
              "always improves with")
        print("           focal length and cannot be used to choose one.")

    if failures:
        print(f"  WARNING  {failures} of {len(images) - 1} frame pairs gave "
              f"no pose. Those frames reuse the previous viewpoint.")

    if len(images) > 1 and args.baseline_m == 0.0:
        print("  NOTE     baseline 0: frames are stitched by rotation only. "
              "Correct for a")
        print("           panning camera, wrong for a walking one. Pass "
              "--baseline-m for that.")

    if focal_source == "assumed":
        print(f"  WARNING  focal length is ASSUMED from --hfov-deg "
              f"{args.hfov_deg:.0f}, not measured.")
        print("           calibrate_fov.py measures 1641 px on IMG_0300.MOV "
              "against this default's")
        print("           935 px; a COLMAP run measured 983 px. That "
              "disagreement is unresolved")
        print("           (see README). Run calibrate_fov.py or pass "
              "--solve-fov to decide it.")

    print(f"  NOTE     scale is arbitrary (near surface pinned to "
          f"{args.near_m} m). Distances in this")
    print("           cloud are not measurements.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
