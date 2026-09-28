"""Prove the marker pose chain numerically, with no hardware.

    ./.venv/bin/python test_markers.py

The core of this file is a round trip. Put a robot at a pose we chose,
project the real marker corners into the image with cv2.projectPoints, hand
those corners to the solver, and demand the original pose back. An inverted
transform or a swapped frame cannot survive that: it does not raise, it
returns a confident pose pointing somewhere else, and on hardware it looks
like a robot that drives the wrong way for reasons nobody can see.

The forward model (markers.project_markers) is written from the map and the
mount directly and shares no code with the solver, so a sign error cannot
cancel itself out across the two.
"""

import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

import markers as mk
from geometry import Pose


np.random.seed(7)                  # the noise trials must give the same verdict twice

W, H = 320, 240                    # QVGA, what the robots actually send
ARENA_X, ARENA_Y = 3.0, 2.5
TAG_H = 0.25                       # tag centre height, metres
MOUNT = mk.CameraMount(z=0.10)


def arena_map(size_m=0.15):
    """A rectangular arena with tags along every wall, facing inward."""
    m = mk.MarkerMap()
    mid = 0

    # (wall coordinate generator, yaw the face points)
    walls = [
        ([(x, 0.0) for x in (0.5, 1.2, 1.9, 2.6)], 90.0),      # y=0 wall, faces +y
        ([(x, ARENA_Y) for x in (0.5, 1.2, 1.9, 2.6)], -90.0),  # y=max, faces -y
        ([(0.0, y) for y in (0.5, 1.25, 2.0)], 0.0),            # x=0 wall, faces +x
        ([(ARENA_X, y) for y in (0.5, 1.25, 2.0)], 180.0),      # x=max, faces -x
    ]

    for positions, yaw in walls:
        for (x, y) in positions:
            m.add(mk.MarkerPlacement(mid, x, y, TAG_H, yaw, size_m,
                                     note=f"wall yaw {yaw:+.0f}"))
            mid += 1

    return m


def pose_error(got: Pose, want: Pose):
    """(metres, degrees)."""
    d = math.hypot(got.x - want.x, got.y - want.y)
    a = math.atan2(math.sin(got.theta - want.theta), math.cos(got.theta - want.theta))
    return d, abs(math.degrees(a))


MAP = arena_map()
K, DIST = mk.default_camera_matrix(W, H)


print("marker pose checks\n")

# --- 1. cv2.aruco is present and the tags we intend to print exist ---
assert hasattr(cv2, "aruco"), "no cv2.aruco: install opencv-contrib-python"
_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, mk.MARKERS.dictionary))
assert cv2.aruco.generateImageMarker(_dict, 0, 60).shape == (60, 60)
print(f"  cv2 {cv2.__version__}, aruco present, {mk.MARKERS.dictionary} generates")

# --- 2. transform algebra ---
T = mk._rt(mk._rz(0.7) @ mk._ry(-0.2), [1.0, -2.0, 0.3])
assert np.allclose(T @ mk._inv(T), np.eye(4), atol=1e-12), "rigid inverse is wrong"

p = Pose(1.3, -0.4, 2.0)
assert pose_error(mk._flatten(mk.body_transform(p)), p) < (1e-12, 1e-9)
print("  rigid inverse and flatten/unflatten are exact round trips")

# A tag on the y=0 wall faces +y, its printed 'up' is world +z, and its
# printed 'right' therefore runs along world -x. Hand-checked, because every
# later result is built on this rotation.
probe = mk.MarkerPlacement(0, 1.0, 0.0, 0.5, 90.0, 0.20)
corners = probe.world_points()
assert np.allclose(probe.rotation[:, 2], [0.0, 1.0, 0.0], atol=1e-12), "face normal"
assert np.allclose(probe.rotation[:, 1], [0.0, 0.0, 1.0], atol=1e-12), "tag up"
assert np.allclose(corners[0], [1.1, 0.0, 0.6], atol=1e-12), corners[0]   # top-left
assert np.allclose(corners[2], [0.9, 0.0, 0.4], atol=1e-12), corners[2]   # bottom-right
assert np.allclose(np.linalg.norm(corners[0] - corners[1]), 0.20)
print("  wall-tag corners land where a tape measure says they do")

# --- 3. marker map file round trip ---
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "marker_map.json")
    MAP.save(path)
    reloaded = mk.MarkerMap.load(path)
    assert reloaded.ids == MAP.ids
    for i in MAP.ids:
        assert np.allclose(reloaded[i].world_points(), MAP[i].world_points())
    assert mk.load_marker_map(os.path.join(tmp, "nope.json")).ids == []
    hand_written = mk.MarkerMap.from_dict(
        {"default_size_m": 0.2, "markers": [{"id": 4, "x": 1, "y": 0, "yaw_deg": 90}]})
    assert hand_written[4].size_m == 0.2 and hand_written[4].z == 0.0
print(f"  map file round trips {len(MAP)} markers; missing file -> empty map")

# --- 4. degenerate inputs return None, never an exception ---
assert mk.pose_from_markers([], MAP, K, DIST, MOUNT) is None
assert mk.pose_from_markers(None, MAP, K, DIST, MOUNT) is None
assert mk.detect(None) == [] and mk.detect(np.zeros((0, 0), np.uint8)) == []
assert mk.detect(np.full((H, W, 3), 128, np.uint8)) == []

stranger = mk.MarkerDetection(999, np.array([[10, 10], [40, 10], [40, 40], [10, 40]],
                                            np.float32))
assert mk.pose_from_markers([stranger], MAP, K, DIST, MOUNT) is None

tiny = mk.MarkerDetection(0, np.array([[10, 10], [16, 10], [16, 16], [10, 16]],
                                      np.float32))
assert mk.pose_from_markers([tiny], MAP, K, DIST, MOUNT) is None
assert mk.pose_from_markers([stranger], mk.MarkerMap(), K, DIST, MOUNT) is None
print("  no markers / unknown id / sub-pixel-sized tag / empty map -> None")

# --- 5. heading convention: theta=0 must mean facing world +x ---
facing = {}
for theta_deg, expect in ((0.0, "+x"), (90.0, "+y"), (180.0, "-x"), (-90.0, "-y")):
    p = Pose(1.5, 1.25, math.radians(theta_deg))
    dets = mk.project_markers(p, MAP, K, DIST, MOUNT, W, H)
    walls = {round(MAP[d.id].yaw_deg) for d in dets}
    facing[expect] = sorted(walls)

# A robot facing +x sees the wall at x=max, whose tags face back at -x (180).
assert facing["+x"] == [180], facing
assert facing["-x"] == [0], facing
assert facing["+y"] == [-90], facing
assert facing["-y"] == [90], facing
print("  theta=0 faces world +x; each heading sees the wall it should")

# --- 6. THE ROUND TRIP: exact corners, a grid of poses ---
print("\n  pose recovery from exact projected corners:")
print("    x      y     theta      n  pos err      heading err   reproj")

grid = [(x, y, t)
        for x in (0.6, 1.5, 2.4)
        for y in (0.6, 1.25, 1.9)
        for t in (0, 37, 90, 143, 180, -128, -90, -22)]

worst_pos = worst_head = 0.0
single_only = multi = 0
in_view = 0
lost = []

for (x, y, t) in grid:
    truth = Pose(x, y, math.radians(t))
    dets = mk.project_markers(truth, MAP, K, DIST, MOUNT, W, H)

    if not dets:
        continue

    in_view += 1
    est = mk.estimate_pose(dets, MAP, K, DIST, MOUNT)

    # Corners this clean are never ill-conditioned, so a rejection here is the
    # solver refusing a pose it should have recovered to machine precision.
    # Counted rather than asserted on the spot, so a regression reports how
    # many poses it lost and where, not just the first one.
    if est is None:
        lost.append((x, y, t))
        continue

    dpos, dhead = pose_error(est.pose, truth)
    worst_pos, worst_head = max(worst_pos, dpos), max(worst_head, dhead)

    if est.n_markers == 1:
        single_only += 1
    else:
        multi += 1

    assert dpos < 2e-3, f"{truth} -> {est.pose}, off by {dpos*1000:.2f} mm"
    assert dhead < 0.2, f"{truth} -> {est.pose}, off by {dhead:.3f} deg"
    assert abs(est.camera_height_m - MOUNT.z) < 2e-3, est.camera_height_m
    assert est.body_tilt_deg < 0.5, est.body_tilt_deg
    assert est.reprojection_error_px < 0.05, est.reprojection_error_px

    if t in (0, 90) and y == 1.25:
        print(f"   {x:4.1f}  {y:5.2f}  {t:+6.0f}deg  {est.n_markers:2d}  "
              f"{dpos*1000:8.4f} mm  {dhead:8.5f} deg  {est.reprojection_error_px:.2e} px")

print(f"    {in_view} of {len(grid)} poses saw a tag, "
      f"{single_only} single-marker, {multi} multi-marker")
print(f"    WORST position error {worst_pos*1000:.4f} mm, "
      f"WORST heading error {worst_head:.5f} deg, "
      f"{len(lost)} rejected")
assert not lost, f"lost the pose at exact corners: {lost}"

# --- 7. the same round trip with a single marker only, forced ---
print("\n  single marker in view, forced:")
worst_pos_1 = worst_head_1 = 0.0
count_1 = 0
lost_1 = []

for (x, y, t) in grid:
    truth = Pose(x, y, math.radians(t))
    dets = mk.project_markers(truth, MAP, K, DIST, MOUNT, W, H)

    if not dets:
        continue

    biggest = max(dets, key=lambda d: d.side_px)
    est = mk.estimate_pose([biggest], MAP, K, DIST, MOUNT)

    # One tag is the hardest case for the ambiguity, but with exact corners
    # the true branch fits ~1e-6 px and the false one does not come close, so
    # there is still nothing here the solver is entitled to be unsure about.
    if est is None:
        lost_1.append((x, y, t, biggest.id))
        continue

    assert est.n_markers == 1

    dpos, dhead = pose_error(est.pose, truth)
    worst_pos_1, worst_head_1 = max(worst_pos_1, dpos), max(worst_head_1, dhead)
    count_1 += 1

    # The planar ambiguity makes a single tag a genuinely worse instrument,
    # but it must never flip: a heading error near 180 deg is the failure
    # this whole file exists to catch.
    assert dhead < 5.0, f"single marker {biggest.id} at {truth} flipped: {dhead:.1f} deg"

print(f"    {count_1} poses, WORST position error {worst_pos_1*1000:.4f} mm, "
      f"WORST heading error {worst_head_1:.5f} deg, {len(lost_1)} rejected")
assert not lost_1, f"single markers rejected on exact corners: {lost_1}"

# --- 8. when the geometry cannot say, the solver must refuse, not guess ---
# The robot drives on this pose. A confidently wrong one is worse than none
# at all, because "none" leaves the caller holding its last known pose and
# its age, while "wrong" is acted on immediately.
#
# One small tag, alone, at 2 m, on a QVGA frame is where that bites: the two
# branches of the square-marker ambiguity fit the noisy corners equally well
# and neither the fit nor the mount height can separate them. The test is
# not that the solver is accurate there -- it cannot be -- but that the
# poses it hands back are better than the ones it refuses.
print("\n  ill-conditioned single tags are refused, not guessed:")
print("    sigma   n    refused      worst KEPT        worst REFUSED")

for sigma in (0.5, 1.0):
    kept, refused = [], []

    for trial in range(400):
        x = np.random.uniform(0.4, 2.6)
        y = np.random.uniform(0.4, 2.1)
        truth = Pose(x, y, np.random.uniform(-math.pi, math.pi))

        clean = mk.project_markers(truth, MAP, K, DIST, MOUNT, W, H)

        if not clean:
            continue

        biggest = max(clean, key=lambda d: d.side_px)
        jittered = mk.MarkerDetection(
            biggest.id, (biggest.corners
                         + np.random.normal(0, sigma, biggest.corners.shape)
                         ).astype(np.float32))

        est = mk.estimate_pose([jittered], MAP, K, DIST, MOUNT)

        if est is not None:
            kept.append(pose_error(est.pose, truth))
            continue

        # What the solver WOULD have returned with no gate: the branch that
        # fits the corners best. Refusing is only worth anything if these are
        # the poses being refused.
        branches = mk._solve_single(MAP[jittered.id], jittered, K, DIST, MOUNT)

        if branches:
            best = min(branches, key=lambda c: c[2])
            refused.append(pose_error(mk._flatten(best[1]), truth))

    n = len(kept) + len(refused)
    worst_kept = max(kept)
    worst_refused = max(refused)

    print(f"    {sigma:4.1f} px {n:4d}  {len(refused):3d} ({100*len(refused)/n:4.1f}%)  "
          f"{worst_kept[0]*100:6.1f} cm {worst_kept[1]:6.2f} deg  "
          f"{worst_refused[0]*100:6.1f} cm {worst_refused[1]:6.2f} deg")

    assert refused, f"nothing refused at sigma={sigma}: the gate is not armed"
    assert worst_refused[0] > worst_kept[0], (
        f"at sigma={sigma} the refused poses ({worst_refused[0]*100:.0f} cm) were no "
        f"worse than the kept ones ({worst_kept[0]*100:.0f} cm): the gate is "
        f"discarding poses at random")

# A tag seen almost edge-on is the textbook ill-conditioned case: the corners
# barely move as the camera swings, so the two branches coincide.
grazing = mk.MarkerPlacement(0, 1.0, 0.0, TAG_H, 90.0, 0.15)
edge_on = mk.MarkerDetection(0, np.array([[150.0, 110.0], [154.0, 109.0],
                                          [154.0, 130.0], [150.0, 131.0]],
                                         np.float32))
solo = mk.MarkerMap(dictionary=MAP.dictionary)
solo.add(grazing)
assert mk.estimate_pose([edge_on], solo, K, DIST, MOUNT) is None, \
    "a 4 px-wide sliver of a tag produced a confident pose"
print("    a tag seen edge-on as a 4 px sliver -> None, not a guess")

# --- 9. a panned and tilted camera, and an offset mount ---
print("\n  non-trivial camera mounts:")
for label, mount in (
        ("straight ahead   ", mk.CameraMount(z=0.10)),
        ("panned +25 deg   ", mk.CameraMount(z=0.10, pan_rad=math.radians(25))),
        ("panned -40 deg   ", mk.CameraMount(z=0.10, pan_rad=math.radians(-40))),
        ("tilted up 12 deg ", mk.CameraMount(z=0.10, tilt_rad=math.radians(12))),
        ("offset + panned  ", mk.CameraMount(x=0.06, y=-0.03, z=0.14,
                                             pan_rad=math.radians(15),
                                             tilt_rad=math.radians(-8))),
):
    worst = 0.0
    seen = 0

    for (x, y, t) in grid:
        truth = Pose(x, y, math.radians(t))
        dets = mk.project_markers(truth, MAP, K, DIST, mount, W, H)

        if not dets:
            continue

        est = mk.estimate_pose(dets, MAP, K, DIST, mount)
        assert est is not None
        dpos, dhead = pose_error(est.pose, truth)
        worst = max(worst, dpos)
        seen += 1

        assert dpos < 2e-3, f"{label} {truth}: {dpos*1000:.3f} mm"
        assert dhead < 0.2, f"{label} {truth}: {dhead:.3f} deg"
        assert abs(est.camera_height_m - mount.z) < 2e-3

    print(f"    {label} {seen:3d} poses  worst {worst*1000:.4f} mm")

# A mount used to solve that is not the mount used to project must move the
# answer. Without this, a mount silently ignored would pass every test above.
truth = Pose(1.5, 1.0, math.radians(40))
panned = mk.CameraMount(z=0.10, pan_rad=math.radians(30))
dets = mk.project_markers(truth, MAP, K, DIST, panned, W, H)
wrong = mk.pose_from_markers(dets, MAP, K, DIST, mk.CameraMount(z=0.10))
assert abs(math.degrees(wrong.theta - truth.theta)) > 25, \
    "the mount pan is being ignored"
print(f"    ignoring a 30 deg pan shifts heading by "
      f"{abs(math.degrees(wrong.theta - truth.theta)):.1f} deg, as it must")

# --- 10. lens distortion is handled, not assumed away ---
K_real = np.array([[248.0, 0.0, 154.0], [0.0, 250.5, 121.0], [0.0, 0.0, 1.0]])
DIST_REAL = np.array([[-0.34], [0.12], [0.0008], [-0.0011], [-0.02]])

worst = 0.0
for (x, y, t) in grid:
    truth = Pose(x, y, math.radians(t))
    dets = mk.project_markers(truth, MAP, K_real, DIST_REAL, MOUNT, W, H)

    if not dets:
        continue

    est = mk.estimate_pose(dets, MAP, K_real, DIST_REAL, MOUNT)
    assert est is not None
    dpos, dhead = pose_error(est.pose, truth)
    worst = max(worst, dpos)
    assert dpos < 2e-3 and dhead < 0.2, f"distorted {truth}: {dpos*1000:.3f} mm"

print(f"\n  with a real distortion model (k1=-0.34): worst {worst*1000:.4f} mm")

# --- 11. multi-marker fusion must beat a single marker on noisy corners ---
# This is the claim the arena layout is built on, so measure it rather than
# asserting it. Sigma is in pixels of corner position, which is where real
# error enters: blur, rolling shutter and JPEG all land on the corners.
print("\n  multi-marker fusion vs single marker, noisy corners:")
print("    sigma   n     single-marker         fused             gain      refused")

TRIALS = 240
noisy_gain = []

for sigma in (0.2, 0.5, 1.0):
    s_pos = s_head = f_pos = f_head = 0.0
    used = 0
    n_markers_seen = []
    lost_one = lost_many = seen = 0

    for trial in range(TRIALS):
        x = np.random.uniform(0.5, 2.5)
        y = np.random.uniform(0.5, 2.0)
        t = np.random.uniform(-math.pi, math.pi)
        truth = Pose(x, y, t)

        clean = mk.project_markers(truth, MAP, K, DIST, MOUNT, W, H)

        if len(clean) < 2:
            continue

        jittered = [mk.MarkerDetection(
            d.id, (d.corners + np.random.normal(0, sigma, d.corners.shape)
                   ).astype(np.float32)) for d in clean]

        biggest = max(jittered, key=lambda d: d.side_px)
        one = mk.estimate_pose([biggest], MAP, K, DIST, MOUNT)
        many = mk.estimate_pose(jittered, MAP, K, DIST, MOUNT)

        seen += 1
        lost_one += one is None
        lost_many += many is None

        # Averaged over the frames where BOTH answered, so the comparison is
        # accuracy alone. Availability is the other half and is printed too:
        # a lone tag is refused far more often than a fused set, which is the
        # same weakness showing up as silence instead of as error.
        if one is None or many is None:
            continue

        dp, dh = pose_error(one.pose, truth)
        s_pos, s_head = s_pos + dp, s_head + dh
        dp, dh = pose_error(many.pose, truth)
        f_pos, f_head = f_pos + dp, f_head + dh

        n_markers_seen.append(many.n_markers)
        used += 1

    s_pos, s_head = s_pos / used, s_head / used
    f_pos, f_head = f_pos / used, f_head / used
    noisy_gain.append((sigma, s_pos, f_pos, s_head, f_head))

    print(f"    {sigma:4.1f} px {np.mean(n_markers_seen):4.1f}  "
          f"{s_pos*100:6.2f} cm {s_head:5.2f} deg   "
          f"{f_pos*100:6.2f} cm {f_head:5.2f} deg   "
          f"{s_pos/f_pos:4.2f}x pos  {s_head/f_head:4.2f}x head  "
          f"{100*lost_one/seen:4.1f}% vs {100*lost_many/seen:4.1f}%")

    assert f_pos < s_pos, (
        f"fusion did not improve position at sigma={sigma}: "
        f"{f_pos*100:.2f} cm vs {s_pos*100:.2f} cm")
    assert f_head < s_head, (
        f"fusion did not improve heading at sigma={sigma}: "
        f"{f_head:.2f} deg vs {s_head:.2f} deg")

assert noisy_gain[-1][1] / noisy_gain[-1][2] > 1.3, \
    "fusion gain at 1 px noise is too small to justify the extra tags"

# --- 12. a mis-measured map entry is rejected, not averaged in ---
bad = mk.MarkerMap(dictionary=MAP.dictionary)
for i in MAP.ids:
    bad.add(MAP[i])

victim = 1
truth_placement = MAP[victim]
bad.add(mk.MarkerPlacement(victim, truth_placement.x + 0.35, truth_placement.y,
                           truth_placement.z, truth_placement.yaw_deg,
                           truth_placement.size_m))

found_case = False
for (x, y, t) in grid:
    truth = Pose(x, y, math.radians(t))
    dets = mk.project_markers(truth, MAP, K, DIST, MOUNT, W, H)

    if victim not in [d.id for d in dets] or len(dets) < 3:
        continue

    est = mk.estimate_pose(dets, bad, K, DIST, MOUNT)
    dpos, dhead = pose_error(est.pose, truth)

    assert victim in est.rejected_ids, \
        f"a marker 35 cm off its map entry was kept: {est}"
    assert dpos < 0.02, f"outlier dragged the pose {dpos*100:.1f} cm"
    found_case = True

assert found_case, "no test pose saw the corrupted marker with 2 good ones"
print(f"\n  a marker 35 cm off its measured position is rejected, "
      f"pose stays within 2 cm")

# --- 13. detect() on rendered tags: the real OpenCV detector, end to end ---
print("\n  detect() -> pose on rendered QVGA frames:")
rendered = 0
worst_render = 0.0

for (x, y, t) in [(1.5, 1.25, 90), (1.0, 0.7, 60), (2.0, 1.6, -100),
                  (1.5, 1.25, -90), (0.8, 1.25, 20)]:
    truth = Pose(x, y, math.radians(t))
    projected = mk.project_markers(truth, MAP, K, DIST, MOUNT, W, H)

    if not projected:
        continue

    frame = mk.render_markers(projected, MAP, W, H)
    est, found = mk.pose_from_frame(frame, MAP, K, DIST, MOUNT)

    assert found, f"detector found nothing at {truth}"
    assert set(d.id for d in found) <= set(d.id for d in projected), "phantom id"
    assert est is not None

    dpos, dhead = pose_error(est.pose, truth)
    worst_render = max(worst_render, dpos)
    rendered += 1

    print(f"    at ({x:.1f}, {y:.2f}, {t:+.0f}deg)  detected "
          f"{len(found)}/{len(projected)} tags  ->  {dpos*100:5.2f} cm  "
          f"{dhead:5.2f} deg  ({est.reprojection_error_px:.2f} px)")

    # Loose on purpose. This path includes real corner extraction from a
    # 320x240 render, so it measures the detector, not the transform chain.
    assert dpos < 0.25, f"rendered frame pose off by {dpos*100:.1f} cm"
    assert dhead < 15.0, f"rendered frame heading off by {dhead:.1f} deg"

assert rendered >= 4
print(f"    worst {worst_render*100:.2f} cm -- this path includes real corner "
      f"extraction, so it is the honest number")

# --- 14. annotate() renders, for the operator display ---
frame = mk.render_markers(mk.project_markers(Pose(1.5, 1.25, math.radians(90)),
                                             MAP, K, DIST, MOUNT, W, H),
                          MAP, W, H)
est, found = mk.pose_from_frame(frame, MAP, K, DIST, MOUNT)
out = mk.annotate(frame, found, est)
assert out.shape == (H, W, 3)
assert mk.annotate(frame, [], None).shape == (H, W, 3)
cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "markers_snapshot.png"), out)
print(f"\n  annotate renders {out.shape[1]}x{out.shape[0]} -> markers_snapshot.png")

# --- 15. uncalibrated intrinsics: what a wrong focal length costs ---
# The team will be tempted to skip calibration. Price it.
K_guess, _ = mk.default_camera_matrix(W, H)
err_guess = []

for (x, y, t) in grid:
    truth = Pose(x, y, math.radians(t))
    dets = mk.project_markers(truth, MAP, K_real, DIST_REAL, MOUNT, W, H)

    if not dets:
        continue

    est = mk.estimate_pose(dets, MAP, K_guess, None, MOUNT)   # solved with the guess

    if est is None:
        continue

    err_guess.append(pose_error(est.pose, truth))

mean_pos = np.mean([e[0] for e in err_guess])
mean_head = np.mean([e[1] for e in err_guess])
print(f"  solving real frames with guessed intrinsics: "
      f"{mean_pos*100:.1f} cm and {mean_head:.1f} deg mean error "
      f"({len(err_guess)} poses)")
assert mean_pos > 0.02, "the guessed-intrinsics penalty vanished; check the test"

print("\nALL MARKER CHECKS PASSED")
