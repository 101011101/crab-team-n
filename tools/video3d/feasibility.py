"""Decide whether a video can be reconstructed, before anyone downloads a model.

    python3 feasibility.py ../../samples/arena.mp4

Needs only opencv-python and numpy, so it runs in central/.venv with no new
installs. That is the point: it is the cheap gate in front of a multi-hundred
megabyte dependency tree.

Four measurements, in rising order of how much they tell you:

  blur          variance of the Laplacian. A blurred frame has no high
                frequency content, so the second derivative is flat.
  texture       ORB keypoint count. A blank wall gives a model nothing to
                anchor on, whatever the depth network claims to see.
  trackability  inter-frame matches surviving RANSAC. This is the single best
                predictor of convergence: it is the same correspondence set
                every downstream method actually consumes.
  parallax      does the camera TRANSLATE or only ROTATE. A camera spinning on
                its own optical centre sees no depth at all, at any resolution,
                with any model. This is the most common way an arena video is
                dead on arrival, and it looks fine to the human who shot it.

SCALE. The first two are resolution dependent, so their raw values are not
comparable between a 1080p frame and its own 320x240 copy:

  var(Laplacian) RISES when you downscale. Averaging pixels away concentrates
  the surviving edge energy into fewer pixels, so the small copy always scores
  sharper than the large one it was made from. Measured here: 14 at 1080x1920,
  158 for the same frames resampled to a 400 px diagonal.
  ORB count rises with pixel count, and also rises with JPEG blocking, which
  manufactures corners out of block edges. Measured here: quality 12 took the
  same frames from 146 keypoints to 266 while var(Laplacian) fell from 158 to
  119. More keypoints, less image.

So both frame metrics are also measured on one common grid (REF_DIAG) and the
absolute thresholds are applied there. Native values are reported as
diagnostics only and are never differenced. Pixel distances are reported as a
fraction of the image diagonal for the same reason.

Thresholds here are priors, not measurements. They were calibrated against
synthetic video (make_test_video.py), never against ESP32-CAM footage. Treat
the numbers as the output and the verdict as a hint.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np


# ---------------------------------------------------------------
# SCALE
# ---------------------------------------------------------------

ESP32_SIZE = (320, 240)
ESP32_JPEG_QUALITY = 12

# Every absolute number below -- a variance, a keypoint count -- only means
# something at one image size. Pick QVGA's diagonal as that size: it is what
# the robot camera will actually deliver, so a frame that passes here passes
# at the resolution the demo runs at. A phone frame is resampled down to it,
# and is then being judged on the detail that survives to the robot, which is
# the only detail that matters.
REF_DIAG = float(np.hypot(*ESP32_SIZE))


def to_reference(gray, ref_diag=REF_DIAG):
    """Resample to the common grid, preserving aspect."""
    height, width = gray.shape[:2]
    scale = ref_diag / float(np.hypot(width, height))

    if abs(scale - 1.0) < 0.02:
        return gray

    # INTER_AREA averages the discarded pixels; INTER_LINEAR is only reached
    # when the source is smaller than QVGA, where upsampling adds no detail
    # but does put both runs on one grid, which is the whole point.
    size = (max(16, int(round(width * scale))), max(16, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    return cv2.resize(gray, size, interpolation=interpolation)


# ---------------------------------------------------------------
# THRESHOLDS
# ---------------------------------------------------------------
# Pixel thresholds are fractions of the image diagonal so that a phone video
# and a 320x240 robot video are judged on the same physical scale. Blur and
# keypoint floors are absolute, so they are applied to the reference-grid
# measurement, never to the native one.

@dataclass
class Thresholds:
    blur_min: float = 60.0          # variance of Laplacian below this is soft
    kp_min: int = 150               # ORB keypoints below this is thin texture

    track_hard: int = 25            # RANSAC inliers below this: nothing works
    track_soft: int = 60            # below this: expect a sparse, holey result
    ratio_soft: float = 0.50        # inliers / raw matches

    flow_min_diag: float = 0.0025   # motion below this: camera is standing still
    parallax_diag: float = 0.0050   # residual above this: real translation
    rotate_diag: float = 0.0025     # residual below this: rotation or flat wall

    # Essential-matrix inliers over homography inliers. A homography models
    # a rotation-only pair exactly, so it only loses ground when the motion
    # contains translation it cannot absorb. Measured on a real walk down a
    # corridor: 1.1 to 1.8. Measured on the rotation-only stretches of the
    # same clip: below 1.0.
    eh_translate: float = 1.15
    eh_rotate: float = 0.95

    # Share of the translation direction lying along the optical axis.
    # Above this the baseline points where the camera is looking, which
    # triangulates the view ahead poorly even though the motion is real.
    forward_dolly: float = 0.80

    translate_frac_hard: float = 0.25   # share of pairs that must translate
    translate_frac_soft: float = 0.50
    blur_bad_frac_soft: float = 0.30


@dataclass
class FrameMetrics:
    index: int
    time_s: float
    blur: float             # var(Laplacian) at native size. DIAGNOSTIC ONLY.
    keypoints: int          # ORB at native size. DIAGNOSTIC ONLY.
    blur_ref: float         # both again on the REF_DIAG grid: these compare
    keypoints_ref: int


@dataclass
class PairMetrics:
    index_a: int
    index_b: int
    matches: int
    inliers: int            # after RANSAC on the fundamental matrix
    inlier_ratio: float
    flow_px: float          # median correspondence displacement
    parallax_px: float      # median residual once rotation is explained away
    h_share: float          # share of the inliers a homography also explains
    eh_ratio: float = 0.0   # essential inliers over homography inliers
    forward: float = 0.0    # |tz| of the recovered translation direction, 0..1
    verdict: str = "FAILED" # TRANSLATE / DOLLY / ROTATE / STATIC / WEAK / FAILED


# ---------------------------------------------------------------
# SAMPLING
# ---------------------------------------------------------------

def sample_frames(path, stride_s, max_frames):
    """Return evenly spaced frames, read sequentially.

    Sequential reading beats seeking by frame index: long-GOP phone footage
    and anything muxed by the ESP32 sketch routinely seek to the wrong frame
    or refuse to seek at all.
    """
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Containers written by cheap encoders report 0, 1000, or nonsense here.
    if not (1.0 < fps < 240.0):
        fps = 30.0

    stride = max(1, int(round(stride_s * fps)))

    frames = []
    index = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if index % stride == 0:
            frames.append((index, frame))

            if len(frames) >= max_frames:
                break

        index += 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or index
    cap.release()

    return frames, fps, stride, total


# ---------------------------------------------------------------
# PER FRAME
# ---------------------------------------------------------------

def make_detector(nfeatures):
    # fastThreshold below the OpenCV default of 20. ESP32-CAM frames are low
    # contrast and the default finds almost nothing on them, which would read
    # as "no texture" when the real problem is only gain.
    return cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8,
                          fastThreshold=12)


def frame_metrics(index, frame, fps, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    reference = to_reference(gray)

    keypoints, descriptors = detector.detectAndCompute(gray, None)

    # Matching downstream uses the native keypoints -- those are the
    # correspondences the pipeline really gets. The reference-grid pass is a
    # second, throwaway measurement whose only job is to be comparable.
    metrics = FrameMetrics(
        index=index, time_s=index / fps,
        blur=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        keypoints=len(keypoints),
        blur_ref=float(cv2.Laplacian(reference, cv2.CV_64F).var()),
        keypoints_ref=len(detector.detect(reference, None)))

    return metrics, keypoints, descriptors


# ---------------------------------------------------------------
# PER PAIR
# ---------------------------------------------------------------

def match(desc_a, desc_b, ratio=0.75):
    """Lowe ratio test. A single nearest neighbour is not evidence of a match."""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)

    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < ratio * n.distance]

    return good


def pair_metrics(index_a, kp_a, desc_a, index_b, kp_b, desc_b, diag, thr):
    good = match(desc_a, desc_b)

    empty = PairMetrics(index_a, index_b, len(good), 0, 0.0, 0.0, 0.0, 0.0)

    # Eight correspondences is the algebraic minimum for the fundamental
    # matrix. Below it RANSAC returns something, and it means nothing.
    if len(good) < 8:
        return empty

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

    fundamental, mask_f = cv2.findFundamentalMat(
        pts_a, pts_b, cv2.FM_RANSAC,
        ransacReprojThreshold=0.006 * diag, confidence=0.999)

    if fundamental is None or mask_f is None:
        return empty

    keep = mask_f.ravel().astype(bool)
    inliers = int(keep.sum())

    result = PairMetrics(
        index_a=index_a, index_b=index_b, matches=len(good),
        inliers=inliers, inlier_ratio=inliers / float(len(good)),
        flow_px=0.0, parallax_px=0.0, h_share=0.0, verdict="FAILED")

    if inliers < 8:
        return result

    in_a, in_b = pts_a[keep], pts_b[keep]
    result.flow_px = float(np.median(np.linalg.norm(in_b - in_a, axis=1)))

    # A homography maps frame to frame exactly when the camera only rotates
    # about its optical centre, and also when the whole scene lies on one
    # plane. Fit it to the geometrically consistent correspondences only:
    # outliers would inflate the residual and fake parallax that is not there.
    homography, mask_h = cv2.findHomography(in_a, in_b, cv2.RANSAC,
                                            0.008 * diag)

    if homography is None:
        return result

    result.h_share = float(mask_h.ravel().astype(bool).mean())

    warped = cv2.perspectiveTransform(in_a.reshape(-1, 1, 2),
                                      homography).reshape(-1, 2)
    back = cv2.perspectiveTransform(in_b.reshape(-1, 1, 2),
                                    np.linalg.inv(homography)).reshape(-1, 2)

    # Symmetric transfer error. What a single homography CANNOT explain is
    # exactly the depth-dependent part of the motion, which is parallax.
    forward = np.linalg.norm(warped - in_b, axis=1)
    reverse = np.linalg.norm(back - in_a, axis=1)
    result.parallax_px = float(np.median(0.5 * (forward + reverse)))

    # A homography models a rotation-only pair exactly. It cannot model
    # translation unless the scene is one plane. So the honest question is
    # not "how big is the residual" but "does an essential matrix explain
    # more correspondences than a homography can".
    #
    # This replaces a residual-only test that called a real walk down a
    # corridor rotation-only. Forward motion moves every feature along a ray
    # from the focus of expansion, and in a corridor -- walls at similar
    # depth, a distant vanishing point -- a homography absorbs almost all of
    # that. The residual stays small while the translation is perfectly real.
    result.eh_ratio, result.forward = _epipolar(in_a, in_b, diag)

    result.verdict = classify(result, diag, thr)

    return result


def _epipolar(in_a, in_b, diag):
    """Return (essential/homography inlier ratio, forwardness of translation).

    Intrinsics are unknown, so assume a 60 degree horizontal field of view.
    The ratio is a comparison between two fits of the same points, so a
    wrong focal length shifts both sides and largely cancels. Forwardness is
    more sensitive and is reported as a hint, not a gate.
    """
    if len(in_a) < 8:
        return 0.0, 0.0

    span = float(diag) / math.sqrt(2.0)
    focal = 0.9 * span
    centre = (float(in_a[:, 0].mean()), float(in_a[:, 1].mean()))

    K = np.array([[focal, 0.0, centre[0]],
                  [0.0, focal, centre[1]],
                  [0.0, 0.0, 1.0]])

    essential, mask_e = cv2.findEssentialMat(in_a, in_b, K, cv2.RANSAC,
                                             0.999, 0.004 * diag)

    homography, mask_h = cv2.findHomography(in_a, in_b, cv2.RANSAC,
                                            0.004 * diag)

    if essential is None or mask_e is None or mask_h is None:
        return 0.0, 0.0

    if essential.shape != (3, 3):          # degenerate: several stacked
        essential = essential[:3, :3]

    e_in = float(mask_e.sum())
    h_in = max(float(mask_h.sum()), 1.0)

    forward = 0.0

    try:
        _, _, t, _ = cv2.recoverPose(essential, in_a, in_b, K,
                                     mask=mask_e.copy())
        t = t.ravel()
        norm = float(np.linalg.norm(t))

        if norm > 1e-9:
            forward = abs(float(t[2])) / norm
    except cv2.error:
        pass

    return e_in / h_in, forward


def classify(pair, diag, thr):
    """TRANSLATE and DOLLY both reconstruct. ROTATE never does.

    DOLLY is forward motion: real translation, but the baseline points
    along the view direction, which is the worst geometry for triangulating
    what is straight ahead. It reconstructs, badly. Saying so is more useful
    than lumping it in with either neighbour.
    """
    if pair.flow_px < thr.flow_min_diag * diag:
        return "STATIC"

    translating = (pair.eh_ratio >= thr.eh_translate
                   or pair.parallax_px >= thr.parallax_diag * diag)

    if translating:
        return "DOLLY" if pair.forward >= thr.forward_dolly else "TRANSLATE"

    if pair.eh_ratio <= thr.eh_rotate and \
            pair.parallax_px <= thr.rotate_diag * diag:
        return "ROTATE"

    return "WEAK"


# ---------------------------------------------------------------
# DEGRADATION
# ---------------------------------------------------------------
# The team can supply a phone video today and robot footage only later. A
# phone video answers "does the method work". It does not answer "does it
# work on OUR cameras", which is the question that decides the demo. So push
# the phone frames through the ESP32-CAM's actual losses and measure again.
# The DELTA between the two runs is the finding: it is how much margin the
# robot camera costs, and therefore whether a robot-camera test is worth
# running at all.


class Degrader:
    """Approximate an ESP32-CAM feed by damaging real frames, not by modelling.

    Order follows the physical path: the chassis shakes the sensor, then the
    chip downsamples, then it JPEGs. Applying the shake after downscaling
    would blur artefacts instead of image content and understate the loss.
    """

    def __init__(self, size=ESP32_SIZE, quality=ESP32_JPEG_QUALITY,
                 shake_px=0.0, seed=0):
        self.size = size
        self.quality = quality
        self.shake_px = shake_px
        self.rng = np.random.default_rng(seed)

    def __call__(self, frame):
        height, width = frame.shape[:2]

        if self.shake_px > 0:
            # shake_px is given at QVGA scale so the number means the same
            # thing whatever the source resolution is.
            length = max(2, int(round(self.shake_px * width / self.size[0])))
            frame = cv2.filter2D(frame, -1, self._streak(length))

        # INTER_AREA is the only correct downscale: it averages the pixels
        # being thrown away instead of point-sampling them, which is what a
        # real sensor binning down to QVGA does.
        frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)

        # Encode and decode for real. A synthetic "compression artefact" would
        # be a guess; this is the same libjpeg path the camera uses.
        ok, buffer = cv2.imencode(".jpg", frame,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])

        if not ok:
            return frame

        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def _streak(self, length):
        """Linear motion blur along a random direction, one draw per frame.

        A vibrating toy car smears in a direction that changes frame to frame.
        A fixed direction would let ORB learn the anisotropy and overstate how
        well the degraded video tracks.
        """
        angle = self.rng.uniform(0, np.pi)
        kernel = np.zeros((length, length), np.float32)
        centre = (length - 1) / 2.0

        for step in range(length):
            offset = step - centre
            row = int(round(centre + offset * np.sin(angle)))
            col = int(round(centre + offset * np.cos(angle)))
            kernel[row, col] = 1.0

        return kernel / kernel.sum()


def describe(degrader):
    parts = [f"{degrader.size[0]}x{degrader.size[1]}",
             f"JPEG q{degrader.quality}"]

    if degrader.shake_px > 0:
        parts.append(f"{degrader.shake_px:.1f} px motion blur")

    return ", ".join(parts)


# ---------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------

def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else 0.0


def analyse(path, stride_s, max_frames, nfeatures, thr, transform=None):
    frames, fps, stride, total = sample_frames(path, stride_s, max_frames)

    if len(frames) < 2:
        raise SystemExit(f"only {len(frames)} frame(s) read from {path}")

    if transform is not None:
        frames = [(index, transform(frame)) for index, frame in frames]

    height, width = frames[0][1].shape[:2]
    diag = float(np.hypot(width, height))
    detector = make_detector(nfeatures)

    per_frame, features = [], []

    for index, frame in frames:
        metrics, keypoints, descriptors = frame_metrics(index, frame, fps,
                                                        detector)
        per_frame.append(metrics)
        features.append((index, keypoints, descriptors))

    per_pair = []

    for (ia, ka, da), (ib, kb, db) in zip(features, features[1:]):
        per_pair.append(pair_metrics(ia, ka, da, ib, kb, db, diag, thr))

    return dict(path=path, width=width, height=height, diag=diag, fps=fps,
                stride=stride, stride_s=stride / fps, total_frames=total,
                duration_s=total / fps, nfeatures=nfeatures,
                frames=per_frame, pairs=per_pair)


def summarise(report, thr):
    frames, pairs = report["frames"], report["pairs"]
    diag = report["diag"]

    blur_ref = [f.blur_ref for f in frames]
    kp_ref = [f.keypoints_ref for f in frames]
    live = [p for p in pairs if p.verdict != "FAILED"]

    inliers = [p.inliers for p in pairs]
    ratios = [p.inlier_ratio for p in pairs]
    counts = {name: sum(1 for p in pairs if p.verdict == name)
              for name in ("TRANSLATE", "DOLLY", "ROTATE", "STATIC", "WEAK",
                           "FAILED")}

    moving = [p for p in live if p.verdict != "STATIC"]

    flow_median = pct([p.flow_px for p in live], 50)
    parallax_median = pct([p.parallax_px for p in moving], 50)

    # ORB returns at most nfeatures, so a count at the cap is censored: the
    # frame has at least that much texture and possibly far more. Differencing
    # a censored number against an uncensored one invents a loss.
    cap = report["nfeatures"]

    return dict(
        blur_median=pct([f.blur for f in frames], 50),
        blur_ref_median=pct(blur_ref, 50), blur_ref_p10=pct(blur_ref, 10),
        blur_bad_frac=float(np.mean([b < thr.blur_min for b in blur_ref])),
        kp_median=pct([f.keypoints for f in frames], 50),
        kp_ref_median=pct(kp_ref, 50), kp_ref_p10=pct(kp_ref, 10),
        kp_bad_frac=float(np.mean([k < thr.kp_min for k in kp_ref])),
        kp_capped_frac=float(np.mean([f.keypoints >= cap for f in frames])),
        inliers_median=pct(inliers, 50), inliers_p10=pct(inliers, 10),
        ratio_median=pct(ratios, 50),
        flow_median=flow_median, parallax_median=parallax_median,
        # Per mille of the diagonal: the same camera motion measured on two
        # resolutions gives the same number here and different pixel counts.
        flow_permille=1000.0 * flow_median / diag,
        parallax_permille=1000.0 * parallax_median / diag,
        h_share_median=pct([p.h_share for p in moving], 50),
        counts=counts,
        # DOLLY is forward translation. The geometry is poor but the
        # baseline is real, so it counts towards reconstructing.
        translate_frac=(counts["TRANSLATE"] + counts["DOLLY"]) / float(len(pairs)),
        dolly_frac=counts["DOLLY"] / float(len(pairs)),
        forward_median=pct([p.forward for p in moving if p.forward > 0], 50),
        eh_median=pct([p.eh_ratio for p in moving if p.eh_ratio > 0], 50),
        pair_count=len(pairs), diag=diag)


def verdict(summary, thr):
    """Return (label, reasons). Hard reasons veto; soft reasons only warn."""
    hard, soft = [], []

    counts = summary["counts"]

    if summary["translate_frac"] < thr.translate_frac_hard:
        hard.append(
            f"the camera does not translate: only "
            f"{counts['TRANSLATE'] + counts['DOLLY']} of "
            f"{summary['pair_count']} frame pairs show parallax "
            f"({counts['ROTATE']} rotation-only, {counts['STATIC']} static). "
            f"A camera that pans reconstructs nothing, at any resolution.")
    elif summary["translate_frac"] < thr.translate_frac_soft:
        soft.append(
            f"only {summary['translate_frac']*100:.0f}% of pairs translate; "
            f"the rest add frames without adding geometry.")

    # Forward motion is real translation with the worst possible baseline:
    # it points where the camera is looking, so whatever is straight ahead
    # barely moves between frames and triangulates badly. Worth saying out
    # loud, because the fix is free -- walk sideways instead.
    if counts["DOLLY"] > counts["TRANSLATE"] and summary["translate_frac"] > 0:
        soft.append(
            f"the camera mostly moves forward ({counts['DOLLY']} pairs "
            f"against {counts['TRANSLATE']} sideways). Forward motion "
            f"reconstructs, but the baseline points along the view "
            f"direction, so the view ahead is triangulated poorly. Strafing "
            f"past the scene gives far more depth for the same walking.")

    if summary["inliers_median"] < thr.track_hard:
        hard.append(
            f"frames do not track: median {summary['inliers_median']:.0f} "
            f"RANSAC inliers per pair, below {thr.track_hard}. Nothing "
            f"downstream can recover correspondence the frames do not have.")
    elif summary["inliers_median"] < thr.track_soft:
        soft.append(
            f"weak tracking: median {summary['inliers_median']:.0f} inliers "
            f"per pair. Expect a sparse and holey result.")

    if summary["ratio_median"] < thr.ratio_soft:
        soft.append(
            f"only {summary['ratio_median']*100:.0f}% of raw matches survive "
            f"RANSAC; repeated texture or motion blur is producing wrong "
            f"correspondences.")

    if summary["blur_bad_frac"] > thr.blur_bad_frac_soft:
        soft.append(
            f"{summary['blur_bad_frac']*100:.0f}% of sampled frames are below "
            f"the blur floor ({thr.blur_min:.0f} at {REF_DIAG:.0f} px "
            f"diagonal). Drive slower, or sample only the sharpest frames.")

    if summary["kp_bad_frac"] > 0.5:
        soft.append(
            f"{summary['kp_bad_frac']*100:.0f}% of frames carry under "
            f"{thr.kp_min} keypoints at {REF_DIAG:.0f} px diagonal. The arena "
            f"needs more surface texture.")

    if summary["h_share_median"] > 0.95 and summary["translate_frac"] > 0:
        soft.append(
            "one homography explains almost every inlier. That is a flat "
            "scene or a near-flat one; depth will be close to a single plane.")

    if hard:
        return "DO NOT ATTEMPT", hard, soft

    if soft:
        return "MARGINAL", hard, soft

    return "WORTH ATTEMPTING", hard, soft


def print_report(report, summary, label, hard, soft, thr, show_frames):
    print()
    print("=" * 72)
    print(f"  FEASIBILITY  {report['path']}")
    print("=" * 72)
    print(f"  {report['width']}x{report['height']}  "
          f"{report['fps']:.1f} fps  {report['duration_s']:.1f} s  "
          f"{report['total_frames']} frames")
    print(f"  sampled every {report['stride']} frames "
          f"({report['stride_s']:.2f} s apart), "
          f"{len(report['frames'])} frames, {len(report['pairs'])} pairs")

    if show_frames:
        print()
        print("  blur and kpts below are measured at "
              f"{REF_DIAG:.0f} px diagonal, not at native size.")
        print("  frame      t/s     blur    kpts   matches  inliers  ratio"
              "   flow  parallax  motion")
        print("  " + "-" * 82)

        pairs = {p.index_a: p for p in report["pairs"]}

        for frame in report["frames"]:
            pair = pairs.get(frame.index)
            tail = (f"{pair.matches:9d}{pair.inliers:9d}{pair.inlier_ratio:7.2f}"
                    f"{pair.flow_px:7.1f}{pair.parallax_px:10.2f}  {pair.verdict}"
                    if pair else "")
            print(f"  {frame.index:5d}{frame.time_s:9.2f}{frame.blur_ref:9.1f}"
                  f"{frame.keypoints_ref:8d}{tail}")

    print()
    print(f"  AGGREGATE  (blur and texture at {REF_DIAG:.0f} px diagonal)")
    print(f"    blur            median {summary['blur_ref_median']:8.1f}   "
          f"p10 {summary['blur_ref_p10']:8.1f}   "
          f"{summary['blur_bad_frac']*100:5.1f}% under {thr.blur_min:.0f}")
    print(f"    texture (ORB)   median {summary['kp_ref_median']:8.0f}   "
          f"p10 {summary['kp_ref_p10']:8.0f}   "
          f"{summary['kp_bad_frac']*100:5.1f}% under {thr.kp_min}")
    print(f"    trackability    median {summary['inliers_median']:8.0f}   "
          f"p10 {summary['inliers_p10']:8.0f}   inliers, "
          f"{summary['ratio_median']*100:.0f}% of matches survive RANSAC")
    print(f"    motion          median {summary['flow_median']:8.2f} px flow "
          f"({summary['flow_permille']:.1f} per mille of the diagonal), "
          f"{summary['parallax_median']:.2f} px unexplained by rotation "
          f"(threshold {thr.parallax_diag*summary['diag']:.2f})")

    counts = summary["counts"]
    print(f"    parallax        {counts['TRANSLATE']} sideways, "
          f"{counts['DOLLY']} forward, {counts['ROTATE']} rotate-only, "
          f"{counts['STATIC']} static, {counts['WEAK']} weak, "
          f"{counts['FAILED']} unmeasurable")
    print(f"    epipolar        median E/H {summary['eh_median']:.2f} "
          f"(translation above {thr.eh_translate:.2f}), "
          f"forwardness {summary['forward_median']:.2f}")

    capped = ("  CAPPED by --features" if summary["kp_capped_frac"] > 0.1
              else "")
    print(f"    at native size  blur {summary['blur_median']:.1f}, "
          f"{summary['kp_median']:.0f} ORB keypoints{capped}")
    print(f"                    diagnostic only: both rise as resolution "
          f"falls, so they do not compare across sizes")

    print()
    print(f"  VERDICT: {label}")

    for reason in hard:
        print(f"    [STOP] {reason}")

    for reason in soft:
        print(f"    [warn] {reason}")

    if not hard and not soft:
        print("    [ok]   sharp, textured, tracks cleanly, and the camera "
              "translates.")

    print()
    print("  Thresholds are priors calibrated on synthetic video only.")
    print("  Read the numbers before you trust the verdict.")
    print()


# Flow is a property of the camera path, not of image quality, so in diagonal
# units it should barely move between the two runs. If it does, the
# degradation broke matching rather than the scene, and every other delta
# below is suspect. In PIXELS it moves by the scale factor alone, which is why
# it is differenced as per mille of the diagonal instead.
ROWS = [
    (f"blur @ {REF_DIAG:.0f}px diag", "blur_ref_median", "{:.1f}"),
    (f"texture @ {REF_DIAG:.0f}px diag", "kp_ref_median", "{:.0f}"),
    ("trackability (inliers)", "inliers_median", "{:.0f}"),
    ("RANSAC survival", "ratio_median", "{:.2f}"),
    ("flow (per mille diag)", "flow_permille", "{:.2f}"),
    ("parallax (per mille diag)", "parallax_permille", "{:.2f}"),
    ("pairs translating", "translate_frac", "{:.2f}"),
]

# Same numbers the two per-run reports print, repeated here only so nobody
# goes looking for them and differences them by hand.
NATIVE_ROWS = [
    ("blur, native size", "blur_median", "{:.1f}"),
    ("texture, native size", "kp_median", "{:.0f}"),
]


def print_comparison(source, degraded, note, source_size, degraded_size):
    print()
    print("=" * 72)
    print(f"  SOURCE  vs  DEGRADED ({note})")
    print("=" * 72)
    print(f"  {'metric':<30}{'source':>11}{'degraded':>11}"
          f"{'delta':>11}{'change':>10}")
    print("  " + "-" * 73)

    # A metric that IMPROVES under damage is measuring an artefact, not the
    # scene. Mark those rows in place: a reader who skims the table and never
    # reaches the notes must still not read them as a gain.
    suspect = {key for key in ("kp_ref_median", "parallax_permille")
               if degraded[key] > source[key]}

    for label, key, fmt in ROWS:
        before, after = source[key], degraded[key]
        delta = after - before
        change = f"{delta / before * 100.0:>9.0f}%" if before else f"{'--':>10}"
        mark = " (*)" if key in suspect else ""

        print(f"  {label + mark:<30}{fmt.format(before):>11}"
              f"{fmt.format(after):>11}{delta:>+11.2f}{change}")

    print("  " + "-" * 73)

    for label, key, fmt in NATIVE_ROWS:
        print(f"  {label:<30}{fmt.format(source[key]):>11}"
              f"{fmt.format(degraded[key]):>11}{'--':>11}{'--':>10}")

    print(f"  {source_size} -> {degraded_size}: these two columns are measured "
          f"on different grids.")
    print( "  Both metrics rise as pixels are thrown away, so their difference "
          "is not a quality")
    print( "  change and is not shown. Use the two rows at the top of the "
          "table instead.")

    print()
    print(f"  frames under blur floor   "
          f"{source['blur_bad_frac']*100:5.1f}%  ->"
          f"{degraded['blur_bad_frac']*100:6.1f}%")
    print(f"  frames under kpt floor    "
          f"{source['kp_bad_frac']*100:5.1f}%  ->"
          f"{degraded['kp_bad_frac']*100:6.1f}%")

    # Damage cannot add scene content, so any rise is the metric failing, and
    # each one fails for its own reason. Name the reason: an unexplained "+"
    # in this table is how a team talks itself into trusting a worse camera.
    if "kp_ref_median" in suspect:
        print()
        print("  (*)   keypoint count ROSE while sharpness FELL, on one grid. "
              "JPEG blocking")
        print("        manufactures corners out of block edges. They are worse "
              "than useless:")
        print("        they match badly and pull RANSAC survival down. The "
              "keypoint floor line")
        print("        above inherits the same lie. Read trackability, not "
              "keypoint count.")

    if "parallax_permille" in suspect:
        print()
        print("  (*)   parallax ROSE. Degrading a video cannot move the "
              "camera. Noisier corner")
        print("        localisation inflates the residual a homography cannot "
              "explain, which is")
        print("        what this metric measures. It is false depth, and the "
              "TRANSLATE count")
        print("        above inherits it.")

    print()


def combined_verdict(source_label, degraded_label, source_summary,
                     degraded_summary):
    """State both answers. They are different questions with different fixes."""
    print("  COMBINED VERDICT")

    if source_label == "DO NOT ATTEMPT":
        print(f"    source: {source_label}. Fix the capture before you look "
              f"at the robot camera at all -- the degraded run cannot be "
              f"better than its input, so it tells you nothing yet.")
        return 2

    retained = (degraded_summary["inliers_median"]
                / max(source_summary["inliers_median"], 1.0))

    if degraded_label == "DO NOT ATTEMPT":
        print(f"    source: {source_label}. Degraded: {degraded_label}.")
        print(f"    The method works on this footage and the CAMERA is the "
              f"limit. Only {retained*100:.0f}% of tracking survives the trip "
              f"to QVGA. Reconstruct from the phone video and treat the robot "
              f"feed as a stretch goal, or improve the feed (light the arena, "
              f"raise JPEG quality, damp the chassis) before testing it.")
        return 2

    if degraded_label == "MARGINAL" or source_label == "MARGINAL":
        print(f"    source: {source_label}. Degraded: {degraded_label}.")
        print(f"    Viable on the source, thin at QVGA "
              f"({retained*100:.0f}% of tracking retained). A robot-camera "
              f"test is worth running, and expect a sparser cloud.")
        return 0

    print(f"    source: {source_label}. Degraded: {degraded_label}.")
    print(f"    Viable on both. {retained*100:.0f}% of tracking survives "
          f"QVGA and JPEG q12, so the robot camera is not the bottleneck. "
          f"Run the robot-camera test.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--stride-s", type=float, default=0.25,
                        help="seconds between sampled frames. Longer means "
                             "more parallax per pair and harder matching "
                             "(default 0.25)")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--features", type=int, default=2000)
    parser.add_argument("--blur-min", type=float, default=Thresholds.blur_min)
    parser.add_argument("--kp-min", type=int, default=Thresholds.kp_min)
    parser.add_argument("--no-per-frame", action="store_true")
    parser.add_argument("--degrade", action="store_true",
                        help="also run the whole suite on the same video "
                             "degraded to an ESP32-CAM feed "
                             f"({ESP32_SIZE[0]}x{ESP32_SIZE[1]}, real JPEG "
                             f"round trip at quality {ESP32_JPEG_QUALITY}), "
                             "and print the delta on every metric")
    parser.add_argument("--shake-px", type=float, default=0.0,
                        help="motion blur length in QVGA pixels added by "
                             "--degrade, approximating a vibrating chassis. "
                             "Separate flag so its effect stays isolable "
                             "(default 0, meaning off)")
    parser.add_argument("--jpeg-quality", type=int, default=ESP32_JPEG_QUALITY)
    parser.add_argument("--json", metavar="PATH",
                        help="also write the full report as JSON")

    args = parser.parse_args(argv)

    thr = Thresholds(blur_min=args.blur_min, kp_min=args.kp_min)

    report = analyse(args.video, args.stride_s, args.max_frames,
                     args.features, thr)
    summary = summarise(report, thr)
    label, hard, soft = verdict(summary, thr)

    print_report(report, summary, label, hard, soft, thr,
                 not args.no_per_frame)

    status = {"WORTH ATTEMPTING": 0, "MARGINAL": 0, "DO NOT ATTEMPT": 2}[label]

    degraded = degraded_summary = None

    if args.degrade:
        degrader = Degrader(quality=args.jpeg_quality, shake_px=args.shake_px)

        degraded = analyse(args.video, args.stride_s, args.max_frames,
                           args.features, thr, transform=degrader)
        degraded_summary = summarise(degraded, thr)
        degraded_label, d_hard, d_soft = verdict(degraded_summary, thr)

        print_report(degraded, degraded_summary, degraded_label, d_hard,
                     d_soft, thr, not args.no_per_frame)
        print_comparison(summary, degraded_summary, describe(degrader),
                         f"{report['width']}x{report['height']}",
                         f"{degraded['width']}x{degraded['height']}")

        status = combined_verdict(label, degraded_label, summary,
                                  degraded_summary)
        print()

    if args.json:
        payload = dict(report)
        payload["frames"] = [asdict(f) for f in report["frames"]]
        payload["pairs"] = [asdict(p) for p in report["pairs"]]
        payload["summary"] = summary
        payload["verdict"] = label
        payload["stop"] = hard
        payload["warn"] = soft

        if degraded is not None:
            payload["degraded"] = dict(
                frames=[asdict(f) for f in degraded["frames"]],
                pairs=[asdict(p) for p in degraded["pairs"]],
                summary=degraded_summary,
                verdict=degraded_label)

        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)

        print(f"  wrote {args.json}\n")

    # Exit code lets a script gate on this without parsing the report.
    return status


if __name__ == "__main__":
    sys.exit(main())
