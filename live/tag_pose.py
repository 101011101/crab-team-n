"""Pose from AprilTags. Metres, radians, the room frame — and it is MEASURED.

This replaces SfM registration as the primary pose source, and it is a large
upgrade for three separate reasons:

  SCALE      A tag has a known printed edge (TAG_SIZE, 0.15 m). Four corners at a
             known metre spacing fix the scale outright. Monocular SfM cannot
             recover a metre at all; every number it produces is a model unit.
  ROLL       A straight drive with a thin point cloud cannot observe roll about the
             direction of travel -- docs/robot1-reconstruction.md measured 15.8 deg
             of unobservable rotation left over even after a shared calibration. The
             tag plane gives two axes and its in-plane orientation gives the third.
  FEATURES   Registering a 320x240 robot frame against a model built from 1080p
             phone frames needs hundreds of SIFT matches it does not have. A tag
             needs four corners.

docs/robot1-reconstruction.md put it plainly: one tag of known size collapses the
whole 7-DoF search to a lookup. This is that lookup.

All of the hard geometry already exists in pipeline/: the detector, the camera
intrinsics, the mount transform, the tag map and the tag->robot pose solve. This
module is a thin live wrapper over it, not a reimplementation.
"""
import math, os, sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.common import constants as C          # noqa: E402
from pipeline.common import geometry as Gm          # noqa: E402
from pipeline.localize.record import TagDetector    # noqa: E402

MAX_REPROJ_PX = 4.0     # a tag solved worse than this is a misdetection, not a fix
MAX_RANGE_M = 3.5       # past this a 15 cm tag is too few pixels to trust


class TagPose:
    """One detector, shared by every robot. Detection is stateless."""

    def __init__(self, max_reproj=MAX_REPROJ_PX, max_range=MAX_RANGE_M):
        self.det = TagDetector()
        self.max_reproj, self.max_range = max_reproj, max_range
        self.n_frames = self.n_fixes = 0

    def fix(self, frame):
        """Room-frame pose from one camera frame, or None.

        Every MAPPED tag in view independently implies a full robot pose. They are
        combined by weight rather than picking one: a tag solved at 0.3 px and 0.8 m
        deserves more say than one at 3 px and 3 m. Heading is averaged on the
        circle, because averaging +179 and -179 degrees arithmetically gives 0.
        """
        self.n_frames += 1
        dets = self.det.detect(frame)
        est, w = [], []
        for d in dets:
            if "robot_pose_est" not in d:
                continue                      # an unmapped tag fixes nothing
            if d["reproj_px"] > self.max_reproj or d["range_m"] > self.max_range:
                continue
            p = d["robot_pose_est"]
            est.append((p["x"], p["y"], math.radians(p["theta_deg"])))
            w.append(1.0 / (d["reproj_px"] + 0.25) / (d["range_m"] + 0.5))
        if not est:
            return None
        self.n_fixes += 1
        w = np.asarray(w)
        w /= w.sum()
        xs = np.asarray([e[0] for e in est])
        ys = np.asarray([e[1] for e in est])
        th = np.asarray([e[2] for e in est])
        return dict(
            x=float(xs @ w), y=float(ys @ w),
            theta=float(math.atan2(np.sin(th) @ w, np.cos(th) @ w)),
            n_tags=len(est),
            tags=[d["id"] for d in dets if "robot_pose_est" in d],
            spread_m=float(np.hypot(xs.std(), ys.std())) if len(est) > 1 else 0.0)

    def box(self, frame, robot):
        """Where the pushable box is, from its own tag. MEASURED, not predicted.

        docs/push-planning.md is explicit that where a shoved box ends up depends on
        the friction across its contact patch -- invisible, uneven, changing as it
        slides -- so open-loop pushing drifts and no model fixes it. The answer is
        not a better model. It is a marker on the box, read at camera rate.
        """
        for d in self.det.detect(frame):
            if d["id"] != C.BOX_TAG_ID or d["reproj_px"] > self.max_reproj:
                continue
            # tag centre in the robot frame, then into the room frame
            t = np.asarray(d["pose_cam"]["t"], float)
            Rrc, trc = Gm.robot_T_cam()
            p_r = Rrc @ t + trc
            ct, st = math.cos(robot[2]), math.sin(robot[2])
            return dict(x=robot[0] + ct * p_r[0] - st * p_r[1],
                        y=robot[1] + st * p_r[0] + ct * p_r[1],
                        size=C.BOX_SIZE, range_m=d["range_m"],
                        reproj_px=d["reproj_px"])
        return None

    def stats(self):
        return dict(frames=self.n_frames, fixes=self.n_fixes,
                    rate=round(self.n_fixes / max(self.n_frames, 1), 3))


def _selftest():
    """Round trip: render a tag at a known pose, detect it, recover the pose.

    This proves the intrinsics, the object points and the PnP flags agree with each
    other. It does not prove the real camera matches CAM_HFOV_DEG -- constants.py
    marks that as assumed and still to be measured with a chessboard.
    """
    import cv2
    K, D = Gm.camera_matrix(), Gm.dist_coeffs()
    obj = Gm.tag_object_points()
    tag_id = sorted(C.TAG_MAP)[0]
    img = np.full((C.CAM_H, C.CAM_W), 255, np.uint8)

    # Put the tag 1.2 m in front of the camera, 0.1 m to the right, FACING it.
    # rvec = 0 leaves the tag normal pointing along +z, i.e. away from the camera:
    # you render its back, the pattern comes out mirrored, and nothing detects.
    tvec = np.array([0.10, 0.0, 1.20])
    rvec = np.array([0.0, math.pi, 0.0])
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    dst = proj.reshape(4, 2).astype(np.float32)
    # A tag with no white margin is undetectable: the detector finds the black
    # square by its contrast against a light quiet zone. Pad it.
    M, PAD = 400, 100
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tag_id, M)
    canvas = np.full((M + 2 * PAD, M + 2 * PAD), 255, np.uint8)
    canvas[PAD:PAD + M, PAD:PAD + M] = marker
    marker = canvas
    src = np.array([[PAD, PAD], [PAD + M - 1, PAD],
                    [PAD + M - 1, PAD + M - 1], [PAD, PAD + M - 1]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(marker, H, (C.CAM_W, C.CAM_H),
                               borderValue=255, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
    img = np.minimum(img, warp)

    tp = TagPose()
    dets = tp.det.detect(img)
    assert dets, "no tag detected in the synthetic view"
    d = dets[0]
    print(f"  tag {d['id']}  reproj {d['reproj_px']} px  "
          f"range {d['range_m']} m (truth {np.linalg.norm(tvec):.3f} incl. mount)")
    assert d["id"] == tag_id, f"wrong id: {d['id']} != {tag_id}"
    assert d["reproj_px"] < 2.0, f"reprojection {d['reproj_px']} px is too high"
    fix = tp.fix(img)
    assert fix is not None, "mapped tag produced no room-frame fix"
    print(f"  room-frame fix: x={fix['x']:.3f} y={fix['y']:.3f} "
          f"theta={math.degrees(fix['theta']):.1f} deg from {fix['n_tags']} tag(s)")
    print("  tag_pose selftest PASSED")


if __name__ == "__main__":
    _selftest()
