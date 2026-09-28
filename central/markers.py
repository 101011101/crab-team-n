"""Marker pose: absolute robot pose in the arena frame, from one frame.

The robots have no encoders and no IMU, so there is nothing to dead-reckon
from. Every pose in this project comes from looking at a printed marker
whose position somebody measured with a tape. That makes pose STATELESS:
each frame yields an absolute pose on its own, there is no drift to
accumulate, and a stale or wrong map cannot produce a wrong pose. Rung 1
exists to make that property real.


FRAMES
------

Four frames, and the whole module is the chain between them.

    W  world / arena    x, y on the floor, z up. The frame every Pose is in.
    B  robot body       x forward, y left, z up. Origin on the floor under
                        the robot's centre of rotation.
    C  camera optical   x right, y down, z along the optical axis. OpenCV's
                        convention, and what solvePnP speaks.
    M  marker           x right along the tag, y up the tag, z out of the
                        printed face toward whoever is reading it. OpenCV's
                        convention for square-marker object points.

solvePnP hands back the pose of the MARKER as seen from the CAMERA. We want
the pose of the ROBOT in the WORLD. That is one inversion and two chained
transforms, and getting any of them backwards produces a robot that drives
confidently in the wrong direction rather than an exception. Written out:

    T_cm  = solvePnP(marker corners in M, image corners)     marker -> camera
    T_mc  = inverse(T_cm)                                    camera -> marker
    T_wm  = from the marker map, measured by hand            marker -> world
    T_wc  = T_wm @ T_mc                                      camera -> world
    T_wb  = T_wc @ inverse(T_bc)                             body   -> world

T_bc is the camera mount: where the camera sits on the chassis and where it
is pointing. See CameraMount.


WHY FLATTENING 6-DOF TO (x, y, theta) IS SAFE HERE
--------------------------------------------------

T_wb is a full 6-DoF rigid transform. Pose is three numbers. We take

    x, y   = the first two components of the body origin in world
    theta  = atan2(R_wb[1, 0], R_wb[0, 0])

theta is the ZYX-Euler yaw: the angle of the body's forward axis after it
is projected onto the floor plane.

This is safe because of a physical fact, not a mathematical one. The arena
floor is flat and the robot is a tracked chassis that cannot leave it, so
the true body roll and pitch are zero and R_wb is a pure rotation about z.
For a pure yaw, that atan2 is not an approximation -- it is exact. Non-zero
roll or pitch in the solution is therefore measurement error, not motion,
and projecting it away is the correct thing to do with it: it is the one
component of the estimate we know the answer to in advance.

Two consequences worth keeping:

  - The discarded components are free diagnostics, not waste. The solved
    camera height must match the measured mount height and the solved body
    tilt must be near zero. Both are reported on PoseEstimate. A pose that
    says the camera is 2 m up is wrong even when its reprojection error
    looks fine, and it is usually a mis-measured marker map entry.
  - Tilt and height VETO solutions; they do not choose between them. A small
    square tag has two solutions that reproject almost equally well and
    differ by a heading flip, and the tempting move is to keep whichever one
    leaves the robot level. Measured, that is a coin flip: on a 20 px tag at
    2 m with half a pixel of corner noise, the solved lens height of a
    CORRECT solve is already 26 cm out at the 95th percentile, so it cannot
    referee a difference it is larger than. Physics is allowed to say "not
    that one". Only the corners may say "this one". See _pick_candidate.


WHEN THERE IS NO ANSWER
-----------------------

The robot drives on this pose, so a confidently wrong pose is much worse
than no pose: None leaves the caller holding its last fix and its age, while
a wrong pose is acted on at once. estimate_pose therefore returns None when

  - no branch is physically possible (chassis on its side, lens underground),
  - two branches fit the corners comparably and point the robot to different
    places, which is the frame saying it cannot tell, or
  - nothing reprojects within MARKERS.max_reproj_px.

This is a real gate, not a formality. One small tag alone, at 2 m, with half
a pixel of corner noise, is refused about half the time -- and the refused
poses are the bad ones: worst refused 294 cm against worst kept 66 cm.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import MARKERS, MOUNTS
from geometry import Pose


# Corner order matters and is not negotiable: cv2.aruco.detectMarkers
# returns corners clockwise from the tag's own top-left as printed, and the
# object points below are written in that same order. Shuffle one and not
# the other and every pose comes back rotated by a multiple of 90 degrees.
_CORNER_ORDER = ("top-left", "top-right", "bottom-right", "bottom-left")

_R_BASE = np.array([[0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]])
"""Rotation of a vertical wall tag whose face points along world +x.

Columns are the marker axes in world coordinates: x_M = +y_W (right along
the wall), y_M = +z_W (up), z_M = +x_W (out of the face).
"""

_R_BODY_CAM = np.array([[0.0,  0.0, 1.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0]])
"""Rotation of a level camera bolted on facing straight ahead.

Columns are the camera axes in body coordinates: x_C (right) = -y_B,
y_C (down) = -z_B, z_C (optical axis) = +x_B.
"""


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _rt(R, t):
    """Pack a rotation and translation into a 4x4."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def _inv(T):
    """Inverse of a rigid transform. Never use np.linalg.inv here.

    The closed form is exact and cheap; a general inverse of a matrix built
    from a solver's output can drift out of SO(3) and silently scale poses.
    """
    R = T[:3, :3]
    return _rt(R.T, -R.T @ T[:3, 3])


# -----------------------------------------------------------
# MARKER MAP
# -----------------------------------------------------------

@dataclass(frozen=True)
class MarkerPlacement:
    """One tag, where it is and which way it faces, measured by hand.

    x, y, z locate the CENTRE of the printed square, not a corner, not the
    edge of the paper. z is height above the floor.

    yaw_deg is the compass direction the printed FACE points, in the world
    xy plane: 0 faces +x, 90 faces +y. A tag on the wall at the +y end of
    the arena looking back down the room is yaw_deg = -90.

    pitch_deg leans the tag's top out toward the viewer; roll_deg spins it
    in its own plane. Both default to zero, which is a tag hung square on a
    vertical wall. A tag lying face-up on the floor is pitch_deg = -90.
    """

    id: int
    x: float
    y: float
    z: float
    yaw_deg: float
    size_m: float
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    note: str = ""

    @property
    def rotation(self) -> np.ndarray:
        """R_wm. Columns are the marker axes expressed in the world frame."""
        return (_rz(math.radians(self.yaw_deg))
                @ _R_BASE
                @ _rx(math.radians(self.pitch_deg))
                @ _rz(math.radians(self.roll_deg)))

    @property
    def translation(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def transform(self) -> np.ndarray:
        """T_wm, 4x4."""
        return _rt(self.rotation, self.translation)

    def object_points(self) -> np.ndarray:
        """The four corners in the MARKER frame, in _CORNER_ORDER."""
        return object_points(self.size_m)

    def world_points(self) -> np.ndarray:
        """The four corners in the WORLD frame, in _CORNER_ORDER."""
        return (self.rotation @ self.object_points().T).T + self.translation


def object_points(size_m: float) -> np.ndarray:
    """Square-marker corners in the marker frame, OpenCV's ordering."""
    h = size_m / 2.0
    return np.array([[-h,  h, 0.0],
                     [ h,  h, 0.0],
                     [ h, -h, 0.0],
                     [-h, -h, 0.0]], dtype=np.float64)


@dataclass
class MarkerMap:
    """id -> MarkerPlacement, plus the dictionary the tags were printed from.

    Lives in a file so the team can measure the arena with a tape and edit
    numbers without touching code. See marker_map.example.json.
    """

    placements: dict = field(default_factory=dict)
    dictionary: str = MARKERS.dictionary
    frame: str = "arena"
    source: str = "<memory>"

    def __len__(self):
        return len(self.placements)

    def __contains__(self, marker_id):
        return int(marker_id) in self.placements

    def __getitem__(self, marker_id) -> MarkerPlacement:
        return self.placements[int(marker_id)]

    def get(self, marker_id):
        return self.placements.get(int(marker_id))

    @property
    def ids(self):
        return sorted(self.placements)

    def add(self, placement: MarkerPlacement) -> None:
        self.placements[int(placement.id)] = placement

    # -----------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict, source: str = "<dict>") -> "MarkerMap":
        default_size = float(data.get("default_size_m", MARKERS.default_size_m))
        out = cls(dictionary=data.get("dictionary", MARKERS.dictionary),
                  frame=data.get("frame", "arena"),
                  source=source)

        entries = data.get("markers", [])

        if isinstance(entries, dict):        # {"0": {...}} is a natural thing to write
            entries = [dict(v, id=int(k)) for k, v in entries.items()]

        for entry in entries:
            out.add(MarkerPlacement(
                id=int(entry["id"]),
                x=float(entry["x"]),
                y=float(entry["y"]),
                z=float(entry.get("z", 0.0)),
                yaw_deg=float(entry["yaw_deg"]),
                size_m=float(entry.get("size_m", default_size)),
                pitch_deg=float(entry.get("pitch_deg", 0.0)),
                roll_deg=float(entry.get("roll_deg", 0.0)),
                note=str(entry.get("note", "")),
            ))

        return out

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "dictionary": self.dictionary,
            "markers": [
                {"id": p.id, "x": p.x, "y": p.y, "z": p.z, "yaw_deg": p.yaw_deg,
                 "pitch_deg": p.pitch_deg, "roll_deg": p.roll_deg,
                 "size_m": p.size_m, "note": p.note}
                for p in (self.placements[i] for i in self.ids)
            ],
        }

    @classmethod
    def load(cls, path: str) -> "MarkerMap":
        """Read a .json, or a .yaml if PyYAML happens to be installed."""
        with open(path, "r") as handle:
            text = handle.read()

        if path.lower().endswith((".yaml", ".yml")):
            import yaml                       # optional; JSON is the supported path
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)

        return cls.from_dict(data, source=path)

    def save(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")


def load_marker_map(path: str = None) -> MarkerMap:
    """Load the configured map, or an empty one if the file is not there yet.

    Empty is a legitimate state before the arena is measured: pose then
    returns None everywhere, which is the same path as "no markers in view"
    and cannot be mistaken for a position.
    """
    path = path or MARKERS.map_path

    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    if not os.path.exists(path):
        return MarkerMap(source=f"{path} (missing)")

    return MarkerMap.load(path)


# -----------------------------------------------------------
# CAMERA MOUNT
# -----------------------------------------------------------

@dataclass(frozen=True)
class CameraMount:
    """Where the camera is on the chassis, in the robot body frame.

    x, y, z is the camera's optical centre relative to the body origin, with
    x forward, y left, z up. z is the lens height above the floor.

    pan/tilt are the servo's current aim, in RADIANS, measured from
    straight-ahead-and-level: pan positive turns the lens to the robot's
    left, tilt positive raises it. The firmware speaks degrees over the wire
    with a centre near 90; mapping those servo degrees onto these radians is
    a per-robot hardware constant that nobody has measured yet. Until they
    are measured, aim the camera once, leave it, and keep these at zero --
    the pose maths is identical, it just assumes the aim it is told.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.10
    pan_rad: float = 0.0
    tilt_rad: float = 0.0
    roll_rad: float = 0.0

    @property
    def rotation(self) -> np.ndarray:
        """R_bc. Columns are the camera axes expressed in the body frame."""
        return (_rz(self.pan_rad)
                @ _ry(-self.tilt_rad)
                @ _rx(self.roll_rad)
                @ _R_BODY_CAM)

    @property
    def translation(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def transform(self) -> np.ndarray:
        """T_bc, 4x4."""
        return _rt(self.rotation, self.translation)

    def aimed(self, pan_rad: float, tilt_rad: float) -> "CameraMount":
        """Same mount, different servo aim. Frozen, so this returns a copy."""
        return CameraMount(self.x, self.y, self.z, pan_rad, tilt_rad, self.roll_rad)


def mount_for(bot_id: int) -> CameraMount:
    """The configured mount for one robot, falling back to the fleet default."""
    cfg = MOUNTS.get(int(bot_id)) or MOUNTS["default"]
    return CameraMount(x=cfg.x, y=cfg.y, z=cfg.z, pan_rad=cfg.pan_rad,
                       tilt_rad=cfg.tilt_rad, roll_rad=cfg.roll_rad)


# -----------------------------------------------------------
# INTRINSICS
# -----------------------------------------------------------

def default_camera_matrix(width: int = 320, height: int = 240,
                          hfov_deg: float = None):
    """A guessed pinhole model for an uncalibrated camera.

    Assumes a centred principal point and square pixels, and derives the
    focal length from a nominal horizontal field of view. It is enough to
    get a robot moving on day one and it is NOT enough for the accuracy the
    rung-1 gate asks for: with no distortion model the corners of a 320x240
    ESP32-CAM frame are the most wrong part of the image, and marker corners
    live wherever they like. Calibrate. It is 15 minutes.
    """
    hfov_deg = MARKERS.default_hfov_deg if hfov_deg is None else hfov_deg
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)

    K = np.array([[f, 0.0, width / 2.0],
                  [0.0, f, height / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    return K, np.zeros((5, 1), dtype=np.float64)


def load_camera(bot_id: int, directory: str = None, width: int = 320,
                height: int = 240):
    """Return (camera_matrix, dist_coeffs, calibrated) for one robot.

    Intrinsics are per camera, not per camera model: two ESP32-CAMs off the
    same reel differ enough to matter at this resolution. `calibrated` is
    returned so callers can tell the operator they are running on a guess.
    """
    directory = directory or MARKERS.calib_dir

    if not os.path.isabs(directory):
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)

    path = os.path.join(directory, MARKERS.calib_name.format(bot_id=int(bot_id)))

    if not os.path.exists(path):
        K, dist = default_camera_matrix(width, height)
        return K, dist, False

    data = np.load(path)
    return (data["camera_matrix"].astype(np.float64),
            data["dist_coeffs"].astype(np.float64).reshape(-1, 1),
            True)


# -----------------------------------------------------------
# DETECTION
# -----------------------------------------------------------

@dataclass(frozen=True)
class MarkerDetection:
    id: int
    corners: np.ndarray          # (4, 2) float32 image pixels, in _CORNER_ORDER

    @property
    def center(self):
        cx, cy = self.corners.mean(axis=0)
        return float(cx), float(cy)

    @property
    def side_px(self) -> float:
        """Mean edge length. Under about 12 px the pose is not worth trusting."""
        c = self.corners
        return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


_detectors = {}


def get_detector(dictionary: str = None):
    """Cached ArucoDetector. Building one per frame costs more than detecting."""
    name = dictionary or MARKERS.dictionary

    if name not in _detectors:
        if not hasattr(cv2.aruco, name):
            raise ValueError(f"unknown ArUco dictionary {name!r}")

        params = cv2.aruco.DetectorParameters()

        # Sub-pixel refinement is the difference between a usable pose and a
        # jittery one at 320x240: a whole-pixel corner error on a tag only
        # 30 px across is over 3% of the tag, and pose error scales with it.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 3
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.minMarkerPerimeterRate = MARKERS.min_perimeter_rate

        _detectors[name] = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)), params)

    return _detectors[name]


def detect(frame, dictionary: str = None) -> list:
    """Find every marker in one frame. Returns [] for None or an empty frame."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return []

    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = get_detector(dictionary).detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return []

    return [MarkerDetection(int(i), np.asarray(c, dtype=np.float32).reshape(4, 2))
            for c, i in zip(corners, ids.ravel())]


# -----------------------------------------------------------
# POSE
# -----------------------------------------------------------

@dataclass(frozen=True)
class PoseEstimate:
    """A pose plus everything needed to decide whether to believe it."""

    pose: Pose
    marker_ids: tuple
    reprojection_error_px: float
    camera_height_m: float       # solved lens height; compare with the mount
    body_tilt_deg: float         # solved tilt off level; the floor is flat, so ~0
    nearest_marker_m: float
    rejected_ids: tuple = ()

    @property
    def n_markers(self) -> int:
        return len(self.marker_ids)


def _camera_from_marker(placement: MarkerPlacement, rvec, tvec) -> np.ndarray:
    """T_wc from one solvePnP result. The inversion the module warns about.

    solvePnP returns T_cm: it maps a point on the marker into the camera.
    The camera's own position in the world is the other direction, through
    the marker's measured place in the world.
    """
    R_cm, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return placement.transform @ _inv(_rt(R_cm, np.asarray(tvec).reshape(3)))


def _body_from_camera(T_wc: np.ndarray, mount: CameraMount) -> np.ndarray:
    return T_wc @ _inv(mount.transform)


def _tilt_deg(T_wb: np.ndarray) -> float:
    """Angle between the solved body up-axis and world up."""
    up = T_wb[:3, 2]
    return math.degrees(math.acos(float(np.clip(up[2] / np.linalg.norm(up), -1.0, 1.0))))


def _flatten(T_wb: np.ndarray) -> Pose:
    """6-DoF body transform -> (x, y, theta). See the module docstring."""
    return Pose(x=float(T_wb[0, 3]),
                y=float(T_wb[1, 3]),
                theta=float(math.atan2(T_wb[1, 0], T_wb[0, 0])))


def _refine(objects, image, K, dist, rvec, tvec):
    """Drive one solution to its own local minimum of reprojection error."""
    r = np.array(rvec, dtype=np.float64).reshape(3, 1)
    t = np.array(tvec, dtype=np.float64).reshape(3, 1)

    try:
        r, t = cv2.solvePnPRefineLM(objects, image, K, dist, r, t)
    except cv2.error:
        pass                              # keep the unrefined branch; it is scored below

    return r, t


def _candidate(placement, rvec, tvec, world, image_pts, K, dist, mount):
    """One solution, packaged with everything a chooser needs to judge it."""
    if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
        return None

    if float(np.asarray(tvec).ravel()[2]) <= 0.0:     # marker behind the lens
        return None

    T_wc = _camera_from_marker(placement, rvec, tvec)

    if not np.all(np.isfinite(T_wc)):
        return None

    err = _reproject_error(T_wc, world, image_pts, K, dist)

    if not math.isfinite(err):
        return None

    return (T_wc, _body_from_camera(T_wc, mount), err)


def _distinct(candidates):
    """Collapse branches that converged onto the same pose, keeping the best fit.

    The two branches of the ambiguity often fall into the same minimum once
    refined -- which is the solver saying the tag is not ambiguous from here.
    Left as duplicates they look like a disagreement and would trip the
    ambiguity rejection below for no reason.
    """
    out = []

    for c in sorted(candidates, key=lambda c: c[2]):
        if not any(_same_pose(c[1], k[1]) for k in out):
            out.append(c)

    return out


def _pose_gap(T_a, T_b):
    """(metres, degrees) between two body transforms, on the floor plane."""
    a, b = _flatten(T_a), _flatten(T_b)
    d = math.hypot(a.x - b.x, a.y - b.y)
    h = abs(math.degrees(math.atan2(math.sin(a.theta - b.theta),
                                    math.cos(a.theta - b.theta))))
    return d, h


def _same_pose(T_a, T_b):
    """Close enough that the robot would drive the same way either way."""
    d, h = _pose_gap(T_a, T_b)
    return d <= MARKERS.ambiguity_pos_m and h <= MARKERS.ambiguity_head_deg


def _solve_single(placement: MarkerPlacement, detection: MarkerDetection,
                  K, dist, mount: CameraMount):
    """Every camera pose consistent with one square marker.

    A square planar target has two poses that reproject almost identically:
    the true one, and its mirror through the tag plane. Both are returned,
    because a solver that silently picks one of them is how a robot ends up
    driving away from the tag it is looking at -- and because a caller given
    both can tell "one answer" from "two answers" and refuse the second.

    Every flag is run and the results pooled, rather than taking the first
    one that answers. Both halves of that matter, and both were measured:

      - IPPE_SQUARE is the solver written for this exact problem and is the
        only one that returns BOTH branches, but it is closed form: neither
        branch sits at a minimum. Each is refined before it is judged, or the
        chooser is comparing two wrong answers. On perfect corners the better
        raw branch was still 3 px and 60 cm out.
      - IPPE_SQUARE is also not always right. On small tags (~20 px across at
        2 m) under OpenCV 5.0.0 both its branches landed a metre out, in a
        minimum LM could not walk out of at any iteration budget, while SQPNP
        and ITERATIVE hit the true pose to 1e-13 px from the same corners.

    So the ambiguity needs IPPE_SQUARE and correctness needs the others. One
    pose per frame per tag is cheap; running all three is the reliable answer.
    """
    image = detection.corners.astype(np.float64).reshape(4, 1, 2)
    objects = placement.object_points()
    world = placement.world_points()

    raw = []

    for flag in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_SQPNP,
                 cv2.SOLVEPNP_ITERATIVE):
        try:
            n, rvecs, tvecs, _ = cv2.solvePnPGeneric(objects, image, K, dist,
                                                     flags=flag)
        except cv2.error:
            continue

        raw += [(rvecs[i], tvecs[i]) for i in range(n or 0)
                if np.all(np.isfinite(rvecs[i])) and np.all(np.isfinite(tvecs[i]))]

    out = []

    for rvec, tvec in raw:
        r, t = _refine(objects, image, K, dist, rvec, tvec)
        c = _candidate(placement, r, t, world, detection.corners, K, dist, mount)

        if c is not None:
            out.append(c)

    return _distinct(out)


def _plausible(candidate, mount: CameraMount) -> bool:
    """Could the chassis physically be here?

    Three facts the robot cannot violate: the floor is flat, so the body is
    level and its origin is on the floor, and the lens is bolted at a height
    somebody measured. A branch that fails any of these is not a competing
    hypothesis, however well it fits the corners.
    """
    T_wc, T_wb = candidate[0], candidate[1]

    return (_tilt_deg(T_wb) <= MARKERS.max_body_tilt_deg
            and abs(T_wb[2, 3]) <= MARKERS.max_body_z_m
            and abs(T_wc[2, 3] - mount.z) <= MARKERS.max_camera_height_err_m)


def _pick_candidate(candidates, mount: CameraMount):
    """Choose between the branches of the planar pose ambiguity, or refuse to.

    Only the corners get to choose. When one branch fits them far better it is
    simply right, and that is the normal case once every branch has been
    refined: on exact corners the true branch lands near 1e-6 px and the false
    one stays above 0.1 px. When no branch is decisively better, there is
    nothing left to choose WITH, and this returns None.

    Refusing is the point of the function, not its failure mode. The robot
    drives on this pose, so "I cannot tell" is a far more useful output than a
    coin flip, and the caller already handles it: None is the same path as a
    blank wall. None comes back when the surviving branches disagree about
    where the robot is, and when no branch is physically possible at all.
    """
    if not candidates:
        return None

    best_err = min(c[2] for c in candidates)

    # Comparable means "within a few times the best fit". The best fit's own
    # residual is the only estimate of the corner noise available, so a branch
    # fitting several times worse is worse than the noise can explain. The
    # absolute term is a floor for rounding when the best fit is essentially
    # exact; it must stay small, or every stuck solver branch becomes a
    # contender and recoverable poses get refused. See MARKERS.ambiguity_*.
    limit = max(best_err * MARKERS.ambiguity_ratio,
                best_err + MARKERS.ambiguity_abs_px)
    contenders = [c for c in candidates if c[2] <= limit]

    # Disagreement is tested BEFORE the physics filter, and that order is the
    # whole point. Physics is a veto here, not a vote: it can say "not that
    # one", it cannot say "therefore this one". Measured on a 20 px tag at
    # 2 m with half a pixel of corner noise, the branch the height check
    # preferred was the one 186 cm from the truth and the branch it threw out
    # was 4.5 cm from it. The lens-height residual of a CORRECT solve reaches
    # 26 cm at the 95th percentile in that regime, so it is noise at the same
    # scale as the thing it is being asked to judge.
    #
    # Filter first and one branch is left standing, which reads as certainty.
    # Reject first and the frame says what is true: it cannot tell.
    if any(not _same_pose(contenders[0][1], c[1]) for c in contenders[1:]):
        return None

    survivors = [c for c in contenders if _plausible(c, mount)]

    if not survivors:
        return None

    # Everything here agrees to within a few centimetres, so this only picks
    # the most physical of a set of answers that would drive the same way.
    return min(survivors, key=lambda c: abs(c[0][2, 3] - mount.z))


def _reproject_error(T_wc, world_pts, image_pts, K, dist) -> float:
    """RMS reprojection error in pixels for a camera-in-world transform."""
    T_cw = _inv(T_wc)
    rvec, _ = cv2.Rodrigues(T_cw[:3, :3])
    projected, _ = cv2.projectPoints(world_pts.reshape(-1, 1, 3), rvec,
                                     T_cw[:3, 3].reshape(3, 1), K, dist)
    residual = projected.reshape(-1, 2) - image_pts.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))


def _fuse(world_pts, image_pts, seed_T_wc, K, dist):
    """Refine one camera pose against every corner of every visible marker.

    This is the whole reason multiple markers help. Each tag alone is a tiny
    planar target whose distance and heading are weakly observed; two tags a
    metre apart on different walls are one wide, non-degenerate target, and
    a single least-squares fit over all their corners is far better
    conditioned than any average of the individual answers. Averaging also
    has nothing sensible to say when one tag picked the wrong ambiguity
    branch; a joint fit simply cannot satisfy both if one is wrong.
    """
    T_cw = _inv(seed_T_wc)
    rvec = cv2.Rodrigues(T_cw[:3, :3])[0].astype(np.float64)
    tvec = T_cw[:3, 3].reshape(3, 1).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(world_pts.reshape(-1, 1, 3),
                                  image_pts.reshape(-1, 1, 2), K, dist,
                                  rvec, tvec, useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)

    if not ok:
        return seed_T_wc, _reproject_error(seed_T_wc, world_pts, image_pts, K, dist)

    R_cw, _ = cv2.Rodrigues(rvec)
    refined = _inv(_rt(R_cw, tvec.reshape(3)))
    err = _reproject_error(refined, world_pts, image_pts, K, dist)
    seed_err = _reproject_error(seed_T_wc, world_pts, image_pts, K, dist)

    # The refinement is a local optimiser seeded from one marker. If it walks
    # somewhere worse, keep the seed rather than shipping the regression.
    return (refined, err) if err <= seed_err else (seed_T_wc, seed_err)


def _solve_all(usable, K, dist, mount: CameraMount):
    """Every camera pose consistent with ALL the visible markers at once.

    With one tag this is just its two ambiguity branches. With more, each
    branch of each tag is only a seed: the answer is the joint fit over every
    corner, and the branches are kept as seeds rather than as answers because
    a local optimiser started on the wrong side of the ambiguity stays there.
    Fitting from both sides and scoring the results is what lets a second tag
    resolve a flip the first tag could not.

    SQPNP is added as a seed that needs no guess at all -- it is a global
    solver over the pooled corners, so it does not inherit anyone's flip.
    """
    if len(usable) == 1:
        placement, det = usable[0]
        return _solve_single(placement, det, K, dist, mount)

    world_pts = np.concatenate([p.world_points() for p, _ in usable]).astype(np.float64)
    image_pts = np.concatenate([d.corners for _, d in usable]).astype(np.float64)

    seeds = []

    try:
        ok, rvec, tvec = cv2.solvePnP(world_pts.reshape(-1, 1, 3),
                                      image_pts.reshape(-1, 1, 2), K, dist,
                                      flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            seeds.append(_inv(_rt(cv2.Rodrigues(rvec)[0], tvec.reshape(3))))
    except cv2.error:
        pass

    # Two tags' worth of branches is enough to cover the flip; every extra tag
    # costs a full joint fit per branch and adds nothing a seed did not.
    for placement, det in usable[:2]:
        seeds += [c[0] for c in _solve_single(placement, det, K, dist, mount)]

    out = []

    for seed in seeds:
        T_wc, err = _fuse(world_pts, image_pts, seed, K, dist)

        if np.all(np.isfinite(T_wc)) and math.isfinite(err):
            out.append((T_wc, _body_from_camera(T_wc, mount), err))

    return _distinct(out)


def estimate_pose(detections, marker_map: MarkerMap, camera_matrix,
                  dist_coeffs=None, mount: CameraMount = None):
    """Absolute robot pose in the world frame, or None.

    None means "this frame did not say where the robot is". It is the normal
    case, not an error: the robot may be facing a blank wall. Callers hold
    the previous pose and its age; nothing here guesses.
    """
    mount = mount or CameraMount()
    dist_coeffs = np.zeros((5, 1)) if dist_coeffs is None else np.asarray(
        dist_coeffs, dtype=np.float64).reshape(-1, 1)
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    if not detections or marker_map is None or not len(marker_map):
        return None

    usable, rejected = [], []

    for det in detections:
        placement = marker_map.get(det.id)

        if placement is None:
            rejected.append(det.id)          # a tag from another arena, or a false read
        elif det.side_px < MARKERS.min_side_px:
            rejected.append(det.id)          # too few pixels for the corners to mean anything
        else:
            usable.append((placement, det))

    if not usable:
        return None

    # Largest tag first: more pixels across it means a better conditioned
    # single-marker solve, and a better seed for the joint one.
    usable.sort(key=lambda pair: pair[1].side_px, reverse=True)

    picked = _pick_candidate(_solve_all(usable, K, dist_coeffs, mount), mount)

    # One badly measured map entry drags the joint fit off EVERY marker at
    # once, so the largest residual does not identify it: with a marker 35 cm
    # off its map entry, the culprit showed 13 px of residual while an
    # innocent neighbour showed 21 px. Dropping on that number throws away
    # the wrong tag and keeps the lie.
    #
    # Refitting without each marker in turn is unambiguous, because it asks
    # the question that matters -- does the rest agree once this one is gone?
    # Measured on the same frame: dropping the bad marker left a set fitting
    # to 0.00 px, dropping any good one left the bad one still dragging at
    # 1.5-9.7 px. Only while three remain: falling back to a single marker
    # loses the reason we fused in the first place.
    #
    # "No answer" drives this loop as well as "a bad answer". A lying tag does
    # not merely add error: it can leave two seeds sitting on two different
    # compromises between the truth and the lie, which is a genuine ambiguity
    # and gets rejected. That rejection is a reason to go looking for an
    # outlier, not a reason to give up.
    while len(usable) > 2 and (picked is None
                               or picked[2] > MARKERS.max_reproj_px):
        trials = []

        for i in range(len(usable)):
            subset = usable[:i] + usable[i + 1:]
            trial = _pick_candidate(_solve_all(subset, K, dist_coeffs, mount), mount)

            if trial is not None:
                trials.append((trial[2], i, subset, trial))

        if not trials:
            break

        trial_err, worst_i, subset, trial = min(trials, key=lambda t: t[0])

        if picked is not None and trial_err >= picked[2]:
            break                    # no single marker explains the misfit

        rejected.append(usable[worst_i][1].id)
        usable, picked = subset, trial

    if picked is None:
        return None

    T_wc, err = picked[0], picked[2]

    # A pose nobody can reproject is a pose nobody should drive on, whatever
    # the geometry checks said about it.
    if err > MARKERS.max_reproj_px:
        return None

    T_wb = _body_from_camera(T_wc, mount)
    cam_xyz = T_wc[:3, 3]

    return PoseEstimate(
        pose=_flatten(T_wb),
        marker_ids=tuple(d.id for _, d in usable),
        reprojection_error_px=float(err),
        camera_height_m=float(cam_xyz[2]),
        body_tilt_deg=_tilt_deg(T_wb),
        nearest_marker_m=min(float(np.linalg.norm(p.translation - cam_xyz))
                             for p, _ in usable),
        rejected_ids=tuple(rejected),
    )


def pose_from_markers(detections, marker_map: MarkerMap, camera_matrix,
                      dist_coeffs=None, mount: CameraMount = None):
    """Pose in the world frame, or None if this frame cannot say. See estimate_pose."""
    estimate = estimate_pose(detections, marker_map, camera_matrix,
                             dist_coeffs, mount)
    return None if estimate is None else estimate.pose


def pose_from_frame(frame, marker_map: MarkerMap, camera_matrix,
                    dist_coeffs=None, mount: CameraMount = None):
    """detect() then estimate_pose(). Returns (estimate_or_None, detections)."""
    detections = detect(frame, marker_map.dictionary if marker_map else None)
    return (estimate_pose(detections, marker_map, camera_matrix, dist_coeffs, mount),
            detections)


# -----------------------------------------------------------
# OPERATOR DISPLAY
# -----------------------------------------------------------

def annotate(frame, detections, estimate: PoseEstimate = None):
    """Draw detected tags and the resulting pose onto a copy of the frame."""
    out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    used = set(estimate.marker_ids) if estimate else set()

    for det in detections:
        pts = det.corners.astype(int).reshape(-1, 1, 2)
        colour = (0, 255, 255) if det.id in used else (110, 110, 110)
        cv2.polylines(out, [pts], True, colour, 1)

        cx, cy = det.center
        cv2.putText(out, str(det.id), (int(cx) - 6, int(cy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

    if estimate is None:
        cv2.putText(out, "NO POSE", (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 1)
        return out

    p = estimate.pose
    cv2.putText(out, f"{p.x:+.2f} {p.y:+.2f} {math.degrees(p.theta):+.0f}deg",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(out, f"n={estimate.n_markers} e={estimate.reprojection_error_px:.2f}px",
                (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)

    return out


# -----------------------------------------------------------
# SYNTHETIC PROJECTION -- used by test_markers.py, and by anyone debugging
# -----------------------------------------------------------

def body_transform(pose: Pose) -> np.ndarray:
    """T_wb for a flat-floor pose. The inverse of _flatten."""
    return _rt(_rz(pose.theta), [pose.x, pose.y, 0.0])


def project_markers(pose: Pose, marker_map: MarkerMap, camera_matrix,
                    dist_coeffs=None, mount: CameraMount = None,
                    width: int = 320, height: int = 240, margin_px: float = 0.0):
    """Where a robot at `pose` would see each marker. Forward model, no solving.

    Skips markers behind the camera, off-frame, or facing away, so the output
    is what a camera would actually return. This is the ground truth the
    pose tests measure against, and it is deliberately written from the map
    and the mount directly rather than by reusing any part of the solver --
    a forward model that shares code with the thing it is testing cannot
    catch a sign error in the shared part.
    """
    mount = mount or CameraMount()
    dist_coeffs = np.zeros((5, 1)) if dist_coeffs is None else np.asarray(
        dist_coeffs, dtype=np.float64).reshape(-1, 1)
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    T_cw = _inv(body_transform(pose) @ mount.transform)
    rvec, _ = cv2.Rodrigues(T_cw[:3, :3])
    tvec = T_cw[:3, 3].reshape(3, 1)

    out = []

    for marker_id in marker_map.ids:
        placement = marker_map[marker_id]
        world = placement.world_points()

        in_cam = (T_cw[:3, :3] @ world.T).T + T_cw[:3, 3]

        if np.any(in_cam[:, 2] <= 0.05):                      # behind or in the lens
            continue

        # The tag is printed on one side of the paper. Seen from behind, or
        # from far enough round the side, there is nothing to detect. A real
        # detector gives up well before edge-on, so the forward model must
        # too, or the tests measure poses no camera could ever produce.
        to_camera = T_cw[:3, :3].T @ (-T_cw[:3, 3]) - placement.translation
        cos_view = float(placement.rotation[:, 2] @ to_camera
                         / max(np.linalg.norm(to_camera), 1e-9))

        if cos_view <= math.cos(math.radians(MARKERS.max_view_angle_deg)):
            continue

        pixels, _ = cv2.projectPoints(world.reshape(-1, 1, 3), rvec, tvec, K,
                                      dist_coeffs)
        pixels = pixels.reshape(4, 2)

        if (np.any(pixels[:, 0] < -margin_px) or np.any(pixels[:, 0] > width + margin_px)
                or np.any(pixels[:, 1] < -margin_px)
                or np.any(pixels[:, 1] > height + margin_px)):
            continue

        detection = MarkerDetection(marker_id, pixels.astype(np.float32))

        if detection.side_px < MARKERS.min_side_px:
            continue

        out.append(detection)

    return out


def render_markers(detections, marker_map: MarkerMap, width: int = 320,
                   height: int = 240, cell_px: int = 12):
    """Paint real tag bitmaps into a blank frame at the given image corners.

    Lets the test exercise cv2.aruco.detectMarkers itself -- corner order,
    dictionary, sub-pixel refinement -- instead of only the pose maths.
    """
    frame = np.full((height, width), 190, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, marker_map.dictionary))

    for det in detections:
        # One quiet-zone cell of white all round: detectMarkers needs the
        # border to find the tag at all.
        bits = cv2.aruco.generateImageMarker(dictionary, det.id, cell_px * 8,
                                             borderBits=1)
        pad = cell_px
        tile = np.full((bits.shape[0] + 2 * pad, bits.shape[1] + 2 * pad), 255,
                       dtype=np.uint8)
        tile[pad:pad + bits.shape[0], pad:pad + bits.shape[1]] = bits

        n = float(bits.shape[0])
        src = np.array([[pad, pad], [pad + n, pad], [pad + n, pad + n], [pad, pad + n]],
                       dtype=np.float32)

        H = cv2.getPerspectiveTransform(src, det.corners.astype(np.float32))
        warped = cv2.warpPerspective(tile, H, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_TRANSPARENT,
                                     dst=frame.copy())

        mask = cv2.warpPerspective(np.ones_like(tile) * 255, H, (width, height))
        frame = np.where(mask > 0, warped, frame).astype(np.uint8)

    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
