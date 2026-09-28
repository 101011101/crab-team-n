"""Turn a raw COLMAP model into the cleaned, floor-aligned world every later stage uses.

Two jobs, both of which every downstream stage would otherwise get wrong:

1. THROW OUT WHAT DIVERGED. Monocular SfM can register a frame and still put the
   camera at 1e14 units. Those poses, and points far outside the camera path, must
   go before any statistic is computed or any map is drawn.

2. PUT THE FLOOR ON THE FLOOR. COLMAP's world axes are arbitrary. A 2D map is only
   meaningful once one axis is "up", so we take the room's dominant plane (the lino)
   as the floor and rotate its normal onto +Y. After this, Y is height and XZ is the
   ground plane.

Scale stays arbitrary throughout. Monocular SfM fixes a scene only up to a 7-DoF
similarity, so every distance here is in MODEL UNITS and there is no metre anywhere.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_io import largest_model, read_images, read_points, read_image_ids

CAM_DIVERGED = 10.0   # model units from the camera median; past this a pose ran away
POINT_RADII = 5.0     # keep points within this many camera-path radii
MIN_TRACK = 3         # a point seen in fewer than this many views is not trustworthy


class World:
    """A cleaned, floor-aligned reconstruction.

    P      (n,3) points, Y up, origin at the point centroid
    C      (n,3) uint8 colours
    track  (n,)  how many images observed each point
    mask   (n,)  bitfield of which source clips observed it (see clip_bits)
    cam    (m,3) camera centres, same frame as P
    cam_clip (m,) one-character clip tag per camera
    R      median camera-path radius: the natural unit for "how big is this room"
    """

    def __init__(self, P, C, track, mask, cam, cam_clip, R, dropped, clip_bits, model,
                 cam_names=None):
        self.P, self.C, self.track, self.mask = P, C, track, mask
        self.cam, self.cam_clip, self.R = cam, cam_clip, R
        self.cam_names = list(cam_names or [])   # image filename per kept camera
        self.dropped, self.clip_bits, self.model = dropped, clip_bits, model

    @property
    def height(self):
        return self.P[:, 1]

    def floor_level(self):
        """Modal height. The floor is the single most-sampled surface in the room."""
        h, edges = np.histogram(self.P[:, 1], bins=60)
        return float(edges[h.argmax()])


def load(sparse_dir, clip_of=lambda name: name[0], min_track=MIN_TRACK):
    """Read the largest model under sparse_dir and clean it. clip_of maps an image
    name to a short source tag, so points can be attributed to the clip that saw them."""
    model, n_img = largest_model(sparse_dir)
    ims = read_images(model)
    P, C, E, T, tracks = read_points(model, with_tracks=True)
    ids = read_image_ids(model)

    names = sorted(ims)
    cam = np.array([ims[n][2] for n in names])
    clips = sorted({clip_of(n) for n in names})
    bits = {c: 1 << i for i, c in enumerate(clips)}

    keep_cam = np.linalg.norm(cam - np.median(cam, 0), axis=1) < CAM_DIVERGED
    dropped = [n for n, k in zip(names, keep_cam) if not k]
    cc = cam[keep_cam]
    R = float(np.median(np.linalg.norm(cc - cc.mean(0), axis=1)))

    mask = np.array([
        int(np.bitwise_or.reduce([bits.get(clip_of(ids.get(i, "?")), 0) for i in tr]))
        if tr else 0 for tr in tracks], dtype=np.uint8)

    keep = (T >= min_track) & (np.linalg.norm(P - cc.mean(0), axis=1) < POINT_RADII * R)
    P, C, T, mask = P[keep], C[keep], T[keep], mask[keep]

    # floor plane: weakest PCA direction of the cloud, signed so cameras are above it
    o = P.mean(0)
    _, _, Vt = np.linalg.svd(P - o, full_matrices=False)
    n = Vt[2]
    if np.dot(cc.mean(0) - o, n) < 0:
        n = -n
    e1 = Vt[0] - np.dot(Vt[0], n) * n
    e1 /= np.linalg.norm(e1)
    B = np.vstack([e1, n, np.cross(n, e1)])       # rows: right, up, forward

    return World(P=(P - o) @ B.T, C=C, track=T, mask=mask,
                 cam=(cc - o) @ B.T,
                 cam_clip=np.array([clip_of(n) for n, k in zip(names, keep_cam) if k]),
                 R=R, dropped=dropped, clip_bits=bits, model=model,
                 cam_names=[n for n, k in zip(names, keep_cam) if k])
