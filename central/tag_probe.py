"""Watch a camera move around an AprilTag, and draw where it has been.

    ./.venv/bin/python tag_probe.py --calib webcam --tag-mm 100
    ./.venv/bin/python tag_probe.py --url http://172.20.10.2:8080/stream --calib robot1

Tape one tag to a wall, point a camera at it, and walk around. The tag is
the origin of the world; everything else is measured from it. This proves
the entire localization chain -- detect, solve, invert, accumulate -- on
hardware anyone already owns, before a robot exists to be wrong on.

The one subtlety worth stating: solvePnP gives the pose of the TAG in the
CAMERA's frame. What we want is the opposite, where the camera is in the
world. That is one inversion, and getting its direction wrong produces a
confident, clean-looking, completely mirrored trajectory. The check is
physical: walk right, the path should go right.

Press S to save the path to a JSON the 3D viewer reads. R clears it.
"""

import argparse, json, os, time
import cv2, numpy as np

# Marker frame: x right across the tag, y UP the tag, z OUT of the printed
# face. This matches markers.py. The other convention in circulation puts z
# INTO the face, which is a 180 degree flip about x -- see docs/merge-plan.md.
def object_points(size_m):
    h = size_m / 2.0
    return np.array([[-h,  h, 0.0],
                     [ h,  h, 0.0],
                     [ h, -h, 0.0],
                     [-h, -h, 0.0]], np.float32)


def solve_tag(objp, img_pts, K, dist):
    """Camera position in the tag's frame, or None when it cannot be told.

    A square tag seen near head-on has TWO poses that reproject almost
    identically and differ by a flip. Taking solvePnP's first answer gets
    one at random, which on screen looks like a path that jumps to the wrong
    side of the wall. So generate both, refine both, and only accept a
    winner when the corners actually prefer it -- or when physics rules the
    other one out, because the printed face has to be pointing at us.
    """
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            objp, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return None, None, None
    if not n:
        return None, None, None

    cands = []
    for i in range(n):
        rv, tv = cv2.solvePnPRefineLM(objp, img_pts, K, dist,
                                      rvecs[i].copy(), tvecs[i].copy())
        proj, _ = cv2.projectPoints(objp, rv, tv, K, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1,2) - img_pts, axis=1)))
        R, _ = cv2.Rodrigues(rv)
        cam = (-R.T @ tv).ravel()
        # The tag's +z points out of its printed face. If the camera is
        # behind that face it cannot be seeing the tag at all.
        visible = cam[2] > 0
        cands.append((err, rv, tv, cam, visible))

    cands.sort(key=lambda c: c[0])
    live = [c for c in cands if c[4]] or cands

    if len(live) > 1 and live[1][0] <= max(live[0][0] * 1.15, live[0][0] + 0.05):
        gap = float(np.linalg.norm(live[0][3] - live[1][3]))
        if gap > 0.05:
            return None, None, None          # genuinely ambiguous, say so

    err, rv, tv, cam, _ = live[0]
    return cam, rv, tv


def load_calib(name):
    path = f"calib/{name}.npz"
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return dict(K=d["K"], dist=d["dist"], rms=float(d["rms"]),
                size=tuple(int(v) for v in d["size"]),
                when=float(d["when"]) if "when" in d else 0.0)


def guessed_calib(w, h, hfov_deg=60.0):
    f = (w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    return dict(K=K, dist=np.zeros((5, 1)), rms=float("nan"), size=(w, h), when=0.0)


def draw_axes(img, K, dist, rvec, tvec, size_m):
    L = size_m * 0.8
    pts = np.float32([[0,0,0],[L,0,0],[0,L,0],[0,0,L]]).reshape(-1,3)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    p = proj.reshape(-1,2).astype(int)
    for end, colour, label in ((1,(60,60,240),"x"), (2,(90,230,140),"y"), (3,(240,180,60),"z")):
        cv2.arrowedLine(img, tuple(p[0]), tuple(p[end]), colour, 3, tipLength=0.18)
        cv2.putText(img, label, tuple(p[end]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def minimap(path, w=300, h=300, span=2.0):
    """Top-down view of the camera track, in the tag's frame.

    The tag sits at the origin facing +z, so the plot is x across and z away
    -- what you would see looking down on the room from above.
    """
    img = np.full((h, w, 3), 18, np.uint8)
    cx, cy = w // 2, int(h * 0.82)
    s = min(w, h) / (2 * span)

    for r in (0.5, 1.0, 1.5, 2.0):
        cv2.circle(img, (cx, cy), int(r * s), (38, 44, 48), 1)
        cv2.putText(img, f"{r:g}m", (cx + int(r*s) - 22, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 105, 112), 1)

    cv2.line(img, (cx-14, cy), (cx+14, cy), (240, 180, 60), 3)   # the tag
    cv2.putText(img, "tag", (cx+18, cy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240,180,60), 1)

    prev = None
    for i, (x, y, z) in enumerate(path):
        px, py = int(cx + x * s), int(cy - z * s)
        if prev is not None:
            t = i / max(len(path)-1, 1)
            cv2.line(img, prev, (px, py),
                     (int(80+120*t), int(130+90*t), int(240-40*t)), 2)
        prev = (px, py)
    if prev is not None:
        cv2.circle(img, prev, 5, (90, 230, 140), -1)
    return img


def save_path(path, args, calib, guessed, W, H, quiet=False):
    """Write the trajectory. Returns the filename, or None if there is none.

    Called on S, periodically, and on the way out. Walking a loop and then
    pressing Q used to discard the whole recording, which is a bad trade for
    the few milliseconds a write costs.
    """
    if not path:
        return None

    out = dict(path=path, tag_mm=args.tag_mm, dict=args.dict,
               calibrated=not guessed, rms=calib["rms"], size=[W, H])
    with open("tag_path.json", "w") as f:
        json.dump(out, f)

    if not quiet:
        span = np.ptp(np.array(path), axis=0)
        print(f"  saved tag_path.json -- {len(path)} poses, "
              f"span {span[0]:.2f} x {span[1]:.2f} x {span[2]:.2f} m")
    return "tag_path.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--url", default=None)
    ap.add_argument("--calib", default="webcam")
    ap.add_argument("--tag-mm", type=float, default=100.0)
    ap.add_argument("--dict", default="36h11", choices=["36h11", "4x4"])
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11 if args.dict == "36h11" else cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    cap = cv2.VideoCapture(args.url) if args.url else cv2.VideoCapture(args.device)
    if not args.url:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("  could not open the camera"); return 1

    ok, probe = cap.read()
    if not ok:
        print("  no frames"); return 1
    H, W = probe.shape[:2]

    calib = load_calib(args.calib)
    guessed = calib is None
    if guessed:
        calib = guessed_calib(W, H)
        print(f"\n  !! NO CALIBRATION -- calib/{args.calib}.npz not found.")
        print("  !! Intrinsics are a 60 deg guess. Measured cost of this:")
        print("  !! 16.5 cm and 4.4 deg of error, against 1.80 cm calibrated.")
        print(f"  !! Fix it:  ./.venv/bin/python calibrate_webcam.py --name {args.calib}\n")
    else:
        if tuple(calib["size"]) != (W, H):
            print(f"\n  !! calibration was taken at {tuple(calib['size'])} but the camera "
                  f"is {W}x{H}. Intrinsics do not survive a resolution change. Recalibrate.\n")
        else:
            age_d = (time.time() - calib["when"]) / 86400 if calib["when"] else 0
            print(f"\n  calibration {args.calib}: RMS {calib['rms']:.3f} px, "
                  f"{W}x{H}, {age_d:.1f} days old\n")

    K, dist = calib["K"], calib["dist"]
    size_m = args.tag_mm / 1000.0
    objp = object_points(size_m)

    path, last_pose, frames, t0 = [], None, 0, time.time()
    last_auto = 0
    print("  Click the VIDEO WINDOW first -- keys go there, not this terminal.")
    print("  S saves now   R resets   Q quits.  Ctrl-C also saves.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.03); continue
            frames += 1

            corners, ids, _ = detector.detectMarkers(frame)
            view = frame.copy()
            pose_txt = "no tag"

            # OpenCV 5 returns ids as (N,); earlier versions returned (N,1).
            # Flatten once so the rest of the loop does not care which.
            tag_ids = np.asarray(ids).ravel() if ids is not None else np.empty(0, int)

            if tag_ids.size:
                cv2.aruco.drawDetectedMarkers(view, corners, ids)

                # Use the largest tag in view: the smallest reprojection error
                # and the least ambiguous geometry go with the most pixels.
                areas = [cv2.contourArea(c.reshape(-1, 2)) for c in corners]
                i = int(np.argmax(areas))
                img_pts = corners[i].reshape(-1, 2).astype(np.float32)

                cam, rvec, tvec = solve_tag(objp, img_pts, K, dist)

                if cam is None:
                    pose_txt = "tag seen, pose ambiguous -- move off head-on"
                else:
                    draw_axes(view, K, dist, rvec, tvec, size_m)
                    path.append(tuple(float(v) for v in cam))
                    if len(path) > 4000:
                        del path[:len(path)-4000]

                    last_pose = cam
                    dist_m = float(np.linalg.norm(tvec))
                    pose_txt = (f"cam x{cam[0]:+.2f} y{cam[1]:+.2f} z{cam[2]:+.2f} m"
                                f"   range {dist_m:.2f} m   id {int(tag_ids[i])}")

            bar = 34
            cv2.rectangle(view, (0,0), (view.shape[1], bar), (0,0,0), -1)
            cv2.putText(view, pose_txt, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (220,230,235) if last_pose is not None else (120,130,138), 1)
            if guessed:
                cv2.putText(view, "UNCALIBRATED", (view.shape[1]-190, 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60,60,240), 2)

            # On the image, not the terminal: this window owns the
            # keyboard and a user watching video never sees the shell.
            cv2.putText(view, "S save   R reset   Q quit",
                        (10, view.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (150, 165, 172), 1)

            mm = minimap(path)
            vh = view.shape[0]
            pad = np.full((vh, mm.shape[1], 3), 18, np.uint8)
            pad[:mm.shape[0]] = mm
            cv2.putText(pad, f"{len(path)} poses", (10, mm.shape[0]+26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,165,172), 1)
            cv2.putText(pad, f"{frames/(time.time()-t0):.1f} fps", (10, mm.shape[0]+48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,165,172), 1)
            cv2.imshow("tag probe", np.hstack([view, pad]))

            # Autosave every 100 new poses, so a crash or a closed window costs
            # a couple of seconds of walking rather than the whole run.
            if len(path) - last_auto >= 100:
                save_path(path, args, calib, guessed, W, H, quiet=True)
                last_auto = len(path)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                path.clear(); last_auto = 0; print("  path cleared")
            if key == ord("s"):
                save_path(path, args, calib, guessed, W, H)

    except KeyboardInterrupt:
        print("\n  interrupted -- saving")
    cap.release(); cv2.destroyAllWindows()

    if save_path(path, args, calib, guessed, W, H) is None:
        print("  no poses recorded -- the tag was never solved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
