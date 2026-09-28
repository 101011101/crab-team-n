"""Checkerboard calibration for one camera.

    ./.venv/bin/python calibrate_webcam.py            # laptop webcam
    ./.venv/bin/python calibrate_webcam.py --device 1
    ./.venv/bin/python calibrate_webcam.py --url http://192.168.1.51:81/stream

Why this has to happen before anything else means anything: with guessed
intrinsics our own marker pose measures 16.5 cm and 4.4 degrees of error.
With a real calibration the same code measures 1.80 cm and 0.48 degrees.
Fifteen minutes buys an order of magnitude, and every distance downstream
inherits it.

Print a 9x6 checkerboard (that is INNER corners, so a 10x7 grid of squares),
measure one square with a ruler, and pass it with --square-mm. Hold the board
at many angles and distances, filling different parts of the frame -- corners
especially, because that is where distortion lives and where a calibration
that only saw the centre will lie to you.
"""

import argparse, json, os, time
import cv2, numpy as np

COLS, ROWS = 9, 6          # inner corners
MIN_SHOTS = 15
MAX_RMS_PX = 1.0


def board_points(square_m):
    p = np.zeros((COLS * ROWS, 3), np.float32)
    p[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)
    return p * square_m


def open_source(args):
    if args.url:
        cap = cv2.VideoCapture(args.url)
    else:
        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--url", default=None, help="MJPEG stream instead of a local device")
    ap.add_argument("--name", default="webcam", help="saved as calib/<name>.npz")
    ap.add_argument("--square-mm", type=float, default=20.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    cap = open_source(args)
    if not cap.isOpened():
        print("  could not open the camera"); return 1

    square_m = args.square_mm / 1000.0
    objp = board_points(square_m)
    obj_pts, img_pts = [], []
    size = None
    last_shot = 0.0
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    print(f"  {COLS}x{ROWS} inner corners, {args.square_mm:.1f} mm squares")
    print("  SPACE captures when the board is found. C calibrates. Q quits.")
    print("  Move the board around the frame -- corners matter most.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05); continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), flags)

        view = frame.copy()
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cv2.drawChessboardCorners(view, (COLS, ROWS), corners, found)

        colour = (90, 230, 140) if found else (90, 90, 240)
        cv2.putText(view, f"captured {len(obj_pts)}/{MIN_SHOTS}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        cv2.putText(view, "board FOUND" if found else "board not found",
                    (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.imshow("calibrate", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # Space, or hold the board still and it auto-captures -- one shot a
        # second, so a slow sweep round the frame fills the set on its own.
        take = (key == ord(" ")) or (found and time.time() - last_shot > 1.0 and key == ord("a"))
        if found and take:
            obj_pts.append(objp.copy()); img_pts.append(corners)
            last_shot = time.time()
            print(f"  captured {len(obj_pts)}")

        if key == ord("c") and len(obj_pts) >= MIN_SHOTS:
            break

    cap.release(); cv2.destroyAllWindows()

    if len(obj_pts) < MIN_SHOTS:
        print(f"\n  only {len(obj_pts)} shots, need {MIN_SHOTS}. Nothing saved.")
        return 1

    print(f"\n  calibrating on {len(obj_pts)} views...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)

    # Per-view error finds the one bad capture that quietly poisons the fit.
    errs = []
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        errs.append(float(cv2.norm(img_pts[i], proj, cv2.NORM_L2) / len(proj)))

    fx, fy = K[0, 0], K[1, 1]
    hfov = 2 * np.degrees(np.arctan(size[0] / (2 * fx)))
    print(f"  RMS reprojection {rms:.3f} px   (gate {MAX_RMS_PX})")
    print(f"  worst view       {max(errs):.3f} px")
    print(f"  fx {fx:.1f}  fy {fy:.1f}  cx {K[0,2]:.1f}  cy {K[1,2]:.1f}")
    print(f"  horizontal FOV   {hfov:.1f} deg")
    print(f"  distortion       {np.round(dist.ravel(), 4)}")

    if rms > MAX_RMS_PX:
        print(f"\n  RMS is above {MAX_RMS_PX} px. Saving anyway, but recapture:")
        print("  more angles, fill the frame corners, keep the board flat and sharp.")

    os.makedirs("calib", exist_ok=True)
    path = f"calib/{args.name}.npz"
    np.savez(path, K=K, dist=dist, size=np.array(size), rms=rms,
             square_m=square_m, shots=len(obj_pts), when=time.time())
    print(f"\n  wrote {path}")
    print(f"  use it:  ./.venv/bin/python tag_probe.py --calib {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
