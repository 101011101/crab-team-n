"""HSV tuner. Run this in the real arena lighting before every demo.

    python3 calibrate.py            # first configured bot
    python3 calibrate.py --bot 2
    python3 calibrate.py --webcam   # laptop camera, for testing with no bots

Click a green block in the video to seed the range from that pixel, then
nudge the sliders until the mask shows the blocks as solid white and
everything else black. Press P to print a config.py block, Q to quit.
"""

import argparse

import cv2
import numpy as np

from botlink import BotLink
from config import BOTS, VISION


WINDOW = "calibrate"
sampled = None


def on_mouse(event, x, y, flags, param):
    global sampled

    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param.get("frame")

        if frame is not None and y < frame.shape[0] and x < frame.shape[1]:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sampled = hsv[y, x].astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", type=int, default=None)
    parser.add_argument("--webcam", action="store_true")
    args = parser.parse_args()

    link = None
    cap = None

    if args.webcam:
        cap = cv2.VideoCapture(0)
    else:
        cfg = next((b for b in BOTS if b.bot_id == args.bot), BOTS[0])
        print(f"streaming from {cfg.name} at {cfg.stream_url}")
        link = BotLink(cfg)
        link.start()

    cv2.namedWindow(WINDOW)

    shared = {"frame": None}
    cv2.setMouseCallback(WINDOW, on_mouse, shared)

    names = ["H low", "S low", "V low", "H high", "S high", "V high"]
    limits = [179, 255, 255, 179, 255, 255]
    start = list(VISION.hsv_low) + list(VISION.hsv_high)

    for name, limit, value in zip(names, limits, start):
        cv2.createTrackbar(name, WINDOW, value, limit, lambda _: None)

    global sampled

    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            frame, _ = link.latest_frame()

        if frame is None:
            blank = np.zeros((240, 320, 3), np.uint8)
            cv2.putText(blank, "waiting for video...", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 240), 1)
            cv2.imshow(WINDOW, blank)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            continue

        shared["frame"] = frame

        if sampled is not None:
            h, s, v = sampled
            # A window around the clicked pixel: hue is tight, saturation
            # and value are loose because lighting moves them the most.
            for name, value in zip(names, [
                max(0, h - 12), max(40, s - 70), max(40, v - 70),
                min(179, h + 12), 255, 255,
            ]):
                cv2.setTrackbarPos(name, WINDOW, int(value))

            sampled = None

        values = [cv2.getTrackbarPos(n, WINDOW) for n in names]
        low = np.array(values[:3], np.uint8)
        high = np.array(values[3:], np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, low, high)

        kernel = np.ones((VISION.kernel, VISION.kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        shown = frame.copy()
        blobs = 0

        for contour in contours:
            if cv2.contourArea(contour) < VISION.min_area_px:
                continue

            blobs += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.putText(shown, f"blobs: {blobs}  (click a block to seed)",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        combined = np.hstack([shown, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(WINDOW, combined)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        if key == ord("p"):
            print("\nPaste into config.py VisionConfig:\n")
            print(f"    hsv_low = {tuple(int(v) for v in values[:3])}")
            print(f"    hsv_high = {tuple(int(v) for v in values[3:])}\n")

    if cap is not None:
        cap.release()
    if link is not None:
        link.shutdown()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
