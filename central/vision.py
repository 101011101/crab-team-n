"""Green block detection.

Colour thresholding in HSV, not RGB. In RGB a green block in shadow and the
same block in light have completely different values, so no fixed range
catches both. HSV splits hue (which colour) from value (how bright), so one
hue range survives the lighting variation across the arena.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from config import VISION


@dataclass
class Detection:
    cx: int              # centre x in pixels
    cy: int
    area: int            # filled pixel count
    bbox: tuple          # x, y, w, h
    frame_w: int
    frame_h: int

    @property
    def bearing(self) -> float:
        """Horizontal offset from centre, -1.0 (hard left) to +1.0 (hard right)."""
        return (self.cx - self.frame_w / 2.0) / (self.frame_w / 2.0)

    @property
    def area_fraction(self) -> float:
        """Share of the frame filled. Our only proxy for distance."""
        return self.area / float(self.frame_w * self.frame_h)


def build_mask(frame, cfg=VISION):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,
                       np.array(cfg.hsv_low, dtype=np.uint8),
                       np.array(cfg.hsv_high, dtype=np.uint8))

    kernel = np.ones((cfg.kernel, cfg.kernel), np.uint8)

    # Open removes isolated speckle, close fills holes punched by glare on
    # the block's face. Without close, one block often splits into two
    # contours and gets counted twice.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def detect_green(frame, cfg=VISION):
    """Return detections, largest first. Largest means nearest, roughly."""
    if frame is None:
        return [], None

    h, w = frame.shape[:2]
    mask = build_mask(frame, cfg)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for contour in contours:
        area = int(cv2.contourArea(contour))

        if area < cfg.min_area_px:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        detections.append(Detection(
            cx=x + bw // 2,
            cy=y + bh // 2,
            area=area,
            bbox=(x, y, bw, bh),
            frame_w=w,
            frame_h=h,
        ))

    detections.sort(key=lambda d: d.area, reverse=True)

    return detections[:cfg.max_detections], mask


def annotate(frame, detections, primary=None):
    """Draw detections onto a copy of the frame for the operator display."""
    out = frame.copy()
    h, w = out.shape[:2]

    cv2.line(out, (w // 2, 0), (w // 2, h), (90, 90, 90), 1)

    for det in detections:
        x, y, bw, bh = det.bbox
        is_primary = primary is not None and det is primary

        colour = (0, 255, 255) if is_primary else (0, 180, 0)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), colour, 2 if is_primary else 1)

        if is_primary:
            cv2.circle(out, (det.cx, det.cy), 4, (0, 255, 255), -1)
            cv2.putText(out, f"{det.area_fraction*100:.1f}%",
                        (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1)

    return out
