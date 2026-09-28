"""Generate printable AprilTags with the size written on them.

    ./.venv/bin/python print_tag.py --ids 0 1 2 3 --mm 100

The printed size is the BLACK SQUARE, not the paper. Printing it on the tag
is not decoration: a tag scaled by a printer's "fit to page" silently scales
every distance derived from it, and nothing downstream can detect that.
Measure one with a ruler after printing and trust the ruler.
"""
import argparse, cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3])
ap.add_argument("--mm", type=float, default=100.0)
ap.add_argument("--dict", default="36h11", choices=["36h11", "4x4"])
ap.add_argument("--dpi", type=int, default=300)
a = ap.parse_args()

d = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11 if a.dict == "36h11" else cv2.aruco.DICT_4X4_50)
px = int(round(a.mm / 25.4 * a.dpi))
quiet = int(px * 0.25)

for i in a.ids:
    tag = cv2.aruco.generateImageMarker(d, i, px)
    sheet = np.full((px + 2*quiet + int(px*0.22), px + 2*quiet), 255, np.uint8)
    sheet[quiet:quiet+px, quiet:quiet+px] = tag
    label = f"{a.dict}  id {i}   black square = {a.mm:.0f} mm"
    cv2.putText(sheet, label, (quiet, px + 2*quiet + int(px*0.13)),
                cv2.FONT_HERSHEY_SIMPLEX, px/900, 0, max(1, px//400))
    name = f"tag_{a.dict}_{i}_{int(a.mm)}mm.png"
    cv2.imwrite(name, sheet)
    print(f"  {name}  ({px}px at {a.dpi}dpi)")
print("\n  Print at 100% / actual size. Do NOT let the dialog scale to fit.")
print("  Then measure the black square with a ruler and use that number.")
