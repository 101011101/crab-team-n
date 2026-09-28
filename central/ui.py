"""Operator display: every robot's view, its state, and the shared target list.

Layout note: the robot panels sit in a row and the mission status runs as a
full-width bar underneath. An earlier version put the status in a side
column, which meant the target and event lists were capped by the height of
a single video panel and showed almost nothing once a mission got going.
Along the bottom there is width to spare, so the lists get real columns.
"""

import cv2
import numpy as np


PANEL_W, PANEL_H = 560, 420      # 4:3, matching the QVGA camera
BAR_H = 210
ROW = 19                          # pixels per list line

FONT = cv2.FONT_HERSHEY_SIMPLEX

BG = (28, 28, 32)
OK = (120, 230, 120)
WARN = (90, 200, 255)
BAD = (90, 90, 240)
TEXT = (235, 235, 235)
DIM = (150, 150, 155)
RULE = (70, 70, 78)

STATE_COLOR = {
    "SEARCH": WARN,
    "APPROACH": (255, 200, 90),
    "CLAIM": OK,
    "BACKOFF": (200, 140, 255),
    "IDLE": DIM,
    "OFFLINE": BAD,
}

KEYS = ("SPACE start/stop   1/2 drop a robot   R restore   "
        "M mask view   S save survey   Q quit")


# ---------------------------------------------------------------
# ROBOT PANELS
# ---------------------------------------------------------------

def _bot_panel(view):
    frame = view.get("frame")

    if frame is None:
        panel = np.full((PANEL_H, PANEL_W, 3), (18, 18, 22), np.uint8)

        if view["online"]:
            # Online but no picture yet: worth saying so. When offline the
            # stamp below already explains it.
            cv2.putText(panel, f"{view['name']}: waiting for video",
                        (20, PANEL_H // 2), FONT, 0.7, WARN, 2)
    else:
        panel = cv2.resize(frame, (PANEL_W, PANEL_H))

    cv2.rectangle(panel, (0, 0), (PANEL_W, 30), (0, 0, 0), -1)

    colour = STATE_COLOR.get(view["state"], TEXT)

    cv2.putText(panel, view["name"], (10, 21), FONT, 0.6, TEXT, 1)
    cv2.putText(panel, view["state"], (150, 21), FONT, 0.6, colour, 2)
    cv2.putText(panel, f"{view['fps']:4.1f} fps", (PANEL_W - 95, 21),
                FONT, 0.5, OK if view["online"] else BAD, 1)

    cv2.rectangle(panel, (0, PANEL_H - 26), (PANEL_W, PANEL_H), (0, 0, 0), -1)

    note = view.get("event", "")

    if view.get("refractory", 0) > 0:
        note = f"ignoring blocks {view['refractory']:.1f}s | {note}"

    if view.get("absorbed"):
        note = f"covering {','.join(view['absorbed'])} | {note}"

    cv2.putText(panel, note[:72], (10, PANEL_H - 8), FONT, 0.42, DIM, 1)

    if not view["online"]:
        tint = panel.copy()
        cv2.rectangle(tint, (0, 0), (PANEL_W, PANEL_H), (0, 0, 60), -1)
        panel = cv2.addWeighted(tint, 0.45, panel, 0.55, 0)
        cv2.putText(panel, "OFFLINE", (PANEL_W // 2 - 100, PANEL_H // 2),
                    FONT, 1.2, BAD, 3)

    cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), (60, 60, 68), 1)

    return panel


# ---------------------------------------------------------------
# STATUS BAR
# ---------------------------------------------------------------

def _column(bar, x, width, title, lines, max_rows, empty_text):
    """One titled list column, newest entries kept, older ones summarised."""
    cv2.putText(bar, title, (x, 24), FONT, 0.5, TEXT, 1)
    cv2.line(bar, (x, 32), (x + width, 32), RULE, 1)

    y = 52

    if not lines:
        cv2.putText(bar, empty_text, (x + 4, y), FONT, 0.45, DIM, 1)
        return

    hidden = len(lines) - max_rows

    if hidden > 0:
        cv2.putText(bar, f"+{hidden} earlier", (x + 4, y), FONT, 0.4,
                    (110, 110, 118), 1)
        y += ROW
        max_rows -= 1

    for text, colour in lines[-max_rows:] if max_rows > 0 else []:
        cv2.putText(bar, text, (x + 4, y), FONT, 0.45, colour, 1)
        y += ROW


def _status_bar(registry, running, elapsed, views, events, width):
    bar = np.full((BAR_H, width, 3), BG, np.uint8)

    rows = (BAR_H - 52 - 26) // ROW

    # --- mission column ---
    cv2.putText(bar, "MISSION", (16, 24), FONT, 0.5, TEXT, 1)
    cv2.line(bar, (16, 32), (300, 32), RULE, 1)

    online = sum(1 for v in views if v["online"])

    cv2.putText(bar, "RUNNING" if running else "STOPPED", (20, 56),
                FONT, 0.6, OK if running else BAD, 2)
    cv2.putText(bar, f"{elapsed:.1f}s", (170, 56), FONT, 0.55, TEXT, 1)

    cv2.putText(bar, f"robots online       {online}/{len(views)}", (20, 84),
                FONT, 0.48, TEXT if online == len(views) else BAD, 1)
    cv2.putText(bar, f"targets found       {registry.count}", (20, 106),
                FONT, 0.48, OK, 1)
    cv2.putText(bar, f"duplicates merged   {registry.merges}", (20, 128),
                FONT, 0.48, DIM, 1)

    # --- target column ---
    targets = registry.all()

    _column(bar, 330, 300, "TARGET RECORD",
            [(f"#{t.target_id}  {t.first_seen_by}  x{t.sightings} sighting(s)", OK)
             for t in targets],
            rows, "none yet")

    # --- event column ---
    _column(bar, 670, width - 690, "EVENTS",
            [(f"{stamp:6.1f}s  {text}", DIM) for stamp, text in events],
            rows, "none yet")

    cv2.line(bar, (16, BAR_H - 24), (width - 16, BAR_H - 24), RULE, 1)
    cv2.putText(bar, KEYS, (20, BAR_H - 8), FONT, 0.42, DIM, 1)

    return bar


def render(views, registry, running, elapsed, events):
    panels = [_bot_panel(v) for v in views]

    if not panels:
        blank = np.full((PANEL_H, PANEL_W, 3), (18, 18, 22), np.uint8)
        cv2.putText(blank, "no robots configured", (20, PANEL_H // 2),
                    FONT, 0.7, BAD, 2)
        panels = [blank]

    strip = np.hstack(panels)

    # The bar needs room for three columns even with one robot on screen.
    width = max(strip.shape[1], 1120)

    if width > strip.shape[1]:
        pad = np.full((strip.shape[0], width - strip.shape[1], 3), BG, np.uint8)
        strip = np.hstack([strip, pad])

    bar = _status_bar(registry, running, elapsed, views, events, width)

    return np.vstack([strip, bar])
