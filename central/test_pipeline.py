"""Drive the full laptop pipeline with synthetic frames: no robots needed.

    ./.venv/bin/python test_pipeline.py

Guards the rungs below whatever is being built. Run it before committing a rung.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from config import VISION, TARGETS
from vision import detect_green, annotate
from targets import TargetRegistry
from behavior import ScoutBehavior, SEARCH, APPROACH, CLAIM, BACKOFF

def frame_with_block(w=320, h=240, cx=160, size=30):
    """Grey room with one green block at cx, `size` px wide."""
    f = np.full((h, w, 3), 120, np.uint8)
    if size > 0:
        x0, y0 = max(0, cx - size//2), h//2 - size//2
        f[y0:y0+size, x0:x0+size] = (60, 200, 60)   # BGR green
    return f

# --- 1. detection + geometry ---
dets, mask = detect_green(frame_with_block(cx=160, size=40))
assert len(dets) == 1, f"expected 1 blob, got {len(dets)}"
assert abs(dets[0].bearing) < 0.05, f"centred block should have ~0 bearing, got {dets[0].bearing}"
print(f"  centred block: bearing={dets[0].bearing:+.3f} area_frac={dets[0].area_fraction:.3f}")

dets_r, _ = detect_green(frame_with_block(cx=280, size=40))
assert dets_r[0].bearing > 0.5, f"right block should have positive bearing, got {dets_r[0].bearing}"
dets_l, _ = detect_green(frame_with_block(cx=40, size=40))
assert dets_l[0].bearing < -0.5, f"left block should have negative bearing, got {dets_l[0].bearing}"
print(f"  right block:   bearing={dets_r[0].bearing:+.3f}   left block: bearing={dets_l[0].bearing:+.3f}")

empty, _ = detect_green(frame_with_block(size=0))
assert empty == [], "empty room must yield no detections"
print("  empty room:    0 detections")

# --- 2. steering sign: target on the right must turn the bot right ---
reg = TargetRegistry(TARGETS)
b = ScoutBehavior("test", "left", reg)
b.state, b.state_since, b.last_seen_at = APPROACH, time.time(), time.time()
l, r = b.step(dets_r, running=True, online=True)
assert l > r, f"target on right => left track must outrun right, got l={l} r={r}"
b.state, b.state_since, b.last_seen_at = APPROACH, time.time(), time.time()
l2, r2 = b.step(dets_l, running=True, online=True)
assert r2 > l2, f"target on left => right track must outrun left, got l={l2} r={r2}"
print(f"  steering: right-target l={l} r={r} | left-target l={l2} r={r2}")

# --- 3. full state walk: SEARCH -> APPROACH -> CLAIM -> BACKOFF -> SEARCH ---
reg = TargetRegistry(TARGETS)
b = ScoutBehavior("scout-1", "left", reg)
seen = []
big = frame_with_block(cx=160, size=150)          # fills >10% of frame
big_dets, _ = detect_green(big)
assert big_dets[0].area_fraction >= VISION.claim_area_fraction, \
    f"big block should exceed claim threshold, got {big_dets[0].area_fraction:.3f}"

# The block grows as the robot "drives" toward it, so APPROACH is a real
# phase rather than being skipped because the block already fills the frame.
t_end = time.time() + 10.0
size = 34
while time.time() < t_end:
    dets_now, _ = detect_green(frame_with_block(cx=200, size=int(size)))
    b.step(dets_now, running=True, online=True)
    if not seen or seen[-1] != b.state:
        seen.append(b.state)
    if b.state == APPROACH:
        size = min(150, size + 1.6)      # closing in
    time.sleep(0.02)

print(f"  state walk: {' -> '.join(seen)}")
for want in (SEARCH, APPROACH, CLAIM, BACKOFF):
    assert want in seen, f"never entered {want}: {seen}"
assert reg.count == 1, f"expected exactly 1 target registered, got {reg.count}"
print(f"  registry after one claim: count={reg.count} merges={reg.merges}")

# --- 4. refractory: the same block must not be re-claimed immediately ---
assert b.refractory_until > time.time(), "refractory timer not armed after backoff"
for _ in range(60):
    b.step(big_dets, running=True, online=True)
    time.sleep(0.01)
assert reg.count == 1, f"refractory failed: re-claimed to {reg.count} targets"
print(f"  refractory holds: still {reg.count} target after re-seeing the block")

# --- 5. dedup across two bots ---
reg2 = TargetRegistry(TARGETS)
t1, new1 = reg2.claim("scout-1", 0.12)
t2, new2 = reg2.claim("scout-2", 0.13)          # similar size, inside window
t3, new3 = reg2.claim("scout-2", 0.60)          # very different size
assert new1 and not new2 and new3, f"dedup wrong: {new1},{new2},{new3}"
assert reg2.count == 2 and reg2.merges == 1
print(f"  dedup: 3 claims -> {reg2.count} targets, {reg2.merges} merged")

# --- 6. safety: not running, or offline, must command zero ---
assert b.step(big_dets, running=False, online=True) == (0, 0), "stopped mission must send 0,0"
assert b.step(big_dets, running=True, online=False) == (0, 0), "offline bot must send 0,0"
print("  safety: stopped mission and offline bot both command (0, 0)")

# --- 7. UI renders ---
import ui
views = [{"name": "scout-1", "frame": annotate(big, big_dets, big_dets[0]), "state": "CLAIM",
          "online": True, "fps": 11.2, "event": "FOUND target #1", "absorbed": [], "refractory": 0.0},
         {"name": "scout-2", "frame": None, "state": "OFFLINE",
          "online": False, "fps": 0.0, "event": "", "absorbed": [], "refractory": 0.0}]
canvas = ui.render(views, reg, True, 42.3, [(1.0, "mission STARTED"), (12.0, "scout-2 LOST")])
assert canvas.shape[0] == ui.PANEL_H + ui.BAR_H, canvas.shape
assert canvas.shape[1] >= ui.PANEL_W * 2, canvas.shape
cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_snapshot.png"), canvas)
print(f"  ui renders: {canvas.shape[1]}x{canvas.shape[0]} -> ui_snapshot.png")

print("\nALL PIPELINE CHECKS PASSED")
