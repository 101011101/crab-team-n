"""Integration check for the live stack, with no hardware and no network.

Proves the chain that actually matters:

    rendered tag view -> TagPose fix -> DeadReckoner -> control loop -> duty

A synthetic camera view is built by projecting a real tag36h11 marker from a KNOWN
robot pose, so the recovered pose can be compared against truth. That is the only
way to test a localiser without a room.

    ../central/.venv/bin/python test_live.py
"""
import math, os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "control"))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, _ROOT)

from pipeline.common import constants as C
from pipeline.common import geometry as Gm
import arena_map, pose_source as P
from tag_pose import TagPose
import grid as G
import reach_field as rf

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'  ok  ' if ok else ' FAIL '}] {name:46} {detail}")


def render_all_tags(robot_xy, robot_theta):
    """The whole camera view: every wall tag in shot, composited.

    This is what the robot actually sees. Testing one tag at a time measures
    single-tag PnP, which is the worst case; the localiser fuses everything in
    frame and is meaningfully better than any one tag.
    """
    img = np.full((C.CAM_H, C.CAM_W), 255, np.uint8)
    n = 0
    for tid in sorted(C.TAG_MAP):
        one = render_tag_view(tid, robot_xy, robot_theta, gray=True)
        if one is None:
            continue
        img = np.minimum(img, one)
        n += 1
    return (cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), n) if n else (None, 0)


def render_tag_view(tag_id, robot_xy, robot_theta, gray=False):
    """What the robot's camera sees of one wall tag, from a known pose."""
    Rwc, twc = Gm.world_T_cam(robot_xy[0], robot_xy[1], robot_theta)
    Rwt, twt = Gm.world_T_tag(tag_id)
    Rcw, tcw = Gm.invert(Rwc, twc)
    Rct, tct = Gm.compose(Rcw, tcw, Rwt, twt)
    if tct[2] <= 0.15:
        return None                      # behind the camera or too close
    obj = Gm.tag_object_points()
    rvec, _ = cv2.Rodrigues(Rct)
    proj, _ = cv2.projectPoints(obj, rvec, tct, Gm.camera_matrix(), Gm.dist_coeffs())
    dst = proj.reshape(4, 2).astype(np.float32)
    if dst.min() < -200 or dst[:, 0].max() > C.CAM_W + 200 or dst[:, 1].max() > C.CAM_H + 200:
        return None
    M, PAD = 400, 120
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tag_id, M)
    canvas = np.full((M + 2 * PAD, M + 2 * PAD), 255, np.uint8)
    canvas[PAD:PAD + M, PAD:PAD + M] = marker
    src = np.array([[PAD, PAD], [PAD + M - 1, PAD],
                    [PAD + M - 1, PAD + M - 1], [PAD, PAD + M - 1]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(canvas, H, (C.CAM_W, C.CAM_H), borderValue=255)
    out = np.minimum(np.full((C.CAM_H, C.CAM_W), 255, np.uint8), warp)
    return out if gray else cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def main():
    print("\n--- 1. the arena is metric and control/ reads it -------------------")
    gr, cell, origin, W, H = arena_map.build()
    m = arena_map.to_map(gr, cell, origin, W, H)
    import json, tempfile
    path = os.path.join(tempfile.gettempdir(), "_test_arena.json")
    json.dump(m, open(path, "w"))
    g = G.Grid(path)
    check("arena grid loads in control/grid.py", g.w == W and g.h == H,
          f"{g.w}x{g.h} cells of {g.cell} m")
    check("cell size is 5 cm", abs(g.cell - 0.05) < 1e-9)
    drive = g.drivable()
    check("most of the floor is drivable", drive.sum() > 0.7 * drive.size,
          f"{int(drive.sum())}/{drive.size}")
    a, b = g.to_cell(0.5, 0.5), g.to_cell(5.5, 3.5)
    d = rf.wavefront(drive, [b])
    reach_m = float(d[a]) * g.cell
    check("zone A reaches zone C", np.isfinite(d[a]), f"{reach_m:.2f} m of path")
    check("path is longer than the straight line", reach_m > math.hypot(5.0, 3.0),
          f"{reach_m:.2f} m > {math.hypot(5.0,3.0):.2f} m straight")

    print("\n--- 2. the motor model round-trips ---------------------------------")
    check("deadband produces no motion", P.duty_to_speed(C.DEADBAND) == 0.0)
    for v in (0.1, 0.22, 0.4):
        rt = P.duty_to_speed(P.speed_to_duty(v))
        check(f"{v} m/s survives the duty round trip", abs(rt - v) < 1e-6, f"{rt}")
    l, r = P.tracks_for(0.0, 1.4)
    check("a pure spin gives opposite duties", l == -r and r > 0, f"({l}, {r})")

    print("\n--- 3. a tag fix recovers a known pose -----------------------------")
    tp = TagPose()
    truths = [((1.0, 1.0), math.radians(80)), ((3.0, 1.2), math.radians(95)),
              ((2.0, 2.2), math.radians(70))]
    # A 0.15 m tag at 2-3 m is about 30 px across, so single-tag IPPE_SQUARE PnP
    # lands several centimetres out. That is the sensor, not a bug. The tolerance
    # below is what this geometry can actually deliver; tightening it would only
    # make the test lie.
    errs = []
    for xy, th in truths:
        img, n = render_all_tags(xy, th)
        if img is None:
            check(f"tag fix from {xy}", False, "no tag in view from this pose")
            continue
        fix = tp.fix(img)
        if fix is None:
            check(f"tag fix from {xy}", False, f"{n} tags rendered, none resolved")
            continue
        e = math.hypot(fix["x"] - xy[0], fix["y"] - xy[1])
        et = abs(math.degrees(math.atan2(math.sin(fix["theta"] - th),
                                         math.cos(fix["theta"] - th))))
        errs.append(e)
        check(f"tag fix from {xy}", e < 0.12 and et < 6.0,
              f"{e*100:.1f} cm, {et:.1f} deg, {fix['n_tags']}/{n} tags fused")
    if errs:
        med = float(np.median(errs))
        check("median tag fix error under 10 cm", med < 0.10, f"{med*100:.2f} cm")
        check("every fix is far better than dead reckoning drift",
              max(errs) < 0.15, f"worst {max(errs)*100:.1f} cm")

    print("\n--- 4. a fix corrects a drifted dead reckoner ----------------------")
    dr = P.DeadReckoner(1.0, 1.0, 0.0)
    t0 = time.time()
    for k in range(20):
        dr.command(1700, 1700, now=t0 + k * 0.1)
    drifted = dr.pose
    check("dead reckoning moves the robot", math.hypot(drifted[0] - 1.0, drifted[1] - 1.0) > 0.2,
          f"{math.hypot(drifted[0]-1.0, drifted[1]-1.0):.2f} m")
    check("confidence is zero before any fix", dr.confidence() == 0.0)
    dr.correct(1.0, 1.0, 0.0)
    check("a fix restores confidence", dr.confidence() > 0.9, f"{dr.confidence():.2f}")
    check("the correction jump is reported", dr.last_jump_m > 0.2,
          f"{dr.last_jump_m:.2f} m")

    print("\n--- 5. control output is a real duty pair --------------------------")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "control"))
    import loop as CL
    import live_loop as LL

    class FakeLink:
        online = True
        frames_received = 0
        command_failures = 0
        stream_error = ""
        def __init__(self): self.sent = []
        def drive(self, l, r): self.sent.append((l, r))
        def stop(self): self.sent.append((0, 0))
        def latest_frame(self): return None, float("inf")

    class FakeFleet:
        def __init__(self):
            self.links = {"scout-1": FakeLink()}
            self.cfg = {}
        def frames(self, max_age=2.0): return {}
        def drive(self, n, l, r): self.links[n].drive(l, r)
        def stop_all(self): [l.stop() for l in self.links.values()]
        def status(self): return {}

    class FakeMap:
        map_path = path
        def stats(self): return dict(poses={})

    fleet = FakeFleet()
    loop = LL.LiveLoop(fleet, FakeMap(), goal=(5.5, 3.5), tags=False)
    loop.dr["scout-1"] = P.DeadReckoner(0.5, 0.5, 0.0)
    loop.dr["scout-1"].correct(0.5, 0.5, 0.0)
    for _ in range(12):
        loop.tick(0.1)
    sent = [s for s in fleet.links["scout-1"].sent if s != (0, 0)]
    check("the loop issued drive commands", len(sent) > 0, f"{len(sent)} commands")
    if sent:
        mx = max(max(abs(a), abs(b)) for a, b in sent)
        check("duty stays inside the drive range", mx <= C.DUTY_MAX, f"peak {mx}")
        check("duty clears the deadband when moving",
              all(max(abs(a), abs(b)) > C.DEADBAND for a, b in sent),
              "no command below stiction")
    check("a goal was planned toward", loop.sim is not None and bool(loop.sim.robots[0].route)
          or loop.status()["mode"]["scout-1"] != "idle",
          loop.status()["mode"]["scout-1"])

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
