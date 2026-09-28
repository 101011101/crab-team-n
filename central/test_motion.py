"""Close the waypoint loop against a kinematic robot: no hardware needed.

    ./.venv/bin/python test_motion.py

The integrator below is the minimum needed to test the controller. T3 owns
the real simulator; this one is deliberately throwaway.
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DRIVE, MOTION
from drive import apply_drive
from motion import Pose, Waypoint, WaypointController, wrap_angle

DT = 0.05                    # 20 Hz, twice the command heartbeat

def integrate(pose, left, right, dt=DT):
    """Differential drive: track speeds in, pose out."""
    lv = left / 4095.0 * MOTION.max_speed_mps
    rv = right / 4095.0 * MOTION.max_speed_mps
    v = 0.5 * (lv + rv)
    # l - r = 2*k*w with w clockwise positive, so CCW theta rate is (r - l)/width.
    omega = (rv - lv) / MOTION.track_width_m
    return Pose(pose.x + v * math.cos(pose.theta) * dt,
                pose.y + v * math.sin(pose.theta) * dt,
                pose.theta + omega * dt)

def run(pose, wp, max_steps=1200):
    """Drive until arrived or out of steps. Returns (controller, poses)."""
    ctl = WaypointController()
    poses = [pose]
    for _ in range(max_steps):
        left, right = ctl.step(pose, wp)
        assert isinstance(left, int) and isinstance(right, int), (left, right)
        assert -4095 <= left <= 4095 and -4095 <= right <= 4095, (left, right)
        if ctl.arrived:
            break
        pose = integrate(pose, left, right)
        poses.append(pose)
    return ctl, poses

def gap(pose, wp):
    return math.hypot(wp.x - pose.x, wp.y - pose.y)

# --- 1. angle wrapping: the classic bug ---
assert wrap_angle(0.0) == 0.0
assert wrap_angle(math.pi) == math.pi                      # (-pi, pi], pi included
assert abs(wrap_angle(-math.pi) - math.pi) < 1e-12         # -pi folds up to +pi
assert abs(wrap_angle(3 * math.pi / 2) + math.pi / 2) < 1e-12
assert abs(wrap_angle(-3 * math.pi / 2) - math.pi / 2) < 1e-12
assert abs(wrap_angle(2 * math.pi)) < 1e-12
# 179 deg to -179 deg is a 2 deg turn the short way, not 358 the long way.
short = wrap_angle(math.radians(-179) - math.radians(179))
assert abs(short - math.radians(2)) < 1e-9, short
for i in range(-2000, 2000):
    w = wrap_angle(i * 0.01)
    assert -math.pi < w <= math.pi, (i, w)
print(f"  wrap_angle: -pi -> {wrap_angle(-math.pi):+.4f}, 179deg to -179deg -> {math.degrees(short):+.1f} deg")

# --- 2. turn direction, asserted not eyeballed ---
left_wp, right_wp = Waypoint(1.0, 1.0), Waypoint(1.0, -1.0)
ctl = WaypointController()
l_l, l_r = ctl.step(Pose(0, 0, 0), left_wp)
assert l_r > l_l, f"waypoint left => right track must outrun left, got l={l_l} r={l_r}"
assert l_l == -l_r, f"45 deg off should be a spin, got l={l_l} r={l_r}"
ctl.reset()
r_l, r_r = ctl.step(Pose(0, 0, 0), right_wp)
assert r_l > r_r, f"waypoint right => left track must outrun right, got l={r_l} r={r_r}"
print(f"  spin sign: left-wp l={l_l} r={l_r} | right-wp l={r_l} r={r_r}")

# Same sign rule inside the steering regime, where the robot is also moving.
ctl.reset()
s_l, s_r = ctl.step(Pose(0.0, 0.0, -0.10), Waypoint(2.0, 0.0))    # waypoint 0.1 rad left
assert s_r > s_l, f"gentle left correction => r > l, got l={s_l} r={s_r}"
assert s_l > 0 and s_r > 0, f"a gentle correction must stay forward, got l={s_l} r={s_r}"
ctl.reset()
t_l, t_r = ctl.step(Pose(0.0, 0.0, +0.10), Waypoint(2.0, 0.0))
assert t_l > t_r and t_l > 0 and t_r > 0, f"gentle right correction wrong: l={t_l} r={t_r}"
print(f"  steer sign: left l={s_l} r={s_r} | right l={t_l} r={t_r}")

# --- 3. the stall floor scales the pair, never one track ---
assert apply_drive(100, 50) == (1200, 600), apply_drive(100, 50)
assert apply_drive(0, 0) == (0, 0)
assert apply_drive(4000, -30) == (4000, -30)
weak = ctl.step(Pose(1.94, 0.0, 0.10), Waypoint(2.0, 0.0))        # nearly there, off centre
assert weak[0] > 0 and weak[1] > 0, f"slow correction became a spin: {weak}"
assert max(abs(weak[0]), abs(weak[1])) >= DRIVE.min_move, weak
print(f"  stall floor: apply_drive(100,50)={apply_drive(100,50)} | near-goal correction={weak}")

# --- 4. convergence from 8 spread start poses ---
# Bearing, range and heading offset all differ, so these are eight distinct
# problems rather than one problem rotated eight times.
wp = Waypoint(1.2, 0.8)
starts = [(0.00, 1.10, math.pi),        # facing directly away
          (0.79, 0.60, 2.40),
          (1.57, 1.70, -1.20),
          (2.36, 0.90, 0.30),
          (3.14, 1.00, math.pi),        # directly behind it, facing away
          (3.93, 1.45, -2.80),
          (4.71, 0.55, 1.90),
          (5.50, 1.80, 0.00)]
print("  convergence:")
for bearing, radius, theta in starts:
    start = Pose(wp.x + radius * math.cos(bearing), wp.y + radius * math.sin(bearing), theta)
    error = wrap_angle(math.atan2(wp.y - start.y, wp.x - start.x) - theta)
    ctl, poses = run(start, wp)
    assert ctl.arrived, f"start ({start.x:.2f},{start.y:.2f},{theta:.2f}) did not arrive"
    assert gap(poses[-1], wp) <= wp.tolerance_m, gap(poses[-1], wp)
    print(f"    start=({start.x:+.2f},{start.y:+.2f}) theta={math.degrees(theta):+7.1f} deg  "
          f"bearing error={math.degrees(error):+7.1f} deg  range={radius:.2f} m  "
          f"steps={len(poses):4d}  gap={gap(poses[-1], wp)*100:.1f} cm")

# --- 5. no oscillation: the gap never grows once the robot is moving ---
ctl, poses = run(Pose(0.0, 0.0, math.pi), Waypoint(1.5, 0.6))
gaps = [gap(p, Waypoint(1.5, 0.6)) for p in poses]
moving = next(i for i in range(len(gaps) - 1) if gaps[i + 1] < gaps[i])   # spin holds position
for i in range(moving, len(gaps) - 1):
    assert gaps[i + 1] <= gaps[i] + 1e-12, f"gap grew at step {i}: {gaps[i]:.4f} -> {gaps[i+1]:.4f}"
for i in range(len(gaps) - 11, len(gaps) - 1):
    assert gaps[i + 1] < gaps[i], f"final approach stalled at step {i}"
print(f"  monotonic: {len(gaps)-moving} moving steps, gap {gaps[moving]:.3f} -> {gaps[-1]:.3f} m, never grew")

# --- 6. a waypoint already satisfied stops at once ---
ctl = WaypointController()
assert ctl.step(Pose(1.0, 1.0, 0.3), Waypoint(1.01, 1.0)) == (0, 0)
assert ctl.arrived, "already inside tolerance must report arrived on the first step"
ctl.reset()
assert not ctl.arrived, "reset must clear arrived"
assert ctl.step(Pose(1.0, 1.0, 0.3), Waypoint(1.0, 1.0, heading=0.35)) == (0, 0)
assert ctl.arrived, "heading inside heading_tolerance_rad must also count as arrived"
ctl.reset()
out = ctl.step(Pose(1.0, 1.0, 0.3), Waypoint(1.0, 1.0, heading=2.0))
assert out != (0, 0) and not ctl.arrived, "wrong heading is not arrival"
print(f"  satisfied waypoint: (0,0) arrived=True | wrong heading: {out} arrived=False")

# --- 7. heading-constrained waypoints reach the heading ---
print("  heading hold:")
for heading in (0.0, math.pi / 2, math.pi, -math.pi / 2, 2.6):
    goal = Waypoint(1.0, -0.7, heading=heading)
    ctl, poses = run(Pose(0.0, 0.0, -2.5), goal, max_steps=2000)
    err = wrap_angle(heading - poses[-1].theta)
    assert ctl.arrived, f"heading {heading:.2f} not reached"
    assert abs(err) <= goal.heading_tolerance_rad, f"heading off by {err:.3f} rad"
    assert gap(poses[-1], goal) <= goal.tolerance_m, "turning to heading lost the position"
    print(f"    want {math.degrees(heading):+7.1f} deg  got {math.degrees(wrap_angle(poses[-1].theta)):+7.1f} deg  "
          f"err={math.degrees(err):+5.2f} deg  gap={gap(poses[-1], goal)*100:.1f} cm")

print("\nALL MOTION CHECKS PASSED")
