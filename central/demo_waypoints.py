"""Drive a simulated robot through waypoints and log every command sent.

    ./.venv/bin/python demo_waypoints.py

Proves the chain the planner depends on:

    waypoint  ->  motion controller  ->  track speeds  ->  robot moves

Every line in the log is a command that would have gone to the ESP32 over
the wire, unchanged. Nothing here knows about vision or the network.
"""

import argparse
import math

from geometry import Pose, Waypoint
from motion import WaypointController, wrap_angle
from sim import SimRobot, two_room_arena


DT = 0.05                  # 20 Hz, the real control rate
LIMIT_S = 40.0             # give up on a waypoint after this long


def run_route(route, start, label, arena=None, verbose=True, log_every=10):
    """Drive `route` in order. Return (reached, log)."""
    robot = SimRobot(1, start)
    controller = WaypointController()

    log = []
    reached = []
    now = 0.0

    if verbose:
        print(f"\n{'=' * 74}")
        print(f"  {label}")
        print(f"{'=' * 74}")
        print(f"  start  ({start.x:.2f}, {start.y:.2f})  "
              f"facing {math.degrees(start.theta):+.0f} deg")

    for index, wp in enumerate(route, 1):
        controller.reset()

        began = now
        ticks = 0
        heading_txt = ("any" if wp.heading is None
                       else f"{math.degrees(wp.heading):+.0f} deg")

        if verbose:
            print(f"\n  waypoint {index}: ({wp.x:.2f}, {wp.y:.2f})  "
                  f"heading {heading_txt}")
            print(f"    {'t':>6}  {'pose':>22}  {'range':>7}  "
                  f"{'bearing':>8}  {'left':>6} {'right':>6}")

        while now - began < LIMIT_S:
            left, right = controller.step(robot.pose, wp)

            if controller.arrived:
                break

            if controller.stalled:
                if verbose:
                    print(f"    {now:6.2f}  STALLED -- wedged, not moving. "
                          f"Reporting failure instead of pushing.")
                break

            robot.command(left, right, now)

            before = robot.pose
            robot.step(DT, now)

            # A real robot stalls against a wall. Undo the move so the
            # controller has to cope with not making progress, exactly as
            # it would on the bench.
            if arena is not None and arena.blocked_at(robot.pose.x, robot.pose.y):
                robot.pose = before

            dx, dy = wp.x - robot.pose.x, wp.y - robot.pose.y
            rng = math.hypot(dx, dy)
            bearing = wrap_angle(math.atan2(dy, dx) - robot.pose.theta)

            entry = {
                "t": now, "waypoint": index,
                "x": robot.pose.x, "y": robot.pose.y, "theta": robot.pose.theta,
                "range": rng, "bearing": bearing,
                "left": left, "right": right,
            }
            log.append(entry)

            if verbose and ticks % log_every == 0:
                print(f"    {now:6.2f}  "
                      f"({robot.pose.x:+.2f},{robot.pose.y:+.2f},"
                      f"{math.degrees(robot.pose.theta):+7.1f}d)  "
                      f"{rng:6.3f}m  {math.degrees(bearing):+7.1f}d  "
                      f"{left:6d} {right:6d}")

            now += DT
            ticks += 1

        dx, dy = wp.x - robot.pose.x, wp.y - robot.pose.y
        gap = math.hypot(dx, dy)
        ok = controller.arrived

        reached.append({"waypoint": index, "arrived": ok, "gap_m": gap,
                        "seconds": now - began, "stalled": controller.stalled})

        if verbose:
            mark = ("reached" if ok
                    else "STALLED" if controller.stalled else "GAVE UP")
            line = (f"    -> {mark} in {now - began:5.2f}s, "
                    f"{gap * 100:.1f} cm from target")

            if wp.heading is not None:
                err = math.degrees(wrap_angle(wp.heading - robot.pose.theta))
                line += f", heading off by {err:+.1f} deg"

            print(line)

        # Stopping between waypoints is the honest test: the planner
        # issues them one at a time, so the controller never gets to
        # carry momentum from the last one.
        robot.command(0, 0, now)

        for _ in range(4):
            robot.step(DT, now)
            now += DT

    return reached, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    results = []

    # 1. A square. Every turn is 90 degrees, so a sign error in the
    #    steering shows up immediately as a robot that spirals.
    square = [Waypoint(1.0, 0.0), Waypoint(1.0, 1.0),
              Waypoint(0.0, 1.0), Waypoint(0.0, 0.0)]
    results.append(("square patrol",
                    run_route(square, Pose(0, 0, 0), "1. SQUARE PATROL",
                              verbose=verbose)))

    # 2. Start facing away from the first waypoint. The controller must
    #    turn in place rather than arc backwards into it.
    away = [Waypoint(1.5, 0.0), Waypoint(0.0, 0.0, heading=math.pi / 2)]
    results.append(("facing away",
                    run_route(away, Pose(0, 0, math.pi), "2. STARTING BACKWARDS",
                              verbose=verbose)))

    # 3. The real route: cross room 1, enter the corridor, stop at the
    #    block. Walls are live, so a controller that cuts the corner
    #    wedges itself in the doorway.
    arena = two_room_arena()
    corridor = [Waypoint(1.60, 1.00), Waypoint(2.20, 1.00),
                Waypoint(2.30, 1.00, heading=0.0)]
    results.append(("corridor approach",
                    run_route(corridor, Pose(0.5, 0.5, 0.0),
                              "3. CORRIDOR APPROACH (walls live)",
                              arena=arena, verbose=verbose)))

    # 4. Aim at a waypoint on the far side of a wall. It can never be
    #    reached, so the run is correct only if the robot gives up.
    wedge = [Waypoint(2.60, 1.90)]
    results.append(("wall wedge (must stall)",
                    run_route(wedge, Pose(2.20, 1.00, 0.0),
                              "4. UNREACHABLE WAYPOINT (must report a stall)",
                              arena=two_room_arena(), verbose=verbose)))

    print(f"\n{'=' * 74}")
    print("  SUMMARY")
    print(f"{'=' * 74}")

    total = failed = 0

    for label, (reached, log) in results:
        must_stall = "must stall" in label

        if must_stall:
            ok = sum(1 for r in reached if r["stalled"] and not r["arrived"])
        else:
            ok = sum(1 for r in reached if r["arrived"])

        total += len(reached)
        failed += len(reached) - ok

        worst = max((r["gap_m"] for r in reached), default=0.0)
        seconds = sum(r["seconds"] for r in reached)

        verb = "stalled" if must_stall else "reached"

        print(f"  {label:24s}  {ok}/{len(reached)} {verb}  "
              f"worst gap {worst * 100:5.1f} cm  "
              f"{seconds:5.1f}s  {len(log):4d} commands")

    print()
    print(f"  {total - failed}/{total} scenarios behaved correctly")

    if failed:
        print("  FAILURES PRESENT")
        return 1

    print("  ALL WAYPOINTS REACHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
