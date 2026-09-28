"""Waypoint motion controller.

The robot is deliberately dumb: it accepts two track speeds and nothing
else. This is the layer that turns "be at (x, y), optionally facing theta"
into those two numbers, using a pose the laptop computed from markers.

Two regimes, because one is not enough:

    bearing error large   ->  turn in place, no forward speed
    bearing error small   ->  drive forward with proportional steering

Pure proportional steering arcs toward the waypoint. An arc that starts
pointing the wrong way sweeps past the waypoint before it lines up, then
arcs back: the robot circles instead of arriving.

Nothing here touches the network. Feed it a Pose, send the result to
botlink.
"""

import math

from config import DRIVE, MOTION
from drive import apply_drive
from geometry import Pose, Waypoint


FULL_SCALE = 4095.0


def wrap_angle(a: float) -> float:
    """Fold an angle into (-pi, pi].

    A robot at +179 deg told to face -179 deg must turn 2 deg, not 358.
    Every heading comparison in this file goes through here.
    """
    a = math.fmod(a, 2.0 * math.pi)

    if a > math.pi:
        a -= 2.0 * math.pi
    elif a <= -math.pi:
        a += 2.0 * math.pi

    return a


class WaypointController:
    """Drives one robot to one waypoint. One controller per robot.

    The only state is the turn/drive latch and the arrival flag, so a
    caller may switch waypoints between steps; reset() clears both.
    """

    def __init__(self, motion=MOTION, drive=DRIVE):
        self.motion = motion
        self.drive = drive
        self.reset()

    def reset(self) -> None:
        self._arrived = False
        self._turning = False
        self._last_pose = None
        self._stalled_ticks = 0

    @property
    def arrived(self) -> bool:
        return self._arrived

    @property
    def stalled(self) -> bool:
        """True when the robot is being commanded to move but is not moving.

        Without this a robot wedged against a wall or an immovable block
        pushes forever: it never arrives, so the task never completes, and
        nothing upstream learns the attempt failed. The planner needs that
        signal -- it is the "execution failed" release condition -- and the
        motors need it more.
        """
        return self._stalled_ticks >= self.motion.stall_ticks

    def _watch_for_stall(self, pose, left, right):
        commanded = left != 0 or right != 0
        moved = self.motion.stall_move_m + 1.0

        if self._last_pose is not None:
            moved = math.hypot(pose.x - self._last_pose.x,
                               pose.y - self._last_pose.y)
            moved += (self.motion.track_width_m / 2.0) * abs(
                wrap_angle(pose.theta - self._last_pose.theta))

        self._last_pose = pose

        if commanded and moved < self.motion.stall_move_m:
            self._stalled_ticks += 1
        else:
            self._stalled_ticks = 0

    # -----------------------------------------------------------

    def step(self, pose: Pose, wp: Waypoint) -> tuple[int, int]:
        """Return (left, right) track speeds, -4095..4095."""
        left, right = self._step(pose, wp)
        self._watch_for_stall(pose, left, right)
        return left, right

    def _step(self, pose: Pose, wp: Waypoint) -> tuple[int, int]:
        dx, dy = wp.x - pose.x, wp.y - pose.y
        distance = math.hypot(dx, dy)

        if distance > wp.tolerance_m:
            self._arrived = False
            return self._go(pose, dx, dy, distance)

        if wp.heading is None:
            self._arrived = True
            self._turning = False
            return 0, 0

        # Inside the position tolerance, so only the heading is left. A spin
        # commands v = 0, which keeps the position we already have.
        error = wrap_angle(wp.heading - pose.theta)

        if abs(error) <= wp.heading_tolerance_rad:
            self._arrived = True
            self._turning = False
            return 0, 0

        self._arrived = False
        return self._tracks(0.0, self._turn_rate(error))

    # -----------------------------------------------------------

    def _go(self, pose, dx, dy, distance):
        error = wrap_angle(math.atan2(dy, dx) - pose.theta)

        # Hysteresis. With a single threshold the robot sits on it, spinning
        # and creeping alternately, and drifts off course while it dithers.
        if self._turning:
            self._turning = abs(error) > self.motion.resume_drive_rad
        else:
            self._turning = abs(error) > self.motion.turn_in_place_rad

        if self._turning:
            return self._tracks(0.0, self._turn_rate(error))

        # Ease off as the waypoint closes, so the last step lands inside the
        # tolerance instead of through it.
        cruise = self.drive.cruise / FULL_SCALE * self.motion.max_speed_mps
        v = min(self.motion.approach_gain * distance, cruise)

        return self._tracks(v, self._turn_rate(error))

    def _turn_rate(self, error):
        """Turn rate for a bearing error, in the wire's sense.

        theta is CCW positive, but l = v + k*w drives the LEFT track harder
        for positive w, which is a clockwise turn. So a CCW correction is a
        negative w. Reversed, the robot flees every waypoint it is given.
        """
        w = -self.motion.turn_gain * error
        return max(-self.motion.max_turn_rate, min(self.motion.max_turn_rate, w))

    def _tracks(self, v, w):
        """(v m/s, w rad/s clockwise) -> conditioned track speeds."""
        k = self.motion.k
        units = FULL_SCALE / self.motion.max_speed_mps

        # apply_drive scales the pair, never each track alone: mid-turn one
        # track is legitimately near zero.
        return apply_drive((v + k * w) * units, (v - k * w) * units)
