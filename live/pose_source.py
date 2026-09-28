"""Where is each robot? Tags say, dead reckoning fills the gaps.

    PRIMARY    AprilTag fix (tag_pose.py). Metres, room frame, MEASURED.
    BETWEEN    dead reckoning from commanded duty and the motor model.

A tag fix CORRECTS the dead reckoner; between fixes the dead reckoner carries the
robot. Neither alone is enough: tags are absent whenever no tag is in view, and dead
reckoning is wrong the moment a wheel slips.

WHY NOT SfM REGISTRATION, WHICH THIS FILE USED TO PREFER
  scale      SfM has no metre in it at all. A tag has a printed 0.15 m edge.
  roll       a straight drive cannot observe roll about the direction of travel;
             docs/robot1-reconstruction.md measured 15.8 deg left over. A tag plane
             fixes all three rotations.
  features   a 320x240 robot frame carries a median 116 SIFT features against 2,340
             for the phone frames the model was built from. A tag needs 4 corners.
  frame      the SfM map is in model units with an arbitrary origin. The tag map is
             the room, in metres. See arena_map.py for why they cannot be mixed.

The motor model is not invented here. DEADBAND, DUTY_PER_M_S and TRACK_WIDTH come
from pipeline/common/constants.py, where they are the CalibPacket values the
firmware reports.
"""
import math, os, sys, time
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from pipeline.common import constants as C        # noqa: E402

STALE_FIX_S = 5.0       # a fix older than this stops being reassuring
DRIFT_FULL_M = 1.5      # dead-reckoned distance at which confidence reaches zero


def duty_to_speed(duty):
    """Commanded duty -> metres per second, per the deadband + linear model.

    Below the deadband the wheel does not turn at all: it is stiction, not a slow
    crawl. Treating duty as proportional right down to zero is the classic way to
    dead-reckon a robot that is standing still.
    """
    s = 1.0 if duty >= 0 else -1.0
    mag = abs(float(duty))
    if mag <= C.DEADBAND:
        return 0.0
    return s * (mag - C.DEADBAND) / C.DUTY_PER_M_S


def speed_to_duty(v):
    """Inverse of duty_to_speed, clamped to the drive's range."""
    if abs(v) < 1e-6:
        return 0
    s = 1.0 if v >= 0 else -1.0
    return int(max(-C.DUTY_MAX, min(C.DUTY_MAX,
                                    s * (abs(v) * C.DUTY_PER_M_S + C.DEADBAND))))


def tracks_for(v, w):
    """Body twist (v m/s, w rad/s) -> left and right duty."""
    return (speed_to_duty(v - w * C.TRACK_WIDTH / 2),
            speed_to_duty(v + w * C.TRACK_WIDTH / 2))


class DeadReckoner:
    """Commanded duty in, a room-frame pose out. Wrong, but never absent."""

    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x, self.y, self.theta = x, y, theta
        self.t = time.time()
        self.fixed_at = 0.0
        self.drift_since_fix = 0.0
        self.n_fixes = 0
        self.last_jump_m = 0.0

    def command(self, left, right, now=None):
        now = now or time.time()
        dt = min(now - self.t, 0.5)      # a long gap is a stall, not a long drive
        self.t = now
        vl, vr = duty_to_speed(left), duty_to_speed(right)
        v = (vl + vr) / 2.0
        w = (vr - vl) / C.TRACK_WIDTH
        self.theta = math.atan2(math.sin(self.theta + w * dt),
                                math.cos(self.theta + w * dt))
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.drift_since_fix += abs(v) * dt

    def correct(self, x, y, theta=None):
        """Apply a tag fix. The jump is recorded: a large one means the dead
        reckoning had drifted badly, which is worth seeing rather than hiding."""
        self.last_jump_m = math.hypot(x - self.x, y - self.y)
        self.x, self.y = x, y
        if theta is not None:
            self.theta = theta
        self.fixed_at = time.time()
        self.drift_since_fix = 0.0
        self.n_fixes += 1

    @property
    def pose(self):
        return (self.x, self.y, self.theta)

    @property
    def fixed(self):
        return self.n_fixes > 0

    def confidence(self):
        """1.0 at a fresh fix, falling with both distance travelled and time since.

        0.0 means the pose is a guess. The loop refuses to drive on a guess and the
        dashboard is told to show it, because a robot drawn confidently at a place
        nobody measured is the worst thing either can do.
        """
        if not self.fixed:
            return 0.0
        by_drift = 1.0 - self.drift_since_fix / DRIFT_FULL_M
        by_age = 1.0 - (time.time() - self.fixed_at) / (STALE_FIX_S * 3)
        return float(max(0.0, min(1.0, min(by_drift, by_age))))
