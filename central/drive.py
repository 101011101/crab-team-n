"""Track-pair conditioning: the one place a command becomes wire units.

Kept out of behavior.py so the waypoint controller can reuse the rule
without importing the reactive state machine.
"""

from config import DRIVE


def apply_drive(left, right):
    """Scale a track pair so the robot actually moves, without distorting the turn.

    Motors stall below roughly DRIVE.min_move, so a command weaker than that
    produces noise and no motion. The fix is NOT to clamp each track to the
    floor: during a turn one track is legitimately near zero, and forcing it
    to full reverse converts a gentle correction into a violent spin.

    Instead scale both tracks together until the stronger one clears the
    stall floor. The ratio between them -- which is the turn -- is preserved.
    """
    peak = max(abs(left), abs(right))

    if peak < 1e-6:
        return 0, 0

    if peak < DRIVE.min_move:
        scale = DRIVE.min_move / peak
        left, right = left * scale, right * scale

    return (int(max(-4095, min(4095, left))),
            int(max(-4095, min(4095, right))))
