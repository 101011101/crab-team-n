"""Per-robot reactive controller.

No map, no global position. Each robot runs the same small state machine,
driven only by what its own camera sees this instant:

    SEARCH   sweep the arena: creep forward, turn, repeat
      |          green blob held steady for lock_on_s
      v
    APPROACH steer toward the blob, proportional to how far off centre
      |          blob fills claim_area_fraction of the frame
      v
    CLAIM    stop, register the target, hold still so it reads on video
      |
      v
    BACKOFF  reverse and turn away, then ignore blocks for refractory_s
      |          so the block just counted is not counted again
      v
    SEARCH

The sector setting only biases which way a robot turns while searching, so
the two robots tend to sweep away from each other rather than trail each
other. That is the reactive stand-in for assigned search areas.
"""

import random
import time

from config import BEHAVIOR, DRIVE
from drive import apply_drive


SEARCH = "SEARCH"
APPROACH = "APPROACH"
CLAIM = "CLAIM"
BACKOFF = "BACKOFF"
IDLE = "IDLE"
OFFLINE = "OFFLINE"


class ScoutBehavior:
    def __init__(self, name: str, sector: str, registry):
        self.name = name
        self.sector = sector
        self.registry = registry

        self.state = IDLE
        self.state_since = time.time()

        self.primary = None            # detection we are chasing, for the UI
        self.refractory_until = 0.0
        self.lock_on_since = None
        self.last_seen_at = 0.0

        # Set when this robot absorbs a dropped robot's work.
        self.absorbed = []

        # SEARCH sub-state: alternating creep and turn.
        self._search_phase = "forward"
        self._phase_until = 0.0
        self._claim_pending = False

        self.claims = 0
        self.last_event = ""

    # -----------------------------------------------------------

    def _enter(self, state: str):
        if state != self.state:
            self.state = state
            self.state_since = time.time()

            if state == CLAIM:
                # Arm the one-shot claim. A time window is not enough: at
                # 20 Hz several ticks fall inside any window, so one arrival
                # would register several times and inflate the merge count.
                self._claim_pending = True

    @property
    def elapsed(self) -> float:
        return time.time() - self.state_since

    @property
    def turn_sign(self) -> int:
        """+1 turns right, -1 turns left."""
        return -1 if self.sector == "left" else 1

    @property
    def sweep_scale(self) -> float:
        """A robot covering two sectors turns further and runs longer."""
        return 1.0 + 0.6 * len(self.absorbed)

    # -----------------------------------------------------------

    def set_idle(self):
        self._enter(IDLE)
        self.primary = None

    def set_offline(self):
        self._enter(OFFLINE)
        self.primary = None

    def absorb(self, other_name: str):
        if other_name not in self.absorbed:
            self.absorbed.append(other_name)
            self.last_event = f"absorbed {other_name}"

    # -----------------------------------------------------------
    # MAIN STEP
    # -----------------------------------------------------------

    def step(self, detections, running: bool, online: bool):
        """Return (left, right) track speeds for this tick."""
        now = time.time()

        if not online:
            self.set_offline()
            return 0, 0

        if not running:
            self.set_idle()
            return 0, 0

        if self.state in (IDLE, OFFLINE):
            self._enter(SEARCH)
            self._phase_until = 0.0

        suppressed = now < self.refractory_until
        candidate = detections[0] if detections else None

        if candidate is not None:
            self.last_seen_at = now

        handler = {
            SEARCH: self._do_search,
            APPROACH: self._do_approach,
            CLAIM: self._do_claim,
            BACKOFF: self._do_backoff,
        }[self.state]

        return handler(candidate, suppressed, now)

    # -----------------------------------------------------------
    # STATES
    # -----------------------------------------------------------

    def _do_search(self, candidate, suppressed, now):
        self.primary = None

        if candidate is not None and not suppressed:
            # Require the blob to persist, so one noisy frame cannot
            # yank the robot off its sweep.
            if self.lock_on_since is None:
                self.lock_on_since = now

            if now - self.lock_on_since >= BEHAVIOR.lock_on_s:
                self.lock_on_since = None
                self._enter(APPROACH)
                return self._do_approach(candidate, suppressed, now)
        else:
            self.lock_on_since = None

        # Alternate creeping forward and turning in this robot's sector
        # direction, so the two robots fan apart instead of converging.
        if now >= self._phase_until:
            if self._search_phase == "forward":
                self._search_phase = "spin"
                jitter = random.uniform(0.8, 1.4)
                self._phase_until = now + BEHAVIOR.search_spin_s * jitter * self.sweep_scale
            else:
                self._search_phase = "forward"
                self._phase_until = now + BEHAVIOR.search_forward_s * self.sweep_scale

        if self._search_phase == "forward":
            speed = DRIVE.search_forward
            return apply_drive(speed, speed)

        turn = DRIVE.spin * self.turn_sign
        return apply_drive(turn, -turn)

    def _do_approach(self, candidate, suppressed, now):
        if candidate is None:
            # Brief dropouts are normal; only give up after a grace period.
            if now - self.last_seen_at > 1.0:
                self.primary = None
                self._enter(SEARCH)
                self._phase_until = 0.0
                self.last_event = "lost target"
                return 0, 0

            creep = DRIVE.search_forward // 2
            return apply_drive(creep, creep)

        self.primary = candidate

        if candidate.area_fraction >= self._claim_threshold():
            self._enter(CLAIM)
            return self._do_claim(candidate, suppressed, now)

        if self.elapsed > BEHAVIOR.approach_timeout_s:
            # Something is wrong: wedged on an obstacle, or chasing a green
            # object it can never reach. Give up rather than push forever.
            self._enter(BACKOFF)
            self.last_event = "approach timeout"
            return 0, 0

        steer = candidate.bearing * DRIVE.steer_gain

        # Slow down as the target drifts off centre, so a sharp correction
        # is a turn in place rather than a wide arc past the block.
        centred = max(0.0, 1.0 - abs(candidate.bearing))
        forward = DRIVE.cruise * (0.45 + 0.55 * centred)

        return apply_drive(forward + steer, forward - steer)

    def _claim_threshold(self):
        from config import VISION
        return VISION.claim_area_fraction

    def _do_claim(self, candidate, suppressed, now):
        if self._claim_pending:
            self._claim_pending = False
            area = candidate.area_fraction if candidate else self._claim_threshold()
            target, is_new = self.registry.claim(self.name, area)

            self.claims += 1
            self.last_event = (f"FOUND target #{target.target_id}" if is_new
                               else f"merged into #{target.target_id}")

        if self.elapsed >= BEHAVIOR.claim_hold_s:
            self._enter(BACKOFF)

        return 0, 0

    def _do_backoff(self, candidate, suppressed, now):
        self.primary = None

        if self.elapsed < BEHAVIOR.backoff_reverse_s:
            speed = -DRIVE.search_forward
            return apply_drive(speed, speed)

        turn_done = BEHAVIOR.backoff_reverse_s + BEHAVIOR.backoff_turn_s

        if self.elapsed < turn_done:
            turn = DRIVE.spin * self.turn_sign
            return apply_drive(turn, -turn)

        self.refractory_until = now + BEHAVIOR.refractory_s
        self._enter(SEARCH)
        self._phase_until = 0.0

        return 0, 0
