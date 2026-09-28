"""Shared record of discovered blocks, with best-effort deduplication.

Honest limitation, stated up front: with no global coordinate frame we
cannot prove that two sightings are the same physical block. What we can do
is refuse to count the same block twice in the situations that actually
occur in the demo:

  a) One bot claims a block, backs off, and sees it again. Handled by the
     per-bot refractory timer in behavior.py.
  b) Both bots converge on the same block. Handled here: a second claim of
     similar apparent size, arriving inside merge_window_s, is merged into
     the first rather than registered as new.

What this does NOT handle: two genuinely different blocks of similar size
claimed by the two bots within the merge window. Those get merged and the
count reads low. Widen the arena spacing or shorten merge_window_s to trade
that against the risk of double counting.
"""

import threading
import time
from dataclasses import dataclass, field


@dataclass
class Target:
    target_id: int
    first_seen_by: str
    first_seen_at: float
    area_fraction: float
    sightings: int = 1
    seen_by: set = field(default_factory=set)
    merged: int = 0


class TargetRegistry:
    def __init__(self, cfg):
        self.cfg = cfg
        self._targets = []
        self._lock = threading.Lock()
        self._next_id = 1

    def claim(self, bot_name: str, area_fraction: float):
        """Record a bot arriving at a block.

        Returns (target, is_new).
        """
        now = time.time()

        with self._lock:
            for target in reversed(self._targets):
                if now - target.first_seen_at > self.cfg.merge_window_s:
                    break                      # list is time-ordered; older only

                if self._similar_size(target.area_fraction, area_fraction):
                    target.sightings += 1
                    target.merged += 1
                    target.seen_by.add(bot_name)
                    return target, False

            target = Target(
                target_id=self._next_id,
                first_seen_by=bot_name,
                first_seen_at=now,
                area_fraction=area_fraction,
                seen_by={bot_name},
            )

            self._next_id += 1
            self._targets.append(target)

            return target, True

    def _similar_size(self, a: float, b: float) -> bool:
        if a <= 0 or b <= 0:
            return False

        ratio = abs(a - b) / max(a, b)
        return ratio <= self.cfg.merge_size_ratio

    def all(self):
        with self._lock:
            return list(self._targets)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._targets)

    @property
    def merges(self) -> int:
        with self._lock:
            return sum(t.merged for t in self._targets)
