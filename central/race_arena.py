"""Two ways to the reward: one blocked, one unknown.

    ROOM A            CORRIDOR N  (blocked)        ROOM B
    +--------+  +--------------XX-------------+  +--------+
    |        +--+                             +--+        |
    |  bots  |                                   | REWARD |
    |        +--+                             +--+        |
    +--------+  +-----------------------------+  +--------+
                 CORRIDOR S  (unknown at start)

This arena exists to make the solver's OR node visible. reach(B) has two
operators that both achieve it, so when the north route turns out to be
blocked the planner does not simply queue a push -- it has a second,
unexplored option, and the allocator hands one crab each.

Whichever opens first satisfies the goal, and the other crab's task is
released as invalidated. Nothing schedules that race; it falls out of two
routes existing and two crabs being free.
"""

from sim import Arena
from navigator import ArenaMap

# Rooms are 2 m wide, corridors 0.6 m -- more than three chassis widths, so
# the gap itself is never the thing that defeats a crab.
PLACES = {"A": (1.00, 1.30), "CN": (3.50, 1.80),
          "CS": (3.50, 0.80), "B": (6.00, 1.30)}

EDGE_XY = {"e_nb": (4.60, 1.80),          # north corridor mouth
           "e_sb": (4.60, 0.80)}          # south, only blocked with --seal-south

_LEGS = {
    ("A", "CN"):  [(1.70, 1.80), (2.40, 1.80)],
    ("A", "CS"):  [(1.70, 0.80), (2.40, 0.80)],
    ("CN", "A"):  [(2.40, 1.80), (1.70, 1.80)],
    ("CS", "A"):  [(2.40, 0.80), (1.70, 0.80)],
    ("CN", "B"):  [(4.80, 1.80), (5.40, 1.80)],
    ("CS", "B"):  [(4.80, 0.80), (5.40, 0.80)],
    ("B", "CN"):  [(5.40, 1.80), (4.80, 1.80)],
    ("B", "CS"):  [(5.40, 0.80), (4.80, 0.80)],
    ("A", "B"):   [(1.70, 0.80), (2.40, 0.80), (4.80, 0.80), (5.40, 0.80)],
    ("CN", "CS"): [(2.40, 1.80), (1.70, 1.30), (1.70, 0.80), (2.40, 0.80)],
    ("CS", "CN"): [(2.40, 0.80), (1.70, 1.30), (1.70, 1.80), (2.40, 1.80)],
    # Approaching the block means entering the north corridor first.
    ("A", "e_nb"):  [(1.70, 1.80), (2.40, 1.80)],
    ("CN", "e_nb"): [],
    ("CS", "e_nb"): [(2.40, 0.80), (1.70, 1.30), (1.70, 1.80), (2.40, 1.80)],
    ("B", "e_nb"):  [(5.40, 1.80), (4.80, 1.80)],
    ("A", "e_sb"):  [(1.70, 0.80), (2.40, 0.80)],
    ("CS", "e_sb"): [],
    ("CN", "e_sb"): [(2.40, 1.80), (1.70, 1.30), (1.70, 0.80), (2.40, 0.80)],
    ("B", "e_sb"):  [(5.40, 0.80), (4.80, 0.80)],
}

# The blocker is a 0.30 x 0.45 box, so its half-width is 0.15; add the
# 0.09 chassis radius for the closest a robot centre can sit to it.
ARENA_MAP = ArenaMap(PLACES, EDGE_XY, _LEGS,
                     sweep=[(-0.55, 0.0), (0.55, 0.0), (0.55, 0.45), (-0.55, 0.45)],
                     push_standoff=0.15 + 0.09)


def build():
    """Walls, the block, and the reward."""
    a = Arena()
    W = a.add_wall

    # Room A
    W(0, 0, 2, 0); W(0, 2.6, 2, 2.6); W(0, 0, 0, 2.6)
    W(2, 0, 2, 0.5); W(2, 1.1, 2, 1.5); W(2, 2.1, 2, 2.6)

    # Corridors, north and south
    W(2, 1.5, 5, 1.5); W(2, 2.1, 5, 2.1)
    W(2, 0.5, 5, 0.5); W(2, 1.1, 5, 1.1)

    # Room B
    W(5, 0, 7, 0); W(5, 2.6, 7, 2.6); W(7, 0, 7, 2.6)
    W(5, 0, 5, 0.5); W(5, 1.1, 5, 1.5); W(5, 2.1, 5, 2.6)

    # A box, not a circle. A circle big enough to seal a 0.60 m corridor
    # would be wider than the corridor itself and could not physically be in
    # it; a 0.30 x 0.45 m box seals the same gap and is a cardboard carton a
    # small robot can actually shove.
    a.add_box(4.60, 1.80, 0.30, 0.45)

    # One reward, deep in room B, reachable by either route.
    a.add_species(6.30, 1.30, "gold")

    return a
