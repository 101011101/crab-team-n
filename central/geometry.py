"""Geometry types shared by every laptop-side module.

Field names and meanings are an interface: the simulator and the planner
both read them. Do not rename, do not add meaning.

Named geometry.py, not types.py: a module named `types` in this directory
shadows the standard library module of that name for every import in the
package, which breaks imports that have nothing to do with us.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pose:
    x: float          # metres, world frame
    y: float
    theta: float      # radians, 0 points along +x, CCW positive


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    heading: float | None = None   # None means arrive at any heading
    tolerance_m: float = 0.05
    heading_tolerance_rad: float = 0.15
