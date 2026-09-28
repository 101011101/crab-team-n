"""World model: places, edges, things and agents, plus the facts that write them.

The model is a cache, not a truth source. Perception writes it through
`observe`; the planner only reads. One writer per layer means no timestamp
arbitration is needed, so `observe` never has to decide which of two
conflicting reports wins.

Two path queries exist because the planner needs both:

    path()    REACHABLE edges only. Where a robot can drive today.
    routes()  every simple route whatever the edge state. What the solver
              reasons over: an obstructed route is what becomes a clear
              task, and a second route is what lets the solver step around
              a cause that nothing can remove.
"""

from dataclasses import dataclass, field


# PLACE states.
UNKNOWN = "UNKNOWN"
FRONTIER = "FRONTIER"
EXPLORED = "EXPLORED"

# EDGE states. UNKNOWN is shared with PLACE on purpose: both mean "not looked at".
REACHABLE = "REACHABLE"
BLOCKED = "BLOCKED"
IMPASSABLE = "IMPASSABLE"

# Causes of BLOCKED. BLOCK is removable, the rest are not, and that single
# distinction is what decides collaborate against route around.
BLOCK = "BLOCK"
WALL = "WALL"
HOLE = "HOLE"
UNSTABLE = "UNSTABLE"
REMOVABLE = (BLOCK,)

# Capabilities.
PUSH = "PUSH"
DIG = "DIG"


# -----------------------------------------------------------
# ENTITIES
# -----------------------------------------------------------

@dataclass
class Place:
    pid: str
    state: str = UNKNOWN
    found_by: int | None = None


@dataclass
class Edge:
    a: str
    b: str
    name: str
    state: str = UNKNOWN
    cause: str | None = None

    def other(self, pid: str) -> str:
        return self.b if pid == self.a else self.a


@dataclass
class Thing:
    tid: str
    place: str
    cls: str = "block"
    weight: int = 1
    blocks: str | None = None      # name of the edge this thing obstructs


@dataclass
class Agent:
    serial: int
    place: str
    capabilities: frozenset = frozenset({PUSH})

    # Latched assignment. Written by the allocator, never by the solver.
    task: object = None
    task_started: float = 0.0
    timeout_s: float = 60.0

    # Release flags raised by whatever executes the task.
    failed: bool = False
    lost: bool = False

    # Tasks this crab has already failed, against the world conditions that
    # defeated them. Keeps the allocator from handing back an attempt that
    # cannot yet succeed. Written by the allocator only.
    blocked_by: dict = field(default_factory=dict)


# -----------------------------------------------------------
# FACTS -- the only things `observe` accepts
# -----------------------------------------------------------

@dataclass(frozen=True)
class PlaceSeen:
    pid: str
    state: str = FRONTIER
    found_by: int | None = None


@dataclass(frozen=True)
class EdgeSeen:
    a: str
    b: str
    name: str | None = None
    state: str = UNKNOWN
    cause: str | None = None


@dataclass(frozen=True)
class ThingSeen:
    tid: str
    place: str
    cls: str = "block"
    weight: int = 1
    blocks: str | None = None


@dataclass(frozen=True)
class AgentAt:
    serial: int
    place: str


@dataclass(frozen=True)
class Surveyed:
    pid: str


@dataclass(frozen=True)
class ThingObserved:
    tid: str


def _key(a: str, b: str):
    return (a, b) if a <= b else (b, a)


class World:
    def __init__(self, root: str | None = None):
        self.root = root
        self.places: dict[str, Place] = {}
        self.edges: dict[tuple, Edge] = {}
        self.things: dict[str, Thing] = {}
        self.agents: dict[int, Agent] = {}
        self.observed: set[str] = set()
        self._by_name: dict[str, Edge] = {}

    # -----------------------------------------------------------
    # WRITE
    # -----------------------------------------------------------

    def observe(self, fact) -> None:
        """Sole writer. Perception hands facts in; the planner only reads."""
        if isinstance(fact, PlaceSeen):
            place = self.places.get(fact.pid)
            if place is None:
                place = self.places[fact.pid] = Place(fact.pid)
                if self.root is None:
                    self.root = fact.pid
            # A later sighting must not undo a completed survey: a robot
            # driving back through an explored room re-reports it as merely
            # seen, and that would put the room back in the pool forever.
            if not (place.state == EXPLORED and fact.state == FRONTIER):
                place.state = fact.state
            if place.found_by is None:
                place.found_by = fact.found_by

        elif isinstance(fact, EdgeSeen):
            for pid in (fact.a, fact.b):
                if pid not in self.places:
                    self.observe(PlaceSeen(pid, UNKNOWN))
            key = _key(fact.a, fact.b)
            edge = self.edges.get(key)
            if edge is None:
                name = fact.name or f"{key[0]}-{key[1]}"

                # Edge names address a specific edge in a task, so a reused
                # name silently re-points every lookup at the newer edge and
                # the older one becomes unaddressable. Fail loudly instead.
                clash = self._by_name.get(name)
                if clash is not None:
                    raise ValueError(
                        f"edge name {name!r} already used by "
                        f"{clash.a}-{clash.b}, cannot reuse for "
                        f"{key[0]}-{key[1]}")

                edge = self.edges[key] = Edge(key[0], key[1], name)
                self._by_name[name] = edge
            edge.state = fact.state
            edge.cause = fact.cause if fact.state == BLOCKED else None

        elif isinstance(fact, ThingSeen):
            self.things[fact.tid] = Thing(fact.tid, fact.place, fact.cls,
                                          fact.weight, fact.blocks)

        elif isinstance(fact, AgentAt):
            self.agents[fact.serial].place = fact.place

        elif isinstance(fact, Surveyed):
            self.observe(PlaceSeen(fact.pid, EXPLORED))

        elif isinstance(fact, ThingObserved):
            self.observed.add(fact.tid)

        else:
            raise TypeError(f"not a fact: {fact!r}")

    def add_agent(self, agent: Agent) -> Agent:
        self.agents[agent.serial] = agent
        return agent

    # -----------------------------------------------------------
    # READ
    # -----------------------------------------------------------

    def place(self, pid: str) -> Place | None:
        return self.places.get(pid)

    def edge(self, a: str, b: str) -> Edge | None:
        return self.edges.get(_key(a, b))

    def edge_named(self, name: str) -> Edge | None:
        return self._by_name.get(name)

    def neighbours(self, pid: str) -> list[str]:
        out = [e.other(pid) for e in self.edges.values() if pid in (e.a, e.b)]
        return sorted(out)                 # sorted so every query is repeatable

    def edges_on(self, route: list[str]) -> list[Edge]:
        return [self.edge(route[i], route[i + 1]) for i in range(len(route) - 1)]

    def path(self, a: str, b: str) -> list[str] | None:
        """Shortest route over REACHABLE edges only. None when there is none."""
        return self._bfs(a, b, lambda e: e.state == REACHABLE)

    def hops(self, a: str, b: str) -> int | None:
        """Distance over known edges whatever their state. None when disconnected."""
        found = self._bfs(a, b, lambda e: True)
        return None if found is None else len(found) - 1

    def depth(self, pid: str) -> int:
        """Hops from the root place. This is the `deepest branch` the allocator ranks."""
        if self.root is None:
            return 0
        found = self.hops(self.root, pid)
        return 0 if found is None else found

    def routes(self, a: str, b: str, limit: int = 16) -> list[list[str]]:
        """Every simple route ignoring edge state, shortest first.

        The solver needs the routes it cannot use yet. Filtering by state here
        would hide exactly the obstruction that should become a task.
        """
        found: list[list[str]] = []

        def walk(cur, route, seen):
            if len(found) >= limit:
                return
            if cur == b:
                found.append(route)
                return
            for nxt in self.neighbours(cur):
                if nxt not in seen:
                    walk(nxt, route + [nxt], seen | {nxt})

        if a in self.places and b in self.places:
            walk(a, [a], {a})

        found.sort(key=lambda r: (len(r), r))
        return found

    def open_branches(self) -> list[str]:
        return sorted(p.pid for p in self.places.values() if p.state == FRONTIER)

    def finder(self, pid: str) -> int | None:
        place = self.places.get(pid)
        return None if place is None else place.found_by

    def blocker_of(self, edge_name: str) -> Thing | None:
        for tid in sorted(self.things):
            if self.things[tid].blocks == edge_name:
                return self.things[tid]
        return None

    def live_agents(self) -> list[Agent]:
        return [self.agents[s] for s in sorted(self.agents) if not self.agents[s].lost]

    # -----------------------------------------------------------

    def _bfs(self, a, b, usable):
        if a not in self.places or b not in self.places:
            return None
        if a == b:
            return [a]
        came = {a: None}
        queue = [a]
        while queue:
            cur = queue.pop(0)
            for nxt in self.neighbours(cur):
                if nxt in came or not usable(self.edge(cur, nxt)):
                    continue
                came[nxt] = cur
                if nxt == b:
                    route = [b]
                    while came[route[-1]] is not None:
                        route.append(came[route[-1]])
                    return list(reversed(route))
                queue.append(nxt)
        return None
