"""Goal solver: four operators, three outcomes.

The whole planner is an OR over operators whose effects include the goal, and
an AND over that operator's unmet preconditions. Nothing else is needed.

Three outcomes, not two, and the third is the point:

    Solved    the goal decomposes into a plan of tasks
    Blocked   an operator matches but cannot apply, and the reason names the
              obstruction, which is work for another crab
    Unknown   nothing known achieves the goal, because the world model does
              not hold the information yet, which is work for a scout

Collapsing Unknown into Blocked would make exploration a separate mechanism
bolted on the side. Keeping it makes EXPLORE fall out of the same recursion.

Preconditions come in two flavours, and the split is what makes a HOLE report
Blocked rather than Unknown:

    subgoal   another operator can achieve it, so recurse    (reach, clear)
    guard     no operator's effects ever include it, so test it here and stop
              (capable(a, PUSH), agent count >= weight, cause is removable)

Decomposing a guard would return Unknown for every failure and bury the one
fact the caller needs: a HOLE is not removable and no amount of scouting
changes that.
"""

from dataclasses import dataclass

from world import (EXPLORED, REACHABLE, REMOVABLE, UNKNOWN as UNSEEN,
                   PUSH as CAP_PUSH)


GOTO = "GOTO"
EXPLORE = "EXPLORE"
PUSH = "PUSH"
OBSERVE = "OBSERVE"
OPERATORS = (GOTO, EXPLORE, PUSH, OBSERVE)

# routes() enumerates simple paths, so it is exponential in a dense graph. Ten
# places never need more than a handful, and the guardrail caps the graph.
MAX_ROUTES = 8

_FAR = 1 << 16


# -----------------------------------------------------------
# GOALS -- what solve() accepts
# -----------------------------------------------------------

@dataclass(frozen=True)
class Reach:
    """at(agent, pid). A bare goal accepts any crab; a bound one names it."""
    pid: str
    agent: int | None = None

    def __str__(self):
        return f"reach({self.pid})"


@dataclass(frozen=True)
class Survey:
    pid: str
    agent: int | None = None

    def __str__(self):
        return f"survey({self.pid})"


@dataclass(frozen=True)
class Clear:
    edge: str                       # edge name, not a place pair

    def __str__(self):
        return f"clear({self.edge})"


@dataclass(frozen=True)
class Observed:
    tid: str
    agent: int | None = None

    def __str__(self):
        return f"observed({self.tid})"


# -----------------------------------------------------------
# RESULTS
# -----------------------------------------------------------

@dataclass(frozen=True)
class Task:
    op: str                         # one of OPERATORS
    target: str                     # place id, edge name or thing id
    agent: int | None = None
    cause: str | None = None        # why the task exists, for a clear task
    goal: object = None             # the goal it serves, so a release can ask
                                    # whether the work still needs doing

    def __str__(self):
        who = "-" if self.agent is None else self.agent
        return f"{self.op}({who}, {self.target})"


@dataclass(frozen=True)
class Solved:
    plan: list


@dataclass(frozen=True)
class Blocked:
    reason: str
    edge: str | None = None
    cause: str | None = None


@dataclass(frozen=True)
class Unknown:
    goal: object


# -----------------------------------------------------------
# FACTS
# -----------------------------------------------------------

def holds(world, goal) -> bool:
    if isinstance(goal, Reach):
        if goal.agent is None:
            return any(a.place == goal.pid for a in world.live_agents())
        agent = world.agents.get(goal.agent)
        return agent is not None and not agent.lost and agent.place == goal.pid

    if isinstance(goal, Survey):
        place = world.place(goal.pid)
        return place is not None and place.state == EXPLORED

    if isinstance(goal, Clear):
        edge = world.edge_named(goal.edge)
        return edge is not None and edge.state == REACHABLE

    if isinstance(goal, Observed):
        return goal.tid in world.observed

    raise TypeError(f"not a goal: {goal!r}")


# -----------------------------------------------------------
# SOLVER
# -----------------------------------------------------------

def solve(world, goal, trace: list | None = None):
    """Solved(plan) | Blocked(reason) | Unknown(goal).

    Pass `trace` to collect the indented derivation. T3 prints it; the tests
    assert on it.
    """
    return _solve(world, goal, frozenset(), trace, 0)


def _solve(world, goal, seen, trace, depth):
    _say(trace, depth, str(goal))

    if holds(world, goal):
        _say(trace, depth + 1, "-> holds")
        return Solved([])

    note = _state_of(world, goal)
    if note:
        _say(trace, depth + 1, note)

    if goal in seen:
        # Clearing an edge needs a crab at the block, getting there may need
        # the same edge. Cutting the loop lets the OR node try another route.
        _say(trace, depth + 1, "-> CYCLE")
        return Blocked("cycle", None, None)
    seen = seen | {goal}

    bindings = _bindings(world, goal)
    if not bindings:
        _say(trace, depth + 1, f"-> UNKNOWN {goal}")
        return Unknown(goal)

    failures = []
    for bind in bindings:
        _say(trace, depth + 1, bind.label)

        stopped = None
        for text, ok, fail in bind.guards:
            _say(trace, depth + 2, f"requires {text} -> {'true' if ok else 'FALSE'}")
            if not ok and stopped is None:
                stopped = fail
        if stopped is not None:
            _say(trace, depth + 2, f"-> BLOCKED {stopped.reason}")
            failures.append(stopped)
            continue

        results = [_solve(world, sub, seen, trace, depth + 2) for sub in bind.subgoals]
        if all(isinstance(r, Solved) for r in results):
            plan = [t for r in results for t in r.plan]
            plan.append(Task(bind.op, bind.target, bind.agent, bind.cause))
            _say(trace, depth + 2, "-> SOLVED")
            return Solved(plan)

        failures.extend(r for r in results if not isinstance(r, Solved))

    best = _best(failures, goal)
    _say(trace, depth + 1, f"-> {type(best).__name__.upper()} "
                           f"{best.reason if isinstance(best, Blocked) else best.goal}")
    return best


def _rank(result):
    """Actionability, lowest first.

    architecture.md says `BLOCKED(best reason among results)` but also that
    UNKNOWN is what emits an EXPLORE task, so folding every failure into
    Blocked would silence exploration. Ranked instead: a removable obstruction
    is a job for another crab now, scouting is a job for a crab soon, and a
    HOLE is neither.
    """
    if isinstance(result, Blocked):
        return 0 if result.cause in REMOVABLE else 2
    return 1


def _best(failures, goal):
    if not failures:
        return Unknown(goal)
    return min(failures, key=_rank)          # min is stable: first best wins


def _state_of(world, goal):
    """The one line of world state that explains why a goal is unmet.

    architecture.md prints `e2 is BLOCKED(BLOCK)` above the operator that
    answers it, because the cause is what the reader is looking for.
    """
    if isinstance(goal, Clear):
        edge = world.edge_named(goal.edge)
        if edge is not None:
            return f"{edge.name} is {edge.state}" + (f"({edge.cause})" if edge.cause else "")
    return None


def _say(trace, depth, text):
    if trace is not None:
        trace.append("  " * depth + text)


# -----------------------------------------------------------
# OPERATORS -- effects decide which bindings a goal produces
# -----------------------------------------------------------

@dataclass
class _Bind:
    op: str
    target: str
    agent: int | None
    label: str
    guards: list                     # (text, ok, Blocked on failure)
    subgoals: list
    cause: str | None = None
    promise: tuple = (0,)


def _bindings(world, goal):
    if isinstance(goal, Survey):
        return _explore_ops(world, goal)
    if isinstance(goal, Reach):
        return _goto_ops(world, goal)
    if isinstance(goal, Clear):
        return _push_ops(world, goal)
    if isinstance(goal, Observed):
        return _observe_ops(world, goal)
    raise TypeError(f"not a goal: {goal!r}")


def _explore_ops(world, goal):
    """EXPLORE(a, p): at(a, p) -> surveyed(p), plus whatever it discovers."""
    if world.place(goal.pid) is None:
        return []                    # the place is not in the model at all
    return [_Bind(EXPLORE, goal.pid, a.serial,
                  f"EXPLORE({a.serial}, {goal.pid})",
                  [], [Reach(goal.pid, a.serial)])
            for a in _crew(world, goal.agent, goal.pid)]


def _goto_ops(world, goal):
    """GOTO(a, p): a route of REACHABLE edges -> at(a, p).

    One binding per route, including routes that are not usable yet. Filtering
    them out here would hide the obstruction that should become a task.
    """
    if world.place(goal.pid) is None:
        return []
    out = []
    for agent in _crew(world, goal.agent, goal.pid):
        for route in world.routes(agent.place, goal.pid, MAX_ROUTES):
            unmet = [e for e in world.edges_on(route) if e.state != REACHABLE]
            out.append(_Bind(GOTO, goal.pid, agent.serial,
                             f"GOTO({agent.serial}, {goal.pid}) via {'-'.join(route)}",
                             [], [Clear(e.name) for e in unmet],
                             promise=(len(unmet), len(route), agent.serial)))
    out.sort(key=lambda b: b.promise)
    return out


def _push_ops(world, goal):
    """PUSH(a, t, e): at(a, near(t)), capable(a, PUSH), agents >= weight(t)
    -> not blocked(e)."""
    edge = world.edge_named(goal.edge)
    if edge is None or edge.state == UNSEEN:
        return []                    # never looked at: scouting, not pushing

    thing = world.blocker_of(edge.name)

    def dead(text, reason):
        return [_Bind(PUSH, edge.name, None, f"PUSH(-, {edge.name})",
                      [(text, False, Blocked(reason, edge.name, edge.cause))],
                      [], cause=edge.cause)]

    # The one distinction that decides collaborate against route around.
    if edge.cause not in REMOVABLE:
        return dead(f"cause {edge.cause or edge.state} is removable",
                    f"{edge.cause or edge.state} on {edge.name}")
    if thing is None:
        return dead(f"a blocker on {edge.name} is known", f"blocker of {edge.name} unknown")

    crew = [a for a in world.live_agents() if CAP_PUSH in a.capabilities]
    if not crew:
        return dead("an agent is capable of PUSH", f"no crab can push {edge.name}")

    out = []
    for agent in _crew(world, None, thing.place, CAP_PUSH):
        out.append(_Bind(
            PUSH, edge.name, agent.serial,
            f"PUSH({agent.serial}, {thing.tid}, {edge.name})",
            [(f"capable({agent.serial}, PUSH)", True, None),
             (f"agents >= weight({thing.tid}) = {thing.weight}",
              len(crew) >= thing.weight,
              Blocked(f"{thing.tid} needs {thing.weight} crabs, {len(crew)} live",
                      edge.name, edge.cause))],
            [Reach(thing.place, agent.serial)], cause=edge.cause))
    return out


def _observe_ops(world, goal):
    """OBSERVE(a, t): t in view from at(a) -> facts about t."""
    thing = world.things.get(goal.tid)
    if thing is None:
        return []
    return [_Bind(OBSERVE, goal.tid, a.serial,
                  f"OBSERVE({a.serial}, {goal.tid})",
                  [], [Reach(thing.place, a.serial)])
            for a in _crew(world, goal.agent, thing.place)]


# -----------------------------------------------------------

def _crew(world, serial, near, cap=None):
    """Candidate agents, nearest first then lower serial, so a rerun on the
    same world state produces the same plan."""
    if serial is not None:
        agent = world.agents.get(serial)
        usable = agent is not None and not agent.lost
        return [agent] if usable and (cap is None or cap in agent.capabilities) else []
    pool = [a for a in world.live_agents() if cap is None or cap in a.capabilities]
    return sorted(pool, key=lambda a: (_hops(world, a.place, near), a.serial))


def _hops(world, a, b):
    found = world.hops(a, b)
    return _FAR if found is None else found
