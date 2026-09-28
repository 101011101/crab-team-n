"""Task allocation: latched, deterministic, no preemption.

The rules, in order, and they are applied in this order because each one
removes work the next one would otherwise fight over:

    1. Discovery implies ownership. A branch is found by a crab standing
       there, so that crab is the nearest crab by construction. It continues
       onto the best fork it found.
    2. Forks the finder did not take enter the pool.
    3. A free crab drains the pool, deepest branch first, ties to the lower
       serial.
    4. A latched crab holds its task until a release fires.

Why no preemption: a nearer branch cannot appear without a crab present to
find it, and rule 1 gives that crab the branch immediately. No other crab is
better placed, so re-ranking a latched crab can only make the assignment
worse and it can thrash. Nothing here re-ranks.

Serial order is the only tie break, which is what makes two runs of the demo
produce the same assignment.
"""

import time
from dataclasses import replace

from solver import (EXPLORE, GOTO, OBSERVE, PUSH, Blocked, Solved, Survey,
                    Task, Unknown, holds, solve)
from world import EXPLORED, REACHABLE, REMOVABLE


RESOLVED = "resolved"
INVALIDATED = "invalidated"
FAILED = "execution failed"
TIMED_OUT = "timed out"
LOST = "robot lost"


def _conditions(world, task) -> tuple:
    """What about the world would have to change for a failed task to be
    worth retrying.

    A task fails because some condition did not hold. Reissuing it while
    every one of those conditions is unchanged just fails again, which is a
    livelock: the crab is released for execution failure and handed the
    identical task on the next tick, forever.

    Crab count is in here because a two-crab block is the collaboration
    case: one crab alone cannot move it, but the attempt becomes worth
    repeating the moment another crab is available.
    """
    edge = world.edge_named(task.target) if task.op == PUSH else None

    return (
        task.op,
        task.target,
        len(world.live_agents()),
        None if edge is None else edge.state,
        None if edge is None else edge.cause,
    )


def assign(world, agents, now: float | None = None, log: list | None = None) -> dict:
    """Return {serial: Task} and latch it onto each agent.

    The allocator is the only writer of `agent.task`.
    """
    now = time.time() if now is None else now
    out = {}
    free = []

    # Rule 4 first: a latched crab is not a candidate for anything else.
    for agent in sorted(agents, key=lambda a: a.serial):
        if agent.task is not None:
            why = release_reason(world, agent, now)
            if why is None:
                out[agent.serial] = agent.task
                continue
            _note(log, f"release {agent.serial}: {agent.task} -- {why}")

            if why == FAILED:
                # Remember the shape of the world that defeated it, so the
                # same attempt is not handed straight back.
                agent.blocked_by[str(agent.task)] = _conditions(world, agent.task)

            agent.task = None
        if agent.lost:
            continue
        free.append(agent)

    claimed = {t.target for t in out.values() if t.op in (EXPLORE, GOTO)}

    # Rule 1: the finder continues onto the best fork it found.
    for agent in list(free):
        mine = [pid for pid in world.open_branches()
                if pid not in claimed and world.finder(pid) == agent.serial]
        if not mine:
            continue
        pid = best_fork(world, mine)
        task = _task_for(world, agent, pid)
        if task is None or not _worth_retrying(world, agent, task):
            continue
        claimed.add(pid)
        free.remove(agent)
        out[agent.serial] = _latch(agent, task, now, log, f"owns {pid}")

    # Rules 2 and 3: everything unclaimed is the pool; deepest first.
    for pid in sorted(world.open_branches(), key=lambda p: (-world.depth(p), p)):
        if not free:
            break
        if pid in claimed:
            continue
        for agent in list(free):          # free is in serial order: lower wins
            task = _task_for(world, agent, pid)
            if task is None or not _worth_retrying(world, agent, task):
                continue
            claimed.add(pid)
            free.remove(agent)
            out[agent.serial] = _latch(agent, task, now, log, f"pool {pid} d{world.depth(pid)}")
            break

    return out


def best_fork(world, pids: list[str]) -> str:
    """Deepest, then lowest place id.

    architecture.md fixes `best` only for the pool. The finder uses the same
    order so one rule covers both and the demo cannot show two orderings.
    """
    return min(pids, key=lambda p: (-world.depth(p), p))


# -----------------------------------------------------------
# RELEASE
# -----------------------------------------------------------

def release_reason(world, agent, now: float) -> str | None:
    """None means hold. Anything else frees the crab this tick."""
    if agent.lost:
        return LOST
    if agent.failed:
        return FAILED
    task = agent.task
    if task is None:
        return None
    if now - agent.task_started > agent.timeout_s:
        return TIMED_OUT
    if _done(world, agent, task):
        return RESOLVED
    if _stale(world, task):
        return INVALIDATED
    return None


def _done(world, agent, task) -> bool:
    # The branch may have been surveyed by whoever got there first, which
    # retires this task whatever step of the plan it was.
    if task.goal is not None and holds(world, task.goal):
        return True
    if task.op == EXPLORE:
        place = world.place(task.target)
        return place is not None and place.state == EXPLORED
    if task.op == GOTO:
        return agent.place == task.target
    if task.op == PUSH:
        edge = world.edge_named(task.target)
        return edge is not None and edge.state == REACHABLE
    if task.op == OBSERVE:
        return task.target in world.observed
    return False


def _stale(world, task) -> bool:
    """The task still names something, but the world no longer supports it."""
    if task.op in (EXPLORE, GOTO):
        return world.place(task.target) is None
    if task.op == PUSH:
        edge = world.edge_named(task.target)
        # A cause that turned out not to be removable is not a push any more,
        # and the parent goal must go back through the solver for a route.
        return edge is None or edge.cause not in REMOVABLE
    if task.op == OBSERVE:
        return task.target not in world.things
    return False


# -----------------------------------------------------------

def _task_for(world, agent, pid):
    """The one task this crab starts now for branch `pid`, or None.

    This is where the three outcomes become work: Solved hands over the first
    step of the plan, Unknown becomes an EXPLORE, and Blocked becomes a clear
    task only when the cause can actually be removed.
    """
    result = solve(world, Survey(pid, agent.serial))

    branch = Survey(pid)                  # agent free: any crab surveying it retires the task

    if isinstance(result, Solved):
        if not result.plan:
            return None                       # already surveyed
        first = result.plan[0]
        # A plan that opens with another crab's step is not this crab's to
        # start. Leave the branch pooled rather than send it somewhere the
        # preconditions do not hold yet.
        return replace(first, goal=branch) if first.agent == agent.serial else None

    if isinstance(result, Unknown):
        return Task(EXPLORE, pid, agent.serial, None, branch)

    if isinstance(result, Blocked) and result.cause in REMOVABLE and result.edge:
        return Task(PUSH, result.edge, agent.serial, result.cause, branch)

    return None


def _worth_retrying(world, agent, task) -> bool:
    """False while the exact conditions that defeated this task still hold."""
    previous = agent.blocked_by.get(str(task))

    if previous is None:
        return True

    if previous == _conditions(world, task):
        return False

    # Something relevant moved. Forget the failure and let it try again.
    del agent.blocked_by[str(task)]
    return True


def _latch(agent, task, now, log, why):
    agent.task = task
    agent.task_started = now
    agent.failed = False
    _note(log, f"assign {agent.serial}: {task} -- {why}")
    return task


def _note(log, text):
    if log is not None:
        log.append(text)
