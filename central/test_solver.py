"""Prove the recursion and the allocation rules with no hardware.

    ./.venv/bin/python test_solver.py

Every acceptance line in docs/tickets.md T2 has a numbered section below.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world import (World, Agent, PlaceSeen, EdgeSeen, ThingSeen, AgentAt, Surveyed,
                   ThingObserved, UNKNOWN, FRONTIER, EXPLORED, REACHABLE, BLOCKED,
                   IMPASSABLE, BLOCK, WALL, HOLE, PUSH as CAP_PUSH)
import solver
from solver import (Blocked, Clear, Observed, Reach, Solved, Survey, Task, Unknown,
                    holds, solve)
from allocator import (assign, best_fork, release_reason,
                       RESOLVED, INVALIDATED, FAILED, TIMED_OUT, LOST)

def ops(result):
    return [str(t) for t in result.plan]

def in_order(lines, *needles):
    """Every needle appears, each at or after the one before it."""
    text, at = "\n".join(lines), 0
    for needle in needles:
        hit = text.find(needle, at)
        assert hit >= 0, f"missing {needle!r} after offset {at}\n{text}"
        at = hit
    return True

def corridor(weight=1, cause=BLOCK, crabs=(1, 2)):
    """R1 --e1-- C --e2-- R2, with e2 obstructed. The demo arena."""
    w = World()
    w.observe(PlaceSeen("R1", EXPLORED, 1))
    w.observe(PlaceSeen("C", EXPLORED, 1))
    w.observe(PlaceSeen("R2", FRONTIER, 1))
    w.observe(EdgeSeen("R1", "C", "e1", REACHABLE))
    w.observe(EdgeSeen("C", "R2", "e2", BLOCKED, cause))
    w.observe(ThingSeen("block", "C", "block", weight, "e2"))
    for serial in crabs:
        w.add_agent(Agent(serial, "R1"))
    return w

# --- 1. four operators, no more ---
assert solver.OPERATORS == ("GOTO", "EXPLORE", "PUSH", "OBSERVE"), solver.OPERATORS
assert len(solver.OPERATORS) == 4
assert len({solver.GOTO, solver.EXPLORE, solver.PUSH, solver.OBSERVE}) == 4
print(f"  operators: {', '.join(solver.OPERATORS)} -- exactly {len(solver.OPERATORS)}")

# --- 2. the corridor trace from architecture.md, line by line ---
w = corridor()
assert not holds(w, Survey("R2")), "survey(R2) must not hold before the room is entered"
assert not holds(w, Reach("R2")), "reach(R2) must not hold: e2 is in the way"
e2 = w.edge_named("e2")
assert (e2.state, e2.cause) == (BLOCKED, BLOCK), (e2.state, e2.cause)

trace = []
first = solve(w, Survey("R2"), trace)
assert isinstance(first, Solved), first
in_order(trace,
         "survey(R2)", "EXPLORE(1, R2)", "reach(R2)", "GOTO(1, R2) via R1-C-R2",
         "clear(e2)", "e2 is BLOCKED(BLOCK)", "PUSH(1, block, e2)",
         "requires capable(1, PUSH) -> true", "requires agents >= weight(block) = 1 -> true",
         "reach(C)", "GOTO(1, C) via R1-C", "-> SOLVED")
clears = [t for t in first.plan if t.op == "PUSH"]
assert [t.target for t in clears] == ["e2"], ops(first)
assert clears[0].cause == BLOCK
assert ops(first) == ["GOTO(1, C)", "PUSH(1, e2)", "GOTO(1, R2)", "EXPLORE(1, R2)"], ops(first)
print("  corridor trace:")
for line in trace:
    print("    " + line)

w.observe(EdgeSeen("C", "R2", "e2", REACHABLE))          # the push lands
after = solve(w, Survey("R2"))
assert isinstance(after, Solved), after
assert not [t for t in after.plan if t.op == "PUSH"], ops(after)
assert ops(after) == ["GOTO(1, R2)", "EXPLORE(1, R2)"], ops(after)
print(f"  clear(e2) emitted -> {ops(first)}")
print(f"  e2 resolved -> survey(R2) {ops(after)}")

# --- 3. a HOLE is Blocked, never a clear task, and the OR node routes around ---
w = corridor(cause=HOLE)
w.observe(PlaceSeen("D", EXPLORED, 1))
w.observe(EdgeSeen("R1", "D", "e3", REACHABLE))
w.observe(EdgeSeen("D", "R2", "e4", BLOCKED, BLOCK))
w.observe(ThingSeen("rock", "D", "block", 1, "e4"))

hole = solve(w, Clear("e2"))
assert isinstance(hole, Blocked), hole
assert hole.cause == HOLE and hole.edge == "e2", hole
trace = []
around = solve(w, Survey("R2"), trace)
assert isinstance(around, Solved), around
assert not [t for t in around.plan if t.target == "e2"], "nothing may be tasked against a HOLE"
assert [t.target for t in around.plan if t.op == "PUSH"] == ["e4"], ops(around)
in_order(trace, "GOTO(1, R2) via R1-C-R2", "requires cause HOLE is removable -> FALSE",
         "BLOCKED HOLE on e2", "GOTO(1, R2) via R1-D-R2", "PUSH(1, rock, e4)")
assert isinstance(solve(w, Clear("e1")), Solved), "a REACHABLE edge already holds"
w.observe(EdgeSeen("C", "R2", "e2", IMPASSABLE))
assert isinstance(solve(w, Clear("e2")), Blocked), "IMPASSABLE is not a push either"
print(f"  HOLE: {hole.reason} -> Blocked, alternate route {ops(around)}")

# --- 4. Unknown is what produces an EXPLORE task ---
w = World()
w.observe(PlaceSeen("R1", EXPLORED, 1))
w.observe(PlaceSeen("R3", FRONTIER, 1))                  # seen, but no edge known to it
w.add_agent(Agent(1, "R1"))
unknown = solve(w, Survey("R3"))
assert isinstance(unknown, Unknown), unknown
assert unknown.goal == Reach("R3", 1), unknown.goal
task = assign(w, w.live_agents(), now=100.0)[1]
assert (task.op, task.target) == ("EXPLORE", "R3"), task
assert isinstance(solve(w, Survey("R9")), Unknown), "a place not in the model is Unknown"
w.observe(EdgeSeen("R1", "R3", "e9", UNKNOWN))
assert isinstance(solve(w, Clear("e9")), Unknown), "an unseen edge is Unknown, not Blocked"
print(f"  Unknown({unknown.goal}) -> {task}")

# --- 5. no preemption: a latched crab ignores a nearer branch ---
w = corridor(crabs=(1,))
w.observe(EdgeSeen("C", "R2", "e2", REACHABLE))
held = assign(w, w.live_agents(), now=0.0)[1]
assert (held.op, held.target) == ("GOTO", "R2"), held
w.observe(AgentAt(1, "C"))
w.observe(PlaceSeen("N1", FRONTIER, None))               # a nearer branch appears
w.observe(EdgeSeen("C", "N1", "e5", REACHABLE))
# A re-ranking allocator would move crab 1 onto N1; this one must not.
assert best_fork(w, ["N1", "R2"]) == "N1", "the test must offer a branch the pool would prefer"
again = assign(w, w.live_agents(), now=1.0)[1]
assert again == held, f"latched crab switched: {held} -> {again}"
assert w.agents[1].task == held
assert release_reason(w, w.agents[1], 1.0) is None
print(f"  latched: {held} held while N1 opened next door")

# --- 6. two free crabs, one pooled branch: lower serial, every run ---
def one_branch():
    w = World()
    w.observe(PlaceSeen("R1", EXPLORED, None))
    w.observe(PlaceSeen("R2", FRONTIER, None))           # no finder: straight to the pool
    w.observe(EdgeSeen("R1", "R2", "e1", REACHABLE))
    w.add_agent(Agent(1, "R1"))
    w.add_agent(Agent(2, "R1"))
    return w

runs = set()
for _ in range(50):
    w = one_branch()
    out = assign(w, w.live_agents(), now=0.0)
    runs.add(tuple(sorted((s, str(t)) for s, t in out.items())))
assert len(runs) == 1, runs
winners = dict(next(iter(runs)))
assert set(winners) == {1}, f"one branch must go to one crab, got {winners}"
assert "GOTO(1, R2)" in winners[1]
print(f"  50 runs, 2 free crabs, 1 branch -> {winners} identical every time")

# --- 7. the pool drains deepest first ---
w = World()
w.observe(PlaceSeen("R1", EXPLORED, None))
w.observe(PlaceSeen("A", EXPLORED, None))
w.observe(PlaceSeen("B", FRONTIER, None))                # depth 1
w.observe(PlaceSeen("Z", FRONTIER, None))                # depth 2
w.observe(EdgeSeen("R1", "A", "a", REACHABLE))
w.observe(EdgeSeen("R1", "B", "b", REACHABLE))
w.observe(EdgeSeen("A", "Z", "z", REACHABLE))
w.add_agent(Agent(1, "R1"))
w.add_agent(Agent(2, "R1"))
assert (w.depth("B"), w.depth("Z")) == (1, 2)
out = assign(w, w.live_agents(), now=0.0)
assert out[1].target == "Z" and out[2].target == "B", {s: str(t) for s, t in out.items()}
assert best_fork(w, ["B", "Z"]) == "Z"
print(f"  pool: crab1 -> {out[1]} (depth 2), crab2 -> {out[2]} (depth 1)")

# --- 8. discovery implies ownership ---
w = World()
w.observe(PlaceSeen("R1", EXPLORED, None))
w.observe(PlaceSeen("F", FRONTIER, 2))                   # crab 2 found it
w.observe(EdgeSeen("R1", "F", "f", REACHABLE))
w.add_agent(Agent(1, "R1"))
w.add_agent(Agent(2, "R1"))
out = assign(w, w.live_agents(), now=0.0)
assert 2 in out and out[2].target == "F", {s: str(t) for s, t in out.items()}
assert 1 not in out, "the finder owns it; the other crab gets nothing"
print(f"  ownership: F found by crab 2 -> {out[2]}, crab 1 idle")

# --- 9. every release condition fires, and only then ---
def latched():
    w = corridor(crabs=(1,))
    w.observe(EdgeSeen("C", "R2", "e2", REACHABLE))
    assign(w, w.live_agents(), now=0.0)
    return w, w.agents[1]

w, a = latched()
assert release_reason(w, a, 10.0) is None
w.observe(AgentAt(1, "R2"))
assert release_reason(w, a, 10.0) == RESOLVED, a.task

w, a = latched()
a.failed = True
assert release_reason(w, a, 10.0) == FAILED

w, a = latched()
assert release_reason(w, a, a.timeout_s + 0.1) == TIMED_OUT

w, a = latched()
a.lost = True
assert release_reason(w, a, 10.0) == LOST
assert assign(w, [a], now=10.0) == {} and a.task is None, "a lost crab holds nothing"

w, a = latched()
a.task = Task("PUSH", "e2", 1, BLOCK)                    # the cause turned out to be a hole
w.observe(EdgeSeen("C", "R2", "e2", BLOCKED, HOLE))
assert release_reason(w, a, 10.0) == INVALIDATED

w, a = latched()
log = []
w.observe(Surveyed("R2"))
assign(w, w.live_agents(), now=1.0, log=log)
assert any(RESOLVED in line for line in log), log
print(f"  releases: {RESOLVED} | {INVALIDATED} | {FAILED} | {TIMED_OUT} | {LOST}")

# --- 10. a block too heavy for one crab is Blocked, and that is the other crab's task ---
w = corridor(weight=2, crabs=(1,))
alone = solve(w, Clear("e2"))
assert isinstance(alone, Blocked) and alone.cause == BLOCK, alone
assert "2 crabs" in alone.reason, alone.reason
pair = corridor(weight=2, crabs=(1, 2))
together = solve(pair, Clear("e2"))
assert isinstance(together, Solved), together
solo = assign(w, w.live_agents(), now=0.0)[1]
assert (solo.op, solo.target) == ("PUSH", "e2"), solo
print(f"  weight 2: 1 crab -> Blocked({alone.reason}) -> {solo};  2 crabs -> {ops(together)}")

# --- 11. OBSERVE reaches a thing and stops once it is observed ---
w = corridor()
w.observe(EdgeSeen("C", "R2", "e2", REACHABLE))
w.observe(ThingSeen("ant", "R2", "species", 0, None))
seen = solve(w, Observed("ant"))
assert isinstance(seen, Solved) and ops(seen) == ["GOTO(1, R2)", "OBSERVE(1, ant)"], ops(seen)
w.observe(ThingObserved("ant"))
assert solve(w, Observed("ant")).plan == [], "an observed thing needs no task"
assert isinstance(solve(w, Observed("moth")), Unknown), "an unheard-of thing is Unknown"
print(f"  observe: {ops(seen)}")

# --- 12. ten places, one frame budget ---
def ten_places():
    """The guardrail limit: 10 places, 13 edges, 3 crabs, and every deep room
    behind an obstruction, so the timing covers the recursion and not a walk."""
    w = World()
    w.observe(PlaceSeen("P0", EXPLORED, None))
    for i in range(1, 10):
        w.observe(PlaceSeen(f"P{i}", FRONTIER if i >= 5 else EXPLORED, None))
        w.observe(EdgeSeen(f"P{(i - 1) // 2}", f"P{i}", f"e{i}", REACHABLE))
    w.observe(EdgeSeen("P1", "P2", "x1", REACHABLE))     # a cycle, so routes branch
    w.observe(EdgeSeen("P3", "P7", "e7", BLOCKED, HOLE))
    w.observe(EdgeSeen("P4", "P7", "x2", BLOCKED, BLOCK))
    w.observe(EdgeSeen("P4", "P9", "e9", BLOCKED, BLOCK))
    w.observe(EdgeSeen("P3", "P9", "x3", BLOCKED, WALL))
    w.observe(EdgeSeen("P8", "P9", "x4", BLOCKED, HOLE))
    w.observe(ThingSeen("crate", "P4", "block", 2, "x2"))
    w.observe(ThingSeen("log", "P4", "block", 1, "e9"))
    for serial in (1, 2, 3):
        w.add_agent(Agent(serial, "P0"))
    return w

w = ten_places()
assert len(w.places) == 10 and len(w.edges) == 13, (len(w.places), len(w.edges))
for target, pushed in (("P9", "e9"), ("P7", "x2")):
    result = solve(w, Survey(target))
    assert isinstance(result, Solved), (target, result)
    assert [t.target for t in result.plan if t.op == "PUSH"] == [pushed], (target, ops(result))

worst = 0.0
for target in ("P9", "P7", "P5", "P8"):
    samples = []
    for _ in range(200):
        t0 = time.perf_counter()
        solve(w, Survey(target))
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    worst = max(worst, samples[-1])
    print(f"    survey({target}): median {samples[len(samples)//2]:.3f} ms  "
          f"p95 {samples[int(len(samples)*0.95)]:.3f} ms  max {samples[-1]:.3f} ms")

worlds = [ten_places() for _ in range(200)]
t0 = time.perf_counter()
for fresh in worlds:
    assign(fresh, fresh.live_agents(), now=0.0)
alloc_ms = (time.perf_counter() - t0) / len(worlds) * 1000.0
assert worst < 16.0, f"solver took {worst:.2f} ms, over one frame"
print(f"  ten places, 3 crabs: worst solve {worst:.3f} ms, full assign {alloc_ms:.3f} ms, "
      f"of a 16 ms frame")

# ---------------------------------------------------------------
# 13. A task that cannot succeed is not reissued forever
# ---------------------------------------------------------------
# Found by running the allocator in a loop: a lone crab facing a two-crab
# block was released for execution failure and handed the identical task on
# the very next tick, forever. The crab must try, learn, and stop.

w = corridor(weight=2, crabs=(1,))
solo_crab = w.agents[1]
attempts = []

for cycle in range(6):
    attempts.append(str(assign(w, [solo_crab], now=cycle * 10.0).get(1)))
    solo_crab.failed = True

assert attempts[0] != "None", f"should attempt once: {attempts}"
assert all(a == "None" for a in attempts[1:]), f"livelock: {attempts}"

# The quarantine must lift on the condition that caused it, or the crab
# would never collaborate once help arrived.
w2 = corridor(weight=2, crabs=(1,))
first = w2.agents[1]
assign(w2, [first], now=0.0)
first.failed = True
assign(w2, [first], now=10.0)
assert assign(w2, [first], now=20.0).get(1) is None, "should stay quarantined"

helper = w2.add_agent(Agent(serial=2, place="R1"))
w2.observe(AgentAt(2, "R1"))
assert assign(w2, [first, helper], now=30.0).get(1) is not None, \
    "must retry once a second crab can help push"

print(f"  livelock: 1 crab vs a 2-crab block tries once then stops, "
      f"retries when a second crab arrives")

print("\nALL SOLVER CHECKS PASSED")
