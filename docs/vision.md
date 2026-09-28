# Vision

> Source: team chat, 2026-09-19. Contains only stated ideas. Open points are listed as gaps, not filled in.

## Event

- Hack the North.
- The demo must be flashy and must show real life impact.

## Core concept

- Robotic collaboration to complete a goal.
- The model is an ant swarm.
- Three robots at the final demo.
- The heuristic is scout and explore.

## Task shape

- The robots work inside a maze.
- The robots must collaborate inside the maze to do something.
- Example: one robot moves a block to unblock a path for another robot.
- The robots must also explore.
- A colored block marks a thing to explore, like ants find food.

## Domain framing

- The framing is swappable. The technology does not change.
- Candidate 1: Amazon rainforest exploration. Ant-like behavior scouts for species.
- Candidate 2: navigate a home and collect data.
- Biodiversity rule: when measured biodiversity increases in an area, concentrate more ants on that area to solve the problem.

## Optional, only if time allows

- Each ant has its own special feature.
- The ants must coordinate which ant to call for a given task sequence.

## Implementation premise

- The general premise: a robot sends video to the computer, the computer processes it, and the processing result produces commands.
- Robots send a video stream.
- The computer turns the video stream into a 3D world. Open source options to be researched.
- The computer overlaps the worlds between the robots. The robots start from a similar location, so they see similar things. At minimum a shared view looks like a tree.
- Using the 3D world, the computer reverse-locates where a robot is and where its partner is.
- The computer calculates how to complete a task. This may use an open source model or an algorithm.
- Hard-coded conditional instructions are acceptable. Examples: if see block trail, go closer. If see block, scoop up. If see block, push.

## Gaps — status

Settled in chat 2026-09-19:

- Collaboration task: one robot clears the way for another.
- 3D world: showpiece. Markers drive navigation.
- Build time: full weekend, about 30 hours.
- Compute: Mac for the live loop. Cloud GPU available as an option.
- Robots: two kits in hand, third arrives Saturday.
- Manipulation: push-only. An arm is printable but is a stretch item.
- Project rule: always keep a working demo, and add to the working thing.

Still open. See [refined-prd.md](refined-prd.md):

- Maze construction: material, wall height, floor size.
- Species marker form.
- Biodiversity score definition.
- Blocked-path win condition.
- Per-robot specialisation.
- Robot count against the "Crab Team 8" name.
