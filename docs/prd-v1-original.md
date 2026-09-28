# Robotic Scout Swarm — Hackathon PRD (v1, archived)

> Archived verbatim. Superseded by the refined PRD. Kept for traceability.

**Status:** Team working draft
**Date:** September 19, 2026

## One sentence

Build a small fleet of camera equipped robots that explores a controlled arena together, records discovered targets on a shared map, and redistributes work when one robot drops out.

## Why we are building it

Inspecting an unfamiliar space can require physical presence. Several inexpensive scouts could search separate areas and combine what they find before a person or larger robot enters. This hackathon tests the coordination idea in a repeatable indoor arena. It does not claim to solve outdoor exploration or industrial inspection yet.

## Demo experience

An operator starts a mission: **"Explore the arena and find all green blocks."** The blocks stand in for objects of interest.

1. At least three robots receive different search areas or routes.
2. Each robot moves without individual manual steering and sends camera observations and status to a central computer.
3. A live display shows robot positions, explored areas, and discovered blocks in one shared view.
4. The system merges repeat sightings of the same block.
5. We stop or disconnect one robot. Its unfinished area is assigned to an available robot.
6. The operator sees a final survey with discovered targets and exploration progress.

**Fleet target:** 12 robots if replication succeeds. **Minimum credible demo:** three working robots. A reliable three robot mission takes priority over the fleet count.

## MVP requirements

### Robots

- Each robot has a mobile chassis, battery, controller, Wi-Fi link, and one onboard camera used to find targets.
- Each robot executes motion locally from high level commands and reports its ID, availability, and observations.
- Each robot stops locally when commands become stale or the connection is lost.
- Robots use a common build so a validated prototype can be replicated.

### Central computer

- Tracks robot positions in a shared arena coordinate system.
- Assigns distinct search areas and sends movement commands.
- Receives camera observations, detects green blocks, and maintains a shared target record.
- Avoids obvious duplicate counting of the same block.
- Reassigns unfinished work when a robot becomes unavailable.
- Provides start/stop controls and a readable live and final survey.

### Demo environment

- A controlled indoor arena with a known floor surface, visible targets, and consistent lighting.
- A localization method that works in the chosen arena. Visual landmarks or overhead tracking are acceptable; if overhead tracking is used, the demo should state that dependency. Target discovery still uses onboard cameras.
- One local network that supports the working fleet.

## Definition of done

The MVP is complete when a repeatable run shows three physical robots covering different areas without individual steering, detecting targets through their cameras, displaying one target record without obvious duplicates, and reassigning a stopped robot's unfinished work. Every robot must stop safely on command timeout.

Before the final demo, set and record practical thresholds for mission time, target detection, position accuracy, reassignment delay, and battery runtime based on the actual arena and prototype. Run the full sequence at least twice under demo conditions.

## Proposed system design

The central computer owns the mission plan, shared map, target records, and operator display. Robots handle motion timing, image capture, communication, and local stopping. The system depends on the coordinator and Wi-Fi; worker loss is recoverable through reassignment.

The preferred body is a small four legged walker inspired by Sesame, with roughly eight servos per robot. Its geometry and electronics are references, not locked requirements. A wheeled chassis is an acceptable fallback if walking threatens the coordination demo. The camera controller, servo driver, battery, and regulators remain selections to validate on one prototype before fleet replication.

Keep localization and mapping as simple as the demo allows. A useful shared arena map is required; dense 3D reconstruction is not.

## Build order and gates

| Stage | Deliverable | Gate to continue |
| --- | --- | --- |
| 1. Arena and interfaces | Arena layout, target appearance, robot command/status format, and localization approach | Team can describe and test one end-to-end mission path |
| 2. One robot | Chassis, power, camera, Wi-Fi, motion, local timeout stop | Robot moves while sending usable images without resets |
| 3. Three robots | Shared map, task allocation, target merging, reassignment, operator display | Complete repeatable MVP run |
| 4. Fleet expansion | Replicated and calibrated robots, network and charging plan | Add robots only while the three robot demo remains reliable |

Four teammates can own mechanical/assembly, power/electronics, robot firmware, and central software, while agreeing on interfaces together. These are work areas, not fixed personal assignments.

## Scope after MVP

- Expand from three robots toward 12.
- Improve localization, coverage planning, and traffic handling.
- Add a lightweight cooperative object task only if the core survey demo is stable.

Outdoor terrain, autonomous charging, decentralized control, dense 3D mapping, biodiversity measurement, and moving heavy objects are outside this hackathon demo.

## Decisions the team must settle now

1. **Arena and localization:** floor size, lighting, block size/color, and whether landmarks or overhead tracking provide positions.
2. **Mobility:** validate the walker's loaded movement and print time; choose the wheeled fallback promptly if needed.
3. **Robot electronics:** confirm camera/controller pinout, servo control, power rails, battery connectors, and safe stop behavior on one unit.
4. **Compute and network:** identify the central computer and test actual image traffic with three robots before scaling.
5. **Budget and inventory:** confirm parts physically available, tools, filament, chargers, spares, and remaining spending limit. Prior delivery dates in planning notes have passed and must not be used as current purchasing assumptions.

## Main risks and responses

| Risk | Response |
| --- | --- |
| Walker cannot carry electronics or walk reliably | Test one loaded build early; switch to wheels if necessary |
| Camera streaming interferes with motion or overloads Wi-Fi | Reduce image size/rate and test progressively at three robots |
| Robot positions are unreliable | Use arena markers or overhead tracking and document the setup |
| Power causes controller resets or unsafe behavior | Measure rails under load; keep servo and controller supplies appropriate; test timeout stop |
| Fleet replication consumes demo time | Preserve a working three robot setup before adding more |

## Team reference material

The detailed vision, build guide, hardware outline, and readiness checklist contain background and candidate parts. They are implementation references; this document defines the shared hackathon outcome.
