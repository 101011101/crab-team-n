# Crab Team N — Refined PRD

> Supersedes [prd-v1-original.md](prd-v1-original.md). Architecture in [architecture.md](architecture.md). Source: team chat 2026-09-19 and the 3D reconstruction research pass. Open points are marked. Nothing here is assumed.

## One sentence

A swarm of cheap camera robots explores a maze like ants, builds one shared 3D world from all their views, concentrates on high-biodiversity areas, and clears blocked paths for each other.

## Framing

- Domain: Amazon rainforest species scouting. The framing is swappable; the technology does not change.
- Stated alternate framings: navigate a home and collect data; search and rescue; exploration and collection; training data for bigger robots.
- Ant model: robots do not talk to each other. They write to a shared world and read it back. The shared map is the pheromone trail.

## Demo narrative

1. Three robots start together and fan out into the maze.
2. One live 3D world builds on screen from all robot feeds. **Wow moment.**
3. A robot finds a coloured species marker. The marker enters the shared record.
4. Biodiversity score rises in one region. The swarm converges on it. **Ant recruitment moment.**
5. A robot meets a blocked corridor. It calls for help. A second robot arrives and pushes the block clear so the first can pass. **Collaboration moment.**
6. An operator kills one robot. Its area is reassigned live. **Resilience moment.**

## Architecture decision: split pose from geometry

- Two jobs have opposite tolerances. Do not fuse them.
- **Pose** = where each robot is. Control depends on it. Must be fast and reliable.
- **Geometry** = what the space looks like. This is the showpiece. It can be slow and noisy.
- Pose comes from ArUco or AprilTag markers fixed to the maze walls. The printed tag size sets the scale. All robots share one coordinate frame from the first frame.
- Geometry comes from monocular depth, back-projected using the marker-derived pose, then merged as point clouds.
- Result: a reconstruction failure degrades the picture. It does not stop a robot.
- Monocular reconstruction alone cannot recover metric scale. Markers remove that problem instead of solving it.

## Build ladder

Rule from the team: always keep a working demo. Add to the working thing. Never break a lower rung to build a higher one.

| Rung | Deliverable | Gate to continue | State |
| --- | --- | --- | --- |
| 0 | Reactive search, 2 robots, dropout reassignment, operator display, timeout stop | A repeatable run finds blocks and survives a robot kill | Built |
| S | Depth spike. Run Depth Anything V2 on one saved frame from a real robot camera | The depth map shows usable structure, not noise | Do first |
| 1 | Marker pose. Calibrate each camera. Tag the maze. Robots appear at true positions on a shared 2D map | Two robots hold position to within the accepted error while driving | |
| 2 | Maze, species markers, biodiversity score, recruitment | Swarm converges on the highest-scoring region without being told to | |
| 3 | Collaboration set piece. One robot clears a blocked path for another | The sequence runs twice without operator help | |
| 4 | 3D world. Depth back-projected through marker pose, merged across robots | One point cloud shows geometry seen only by different robots | |
| 5 | Stretch. Third robot, printed arm, per-robot specialisation, cloud GPU reconstruction | Only after rung 4 is stable | |

- Run the depth spike (S) before rung 1. It is the largest unknown and it costs under one hour.
- Camera calibration is required, not optional. Budget 15 minutes per camera with a checkerboard.

## Hardware state

- Two kits in hand. Third kit arrives Saturday.
- Build the demo for two robots. The third robot must be additive.
- 3D printers available. An arm is printable but is a rung 5 item.
- Robots have no gripper today. The collaboration task is push-only.
- Compute: Mac. Cloud GPU is available as an option.
- Keep the live demo loop on the Mac. A live demo must not depend on venue Wi-Fi reaching a cloud GPU.
- Use a cloud GPU only for a higher-fidelity map rendered beside the live run.

## Confirmed decisions

- Project name: Crab Team N. N scales because work is allocated over a goal tree, not over robot identities.
- Architecture: recursive goal decomposition over an AND-OR tree. See [architecture.md](architecture.md).
- Build direction: fallbacks first, then raise the abstraction. Fix interfaces early, swap implementations upward.
- Collaboration task: one robot clears the way for another.
- 3D world: showpiece. Markers drive navigation.
- Build time: full weekend, about 30 hours.
- Robot count at demo: three, if the third arrives and integrates.

## Language model features

Both are stretch items, both sit outside the control loop, and neither is built.

- **Goal setting from a sentence.** The operator types a mission; one call turns it into a goal spec. Runs once, at start. Demos well because the swarm visibly reorganises around something a person just typed.
- **Live narrator.** Reads the event log and explains what the swarm is doing in plain English. Judges watching a telemetry panel do not parse allocator log lines; a sentence does the explaining.

Rule: a model may propose goals and describe behaviour. It may not choose waypoints, assign robots, or decide whether something is passable. Those are geometry and preconditions, already written and already fast.

## Open points

- Maze construction: material, wall height, floor size.
- Species marker form: coloured blocks, printed cards, or other.
- Biodiversity score definition. Proposal raised in chat, not yet confirmed: score a region by the count of **distinct** colours found, not the count of markers. Confirm or replace.
- Blocked-path win condition: what the second robot reaches after the path clears.
- Per-robot specialisation: which features differ, if rung 5 is reached.

## Dead ends — do not spend hackathon time here

Recorded from the research pass so nobody re-opens them.

| Option | Reason |
| --- | --- |
| DROID-SLAM, DPVO | CUDA-only. Cannot run on Apple Silicon. |
| MASt3R-SLAM | Real-time numbers are RTX-4090 only. No Mac precedent. |
| ORB-SLAM3 | Needs calibration, fragile macOS build, multi-map merging is documented as brittle. |
| stella_vslam | Broken on M1. Contested licence. |
| COVINS, Swarm-SLAM, Kimera-Multi | ROS-locked, Linux only. |
| VGGT, MASt3R, Fast3R, MUSt3R, Spann3R, CUT3R | CUDA-first. Mac ports unofficial or absent. |
| Gaussian splatting SLAM, MAC-Ego3D | CUDA-only rasterisers. |

- A 2025 SLAM study found rolling shutter causes catastrophic tracking divergence in ORB-SLAM3 and RTAB-Map. The robot cameras are rolling shutter. This predicts failure for feature-based SLAM on this hardware.
- DUSt3R runs on Apple Silicon but takes 1 to 2 minutes per batch. It is not a live navigation source. It is a candidate for an offline high-quality map only.

## Risks

| Risk | Response |
| --- | --- |
| Depth model produces noise from a 320x240 rolling-shutter camera | Run the depth spike first. If it fails, rung 4 becomes a 2D occupancy map and the pitch line still holds. |
| Markers not detected at range | Tags must be 10 to 20 cm for 2 to 3 m range at this resolution. Print large. Place many. Multiple visible markers sharply improve accuracy. |
| Third robot arrives late and destabilises the demo | The demo runs on two robots. The third is additive only. |
| Maze walls are featureless and confuse detection | Markers supply the features. Do not rely on wall texture. |
| Wi-Fi saturates with three video streams | Keep QVGA. Test three streams before rung 4. |
| Time lost to a higher rung breaks a working lower rung | Commit each working rung before starting the next. |
