"""Mission coordinator: run this on the laptop.

    python3 mission.py

Per tick, for every robot: take its newest frame, find green blocks, run the
reactive controller, send the resulting track speeds. Robots that stop
sending video are declared down and their sector is handed to a survivor.
"""

import argparse
import json
import time

import cv2

import ui
from behavior import ScoutBehavior
from botlink import BotLink
from config import BEHAVIOR, BOTS, TARGETS, VISION
from targets import TargetRegistry
from vision import annotate, build_mask, detect_green


TICK_HZ = 20.0


class Scout:
    """One robot: its link, its controller, and its display state."""

    def __init__(self, cfg, registry):
        self.cfg = cfg
        self.link = BotLink(cfg)
        self.brain = ScoutBehavior(cfg.name, cfg.sector, registry)

        self.forced_offline = False
        self.was_online = False

        self._frame_count = 0
        self._fps_at = time.time()
        self.fps = 0.0

    @property
    def name(self):
        return self.cfg.name

    @property
    def online(self):
        return self.link.online and not self.forced_offline

    def tick_fps(self):
        now = time.time()
        self._frame_count += 1

        if now - self._fps_at >= 1.0:
            self.fps = self._frame_count / (now - self._fps_at)
            self._frame_count = 0
            self._fps_at = now


class Mission:
    def __init__(self, show_mask=False):
        self.registry = TargetRegistry(TARGETS)
        self.scouts = [Scout(cfg, self.registry) for cfg in BOTS]

        self.running = False
        self.started_at = None
        self.total_runtime = 0.0

        self.show_mask = show_mask
        self.events = []

    # -----------------------------------------------------------

    def log(self, text):
        stamp = self.elapsed
        self.events.append((stamp, text))
        print(f"[{stamp:7.1f}s] {text}", flush=True)

    @property
    def elapsed(self):
        if self.started_at is None:
            return self.total_runtime
        return self.total_runtime + (time.time() - self.started_at)

    # -----------------------------------------------------------

    def start(self):
        for scout in self.scouts:
            scout.link.start()

        print("Connecting to robots...")
        print("  " + "\n  ".join(
            f"{s.name:10s} cmd {s.cfg.command_url}  stream {s.cfg.stream_url}"
            for s in self.scouts))
        print("\nPress SPACE in the display window to begin the mission.\n")

    def shutdown(self):
        for scout in self.scouts:
            scout.link.shutdown()

    def toggle_running(self):
        if self.running:
            self.total_runtime += time.time() - self.started_at
            self.started_at = None
            self.running = False

            for scout in self.scouts:
                scout.link.stop()

            self.log("mission STOPPED by operator")
        else:
            self.started_at = time.time()
            self.running = True
            self.log("mission STARTED")

    # -----------------------------------------------------------
    # DROPOUT AND REASSIGNMENT
    # -----------------------------------------------------------

    def check_dropouts(self):
        for scout in self.scouts:
            if scout.online:
                if not scout.was_online:
                    self.log(f"{scout.name} online")
                scout.was_online = True
                continue

            if not scout.was_online:
                continue

            # Just went down.
            scout.was_online = False
            scout.link.stop()

            self.log(f"{scout.name} LOST (no video for "
                     f"{BEHAVIOR.offline_after_s:.1f}s)")

            survivors = [s for s in self.scouts
                         if s is not scout and s.online]

            if not survivors:
                self.log("no robots left to take over")
                continue

            # Give the work to whichever survivor is carrying least.
            heir = min(survivors, key=lambda s: len(s.brain.absorbed))
            heir.brain.absorb(scout.name)

            self.log(f"{scout.name} sector reassigned to {heir.name}")

    # -----------------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------------

    def run(self):
        self.start()

        window = "Robotic Scout Swarm"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

        period = 1.0 / TICK_HZ

        try:
            while True:
                loop_start = time.time()

                self.check_dropouts()

                views = [self.step_scout(s) for s in self.scouts]

                canvas = ui.render(views, self.registry, self.running,
                                   self.elapsed, self.events)

                cv2.imshow(window, canvas)

                if not self.handle_keys(cv2.waitKey(1) & 0xFF):
                    break

                slack = period - (time.time() - loop_start)

                if slack > 0:
                    time.sleep(slack)

        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            print("\nstopping robots...")
            self.shutdown()
            cv2.destroyAllWindows()
            self.print_survey()

    def step_scout(self, scout):
        frame, age = scout.link.latest_frame()

        online = scout.online

        if frame is not None:
            scout.tick_fps()

        detections, mask = detect_green(frame, VISION) if frame is not None \
            else ([], None)

        left, right = scout.brain.step(detections, self.running, online)

        if online:
            scout.link.drive(left, right)
        else:
            scout.link.stop()

        if frame is None:
            display = None
        elif self.show_mask and mask is not None:
            display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            display = annotate(frame, detections, scout.brain.primary)

        refractory = max(0.0, scout.brain.refractory_until - time.time())

        return {
            "name": scout.name,
            "frame": display,
            "state": scout.brain.state,
            "online": online,
            "fps": scout.fps if online else 0.0,
            "event": scout.brain.last_event,
            "absorbed": scout.brain.absorbed,
            "refractory": refractory,
        }

    # -----------------------------------------------------------

    def handle_keys(self, key):
        if key in (ord("q"), 27):
            return False

        if key == ord(" "):
            self.toggle_running()

        elif key == ord("m"):
            self.show_mask = not self.show_mask

        elif key == ord("s"):
            self.save_survey()

        elif key == ord("r"):
            for scout in self.scouts:
                scout.forced_offline = False
            self.log("all robots restored")

        elif key in (ord("1"), ord("2"), ord("3")):
            index = key - ord("1")

            if index < len(self.scouts):
                scout = self.scouts[index]
                scout.forced_offline = not scout.forced_offline

                self.log(f"{scout.name} "
                         f"{'force-dropped by operator' if scout.forced_offline else 'restored'}")

        return True

    # -----------------------------------------------------------

    def survey(self):
        return {
            "mission_seconds": round(self.elapsed, 1),
            "targets_found": self.registry.count,
            "duplicate_sightings_merged": self.registry.merges,
            "targets": [
                {
                    "id": t.target_id,
                    "first_seen_by": t.first_seen_by,
                    "at_second": round(t.first_seen_at - (self.started_at or t.first_seen_at), 1),
                    "sightings": t.sightings,
                    "seen_by": sorted(t.seen_by),
                }
                for t in self.registry.all()
            ],
            "robots": [
                {
                    "name": s.name,
                    "sector": s.cfg.sector,
                    "online_at_end": s.online,
                    "claims": s.brain.claims,
                    "absorbed_sectors": s.brain.absorbed,
                    "frames_received": s.link.frames_received,
                    "command_failures": s.link.command_failures,
                }
                for s in self.scouts
            ],
            "events": [{"at_second": round(t, 1), "event": e}
                       for t, e in self.events],
        }

    def save_survey(self):
        name = f"survey_{time.strftime('%Y%m%d_%H%M%S')}.json"

        with open(name, "w") as handle:
            json.dump(self.survey(), handle, indent=2)

        self.log(f"survey saved to {name}")

    def print_survey(self):
        data = self.survey()

        print()
        print("=" * 52)
        print("FINAL SURVEY")
        print("=" * 52)
        print(f"  mission time        {data['mission_seconds']}s")
        print(f"  targets found       {data['targets_found']}")
        print(f"  duplicates merged   {data['duplicate_sightings_merged']}")
        print()

        for target in data["targets"]:
            print(f"    #{target['id']}  found by {target['first_seen_by']}"
                  f" at {target['at_second']}s"
                  f"  ({target['sightings']} sighting(s))")

        print()

        for robot in data["robots"]:
            note = (f" absorbed {','.join(robot['absorbed_sectors'])}"
                    if robot["absorbed_sectors"] else "")
            print(f"    {robot['name']:10s} claims {robot['claims']}"
                  f"  frames {robot['frames_received']}"
                  f"  cmd failures {robot['command_failures']}{note}")

        print("=" * 52)


def main():
    parser = argparse.ArgumentParser(description="Robotic Scout Swarm coordinator")
    parser.add_argument("--mask", action="store_true",
                        help="start with the detection mask shown")
    args = parser.parse_args()

    Mission(show_mask=args.mask).run()


if __name__ == "__main__":
    main()
