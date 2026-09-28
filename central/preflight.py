"""Check every robot before the demo. Run this first, every time.

    python3 preflight.py

Verifies for each configured robot: command port answers, /status parses,
the stream delivers frames, and the motors accept a command. It sends a
brief nudge, so put the robots on blocks or clear space around them.
"""

import sys
import time

import requests

from botlink import BotLink
from config import BOTS


def check(cfg):
    print(f"\n{cfg.name}  ({cfg.ip})")
    problems = []

    # 1. Command port.
    try:
        resp = requests.get(f"{cfg.command_url}/status", timeout=2.0)
        status = resp.json()
        print(f"  [ok]   command port 80   battery {status.get('battery_v', '?')} V, "
              f"up {status.get('uptime_ms', 0) / 1000:.0f}s")
    except requests.RequestException as exc:
        problems.append(f"command port unreachable: {type(exc).__name__}")
        print(f"  [FAIL] command port 80   {type(exc).__name__}")
        print("         Is the robot powered and on this network?"
              " Does config.py have the right IP?")
        return problems
    except ValueError:
        problems.append("/status did not return JSON")
        print("  [FAIL] /status returned non-JSON. Old firmware flashed?")

    # 2. Video.
    link = BotLink(cfg)
    link.start()

    deadline = time.time() + 6.0
    frame = None

    while time.time() < deadline:
        frame, age = link.latest_frame()
        if frame is not None:
            break
        time.sleep(0.2)

    if frame is None:
        problems.append("no video frames")
        print(f"  [FAIL] stream port 81    no frames in 6s"
              f"  ({link.stream_error or 'no error reported'})")
        print("         Check the serial log for 'Camera init FAILED'"
              " and verify camera_pins.h.")
    else:
        h, w = frame.shape[:2]
        time.sleep(2.0)
        fps = link.frames_received / 8.0
        print(f"  [ok]   stream port 81    {w}x{h}, ~{fps:.1f} fps")

        if fps < 3.0:
            problems.append(f"low frame rate ({fps:.1f} fps)")
            print("  [warn] frame rate is low; the controller will react slowly.")

    # 3. Motors. Short nudge, then stop.
    print("  [..]   nudging motors for 0.6s")
    link.drive(1500, -1500)
    time.sleep(0.6)
    link.stop()
    time.sleep(0.3)

    if link.command_failures > 2:
        problems.append(f"{link.command_failures} command failures")
        print(f"  [warn] {link.command_failures} command failures during nudge")
    else:
        print("  [ok]   motor commands accepted")

    link.shutdown()
    return problems


def main():
    print("=" * 52)
    print("PREFLIGHT")
    print("=" * 52)

    all_problems = {}

    for cfg in BOTS:
        found = check(cfg)
        if found:
            all_problems[cfg.name] = found

    print("\n" + "=" * 52)

    if not all_problems:
        print("All robots ready. Run: python3 mission.py")
        return 0

    for name, problems in all_problems.items():
        for problem in problems:
            print(f"  {name}: {problem}")

    print("\nFix these before running the mission.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
