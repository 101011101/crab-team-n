"""Every robot, as one object: newest frame in, track speeds out.

Thin wrapper over central/botlink.py, which already solves the two hard parts:

  - MJPEG buffering. cv2.VideoCapture on a stream hands you a frame from a second
    ago; a reactive controller fed that steers toward where the world used to be.
    BotLink reads the multipart stream itself and keeps only the newest frame.
  - The robot's 500 ms watchdog. One "forward" request moves the robot for half a
    second and then it stops. Movement is a continuously refreshed INTENT, so a
    heartbeat thread resends the current command until it changes.

Nothing here talks to the planner. It is the wire, and only the wire.
"""
import os, sys, threading, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "central"))
from botlink import BotLink          # noqa: E402
from config import BOTS              # noqa: E402


class Fleet:
    def __init__(self, bots=None):
        self.links = {b.name: BotLink(b) for b in (bots or BOTS)}
        self.cfg = {b.name: b for b in (bots or BOTS)}
        self._started = False

    def start(self):
        for l in self.links.values():
            l.start()
        self._started = True
        return self

    def shutdown(self):
        for l in self.links.values():
            try:
                l.stop()
                l.shutdown()
            except Exception:
                pass
        self._started = False

    # --- video ---------------------------------------------------------------
    def frames(self, max_age=2.0):
        """name -> (frame, age). Only robots with a RECENT frame.

        latest_frame() returns (None, inf) when there is nothing, so the tuple is
        always truthy -- check the frame, not the tuple, or you hand None to OpenCV.
        A stale frame is worse than no frame: it describes a world that has moved.
        """
        out = {}
        for n, l in self.links.items():
            frame, age = l.latest_frame()
            if frame is not None and age <= max_age:
                out[n] = (frame, age)
        return out

    # --- commands ------------------------------------------------------------
    def drive(self, name, left, right):
        self.links[name].drive(int(left), int(right))

    def stop_all(self):
        for l in self.links.values():
            l.stop()

    # --- health --------------------------------------------------------------
    def status(self):
        out = {}
        for n, l in self.links.items():
            out[n] = dict(online=l.online, frames=l.frames_received,
                          cmd_failures=l.command_failures,
                          stream_error=l.stream_error,
                          ip=self.cfg[n].ip)
        return out

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop_all()
        self.shutdown()
