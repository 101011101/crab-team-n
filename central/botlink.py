"""Network link to one robot: newest-frame video in, drive commands out.

Two problems this module exists to solve.

1.  Stale frames.
    cv2.VideoCapture on an MJPEG URL buffers. A reactive controller fed a
    buffered frame steers toward where the block was a second ago and
    oscillates. So we read the multipart stream ourselves and keep only
    the most recent frame, dropping everything older.

2.  The robot's 500 ms watchdog.
    A single "go forward" request makes the robot move for half a second
    and then stop. Movement must be a continuously refreshed intent, not
    an event, so a heartbeat thread resends the current command at
    DRIVE.command_hz until it is changed.
"""

import threading
import time

import cv2
import numpy as np
import requests

from config import BotConfig, DRIVE, BEHAVIOR


JPEG_SOI = b"\xff\xd8"      # start of image
JPEG_EOI = b"\xff\xd9"      # end of image


class BotLink:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.name = cfg.name
        self.bot_id = cfg.bot_id

        self._frame = None
        self._frame_time = 0.0
        self._frame_lock = threading.Lock()

        self._left = 0
        self._right = 0
        self._cmd_lock = threading.Lock()

        self._running = False
        self._threads = []

        # Diagnostics surfaced in the operator display.
        self.frames_received = 0
        self.command_failures = 0
        self.last_command_ok = 0.0
        self.stream_error = ""

        self._session = requests.Session()

    # -----------------------------------------------------------
    # LIFECYCLE
    # -----------------------------------------------------------

    def start(self):
        if self._running:
            return

        self._running = True

        self._threads = [
            threading.Thread(target=self._stream_loop, daemon=True,
                             name=f"{self.name}-stream"),
            threading.Thread(target=self._heartbeat_loop, daemon=True,
                             name=f"{self.name}-cmd"),
        ]

        for t in self._threads:
            t.start()

    def shutdown(self):
        """Stop the robot, then stop our threads. Order matters."""
        self._running = False

        for _ in range(3):
            try:
                self._session.get(f"{self.cfg.command_url}/stop", timeout=0.5)
                break
            except requests.RequestException:
                time.sleep(0.1)

        self._session.close()

    # -----------------------------------------------------------
    # VIDEO IN
    # -----------------------------------------------------------

    def _stream_loop(self):
        while self._running:
            try:
                resp = self._session.get(self.cfg.stream_url, stream=True,
                                         timeout=(3.0, 5.0))
                resp.raise_for_status()
                self.stream_error = ""

                buf = bytearray()

                for chunk in resp.iter_content(chunk_size=4096):
                    if not self._running:
                        break

                    buf.extend(chunk)

                    # Keep only the newest complete JPEG in the buffer.
                    start = buf.rfind(JPEG_SOI)
                    end = buf.rfind(JPEG_EOI)

                    if start != -1 and end > start:
                        jpg = bytes(buf[start:end + 2])
                        del buf[:end + 2]
                        self._decode(jpg)

                    # Guard against a corrupt stream growing the buffer forever.
                    if len(buf) > 512 * 1024:
                        del buf[:-64 * 1024]

            except requests.RequestException as exc:
                self.stream_error = type(exc).__name__
                time.sleep(0.6)
            except Exception as exc:                      # noqa: BLE001
                self.stream_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.6)

    def _decode(self, jpg: bytes):
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),
                             cv2.IMREAD_COLOR)

        if frame is None:
            return

        with self._frame_lock:
            self._frame = frame
            self._frame_time = time.time()
            self.frames_received += 1

    def latest_frame(self):
        """Return (frame, age_seconds) or (None, inf)."""
        with self._frame_lock:
            if self._frame is None:
                return None, float("inf")
            return self._frame.copy(), time.time() - self._frame_time

    # -----------------------------------------------------------
    # COMMANDS OUT
    # -----------------------------------------------------------

    def drive(self, left: int, right: int):
        """Set the standing intent. The heartbeat thread keeps sending it."""
        left = int(max(-4095, min(4095, left)))
        right = int(max(-4095, min(4095, right)))

        with self._cmd_lock:
            self._left, self._right = left, right

    def stop(self):
        self.drive(0, 0)

    def _heartbeat_loop(self):
        period = 1.0 / DRIVE.command_hz

        while self._running:
            with self._cmd_lock:
                left, right = self._left, self._right

            try:
                self._session.get(
                    f"{self.cfg.command_url}/cmd",
                    params={"l": left, "r": right},
                    timeout=0.4,
                )
                self.last_command_ok = time.time()
            except requests.RequestException:
                self.command_failures += 1

            time.sleep(period)

    # -----------------------------------------------------------
    # HEALTH
    # -----------------------------------------------------------

    @property
    def online(self) -> bool:
        """Online means we are still receiving pictures.

        Video is the strictest liveness test available: it needs the robot
        powered, joined to the network, and its camera working. A bot that
        answers commands but sends no frames is useless to a vision-driven
        mission, so it counts as down and its work is reassigned.
        """
        _, age = self.latest_frame()
        return age < BEHAVIOR.offline_after_s

    def status(self) -> dict:
        """Fetch /status. Blocking; call it off the control loop."""
        try:
            resp = self._session.get(f"{self.cfg.command_url}/status",
                                     timeout=0.6)
            return resp.json()
        except (requests.RequestException, ValueError):
            return {}
