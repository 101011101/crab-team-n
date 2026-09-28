"""The one window the dashboard looks through. HTTP, read-only, 5 Hz.

    Fleet.status()      online / frames / ip
    LiveLoop.status()   pose / mode / confidence / goal
    MapService.stats()  rounds / registration rate
    map2d.json          frame / grid / boxes
                             |
                    GET /api/live/state        <- the dashboard polls this
                    GET /api/live/health

`CONTRACT.md` is the authority on the JSON. This file serves it and nothing else.

TWO RULES, AND THEY ARE WHY THIS FILE IS SHAPED THIS WAY

1. A POLL MUST NOT BE ABLE TO KILL THE ROBOTS. This runs inside the process that is
   driving hardware. Every handler catches everything and answers 500 with a JSON
   body, because a traceback escaping a request thread is a browser tab stopping a
   robot.

2. A POLL MUST NOT BE ABLE TO STALL THE CONTROL LOOP. Shared state is read under the
   owner's lock, copied, and the lock released. Serialisation happens afterwards,
   outside every lock. The map file is parsed once per mtime, not once per poll --
   at 5 Hz a re-parse of a 10k-cell grid every request is pure waste.

UNITS: `metres_per_unit` is 1.0 and `calibrated` is false until somebody measures a
known distance in the room. Monocular SfM fixes a scene only up to scale, so every
number here is a MODEL UNIT. Pass calibrated=True only when that measurement exists.
"""
import base64
import json
import math
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "control"))
import grid as G                     # noqa: E402  (fallback map reader)

SOURCE = "crab-team-n/live"
MAX_EVENTS = 50

# The loop's task names -> the dashboard's RobotStatus values. The dashboard has
# exactly six; inventing a seventh here breaks its renderer, so everything unknown
# lands on "idle" rather than on something that looks plausible.
STATUS_OF_MODE = {
    "push": "pushing",
    "drive": "driving",
    "explore": "driving",
    "idle": "idle",
    "wait": "idle",
    "stale-pose": "idle",
    "offline": "idle",
}


class TelemetryBridge(threading.Thread):
    """Serves the live state over HTTP. Owns no state of its own but the cache."""

    def __init__(self, fleet, mapsvc, loop=None, port=8787, host="0.0.0.0",
                 metres_per_unit=1.0, calibrated=False, t0=None):
        super().__init__(daemon=True, name="TelemetryBridge")
        self.fleet, self.mapsvc, self.loop = fleet, mapsvc, loop
        self.port, self.host = port, host
        self.metres_per_unit, self.calibrated = float(metres_per_unit), bool(calibrated)
        self.t0 = t0 or time.time()

        self._lock = threading.Lock()
        self._cache_key = None       # (path, mtime, size) of the parsed map
        self._cache = None           # {"frame":..., "grid":..., "boxes":...}
        self._events = deque(maxlen=MAX_EVENTS)
        self._seen_online = {}       # robot -> bool, for online/offline events
        self._seen_rounds = 0
        self._seen_map_key = None
        self.last_error = ""
        self.requests = 0
        self.httpd = None

    # --- lifecycle -----------------------------------------------------------
    @property
    def url(self):
        return f"http://localhost:{self.port}/api/live/state"

    def run(self):
        try:
            self.httpd = ThreadingHTTPServer((self.host, self.port), _make_handler(self))
            self.httpd.daemon_threads = True
            self.httpd.serve_forever(poll_interval=0.2)
        except Exception as e:                      # a dead bridge must not be silent
            self.last_error = f"{type(e).__name__}: {e}"
            print(f"[bridge] FAILED on port {self.port}: {self.last_error}")

    def stop(self):
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass

    # --- events --------------------------------------------------------------
    def event(self, text, t=None):
        """Anything worth a line in the dashboard's log. Newest last, last 50."""
        with self._lock:
            self._events.append({"t": round(t if t is not None else self._t(), 1),
                                 "text": str(text)})

    def _t(self):
        return time.time() - self.t0

    # --- the map, parsed once per mtime --------------------------------------
    def _map(self):
        """The current map as contract fields, or None when there is no map yet.

        The map JSON already stores exactly the 0/1/2/3 bytes the contract wants, so
        the base64 goes through verbatim. Recombining control/grid.py's `.state` and
        `.pushable` would give the same answer today and drift the day stage 4 adds
        a fourth code. The Grid path stays only as a fallback for a map that somehow
        has no raw `grid` field.
        """
        with getattr(self.mapsvc, "lock", _NULL_LOCK):
            path = self.mapsvc.map_path          # copied out; lock released here
        if not path or not os.path.exists(path):
            return None

        st = os.stat(path)
        key = (path, st.st_mtime, st.st_size)
        with self._lock:
            if key == self._cache_key:
                return self._cache

        payload = self._parse_map(path)              # slow part, outside the lock
        with self._lock:
            self._cache_key, self._cache = key, payload
        return payload

    @staticmethod
    def _parse_map(path):
        m = json.load(open(path))
        cols, rows = int(m["w"]), int(m["h"])
        raw = m.get("grid")
        if raw:
            cells = np.frombuffer(base64.b64decode(raw), np.uint8)
        else:                                        # fallback: rebuild 0/1/2/3
            g = G.Grid(path)
            cells = np.where(g.pushable, 3,
                             np.where(g.static, 2, g.state)).astype(np.uint8).ravel()
        if cells.size != cols * rows:
            raise ValueError(f"map grid is {cells.size} bytes, expected {cols*rows}")
        bad = int(cells.max()) if cells.size else 0
        if bad > 3:
            raise ValueError(f"map grid holds byte {bad}, contract allows 0-3")
        ox, oy = float(m["origin"][0]), float(m["origin"][1])
        return {
            # not a contract field: which frame this map is drawn in, so the units
            # block can tell a real metre from a model unit
            "_frame_name": m.get("frame"),
            "frame": {"originX": ox, "originY": oy, "cell": float(m["cell"]),
                      "cols": cols, "rows": rows},
            "grid": base64.b64encode(cells.tobytes()).decode("ascii"),
            "boxes": [{"id": str(b.get("id", f"box{i}")),
                       "x": float(b["cx"]), "y": float(b["cz"]),
                       "w": float(b["w"]), "h": float(b["h"])}
                      for i, b in enumerate(m.get("boxes", []))],
        }

    def _units(self, mp=None):
        """A map drawn in the room frame IS metres, and must say so.

        arena_map.py builds its grid from pipeline/common/constants.py, whose scale
        comes from the printed tag edge. That is a measured metre, so `calibrated`
        is true and the dashboard may print "m". A stage-4 SfM map has no metre in
        it at all and stays uncalibrated however convincing its numbers look.
        """
        if (mp or {}).get("_frame_name") == "room":
            return {"metres_per_unit": 1.0, "calibrated": True}
        return {"metres_per_unit": self.metres_per_unit,
                "calibrated": self.calibrated}

    # --- the state -----------------------------------------------------------
    def state(self):
        """The whole contract payload. Never raises; errors surface as fields."""
        now = self._t()
        fleet_st = self.fleet.status()               # plain reads, already a copy
        ms = self.mapsvc.stats()                     # takes MapService.lock itself
        ls = self.loop.status() if self.loop is not None else None

        try:
            mp = self._map()
            map_err = ""
        except Exception as e:
            mp, map_err = None, f"{type(e).__name__}: {e}"
            self.last_error = map_err

        poses = (ls or {}).get("poses", {})
        modes = (ls or {}).get("mode", {})
        robots = []
        for name, s in fleet_st.items():
            cfg = self.fleet.cfg.get(name)
            p = poses.get(name) or [0.0, 0.0, 0.0, 0.0]
            mode = modes.get(name) or ("offline" if not s["online"] else "idle")
            robots.append({
                "id": name,
                "x": _f(p[0]), "y": _f(p[1]), "theta": _f(p[2]),
                "status": STATUS_OF_MODE.get(mode, "idle"),
                "mode": mode,
                "confidence": _f(p[3]) if len(p) > 3 else 0.0,
                "online": bool(s["online"]),
                "frames": int(s["frames"]),
                "camUrl": cfg.stream_url if cfg is not None else "",
            })

        self._note(fleet_st, ms, now)

        goal = (ls or {}).get("goal") if mp else None
        with self._lock:
            events = list(self._events)

        return {
            "t": round(now, 1),
            "source": SOURCE,
            "units": self._units(mp),
            "frame": mp["frame"] if mp else None,
            "grid": mp["grid"] if mp else None,
            "robots": robots,
            "boxes": mp["boxes"] if mp else None,
            "goal": [_f(goal[0]), _f(goal[1])] if goal else None,
            "mapService": {
                "rounds": int(ms["rounds"]),
                "submitted": int(ms["submitted"]),
                "registered": int(ms["registered"]),
                "rate": _f(ms["rate"]),
                "lastRoundS": _f(ms["last_round_s"]),
                "error": str(map_err or ms["error"] or (ls or {}).get("error") or ""),
            },
            "events": events,
        }

    def _note(self, fleet_st, ms, now):
        """Turn observable transitions into the event log. Only things that actually
        happened -- an event stream that guesses is worse than no event stream."""
        for name, s in fleet_st.items():
            was = self._seen_online.get(name)
            if was is not None and was != s["online"]:
                self.event(f"{name} {'online' if s['online'] else 'OFFLINE'}", now)
            self._seen_online[name] = s["online"]
        if ms["rounds"] > self._seen_rounds:
            self._seen_rounds = ms["rounds"]
            self.event(f"map round {ms['rounds']}: {ms['registered']}/{ms['submitted']} "
                       f"frames registered", now)
        with self._lock:
            key = self._cache_key
        if key and key != self._seen_map_key:
            first = self._seen_map_key is None
            self._seen_map_key = key
            self.event("first map available" if first else "map updated", now)

    def health(self):
        return {"ok": True, "t": round(self._t(), 1), "source": SOURCE,
                "port": self.port, "requests": self.requests,
                "robots": len(self.fleet.links),
                "map": self.mapsvc.map_path is not None,
                "driving": self.loop is not None,
                "error": self.last_error}


def _f(v, default=0.0):
    """A float the dashboard can parse. NaN and inf are not JSON, and one bad pose
    must not cost the whole poll its 200."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_NULL_LOCK = _NullLock()


def _make_handler(bridge):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "crab-live/1.0"

        # --- the rule: nothing escapes ------------------------------------
        def do_GET(self):
            try:
                bridge.requests += 1
                route = self.path.split("?", 1)[0].rstrip("/") or "/"
                if route == "/api/live/state":
                    self._json(200, bridge.state())
                elif route == "/api/live/health":
                    self._json(200, bridge.health())
                else:
                    self._json(404, {"error": f"no route {route}",
                                     "routes": ["/api/live/state", "/api/live/health"]})
            except Exception as e:
                try:
                    bridge.last_error = f"{type(e).__name__}: {e}"
                    self._json(500, {"error": f"{type(e).__name__}: {e}"})
                except Exception:
                    pass          # the socket is gone; the robots are not

        def do_OPTIONS(self):
            try:
                self.send_response(204)
                self._cors()
                self.send_header("Content-Length", "0")
                self.end_headers()
            except Exception:
                pass

        def do_HEAD(self):
            self.do_GET()

        # --- plumbing -----------------------------------------------------
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Max-Age", "86400")

        def _json(self, code, payload):
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def log_message(self, *a):
            """Silence. This shares stdout with a status board that redraws every
            second; 5 requests per second of access log makes it unreadable."""

    return Handler
