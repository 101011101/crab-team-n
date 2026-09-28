"""Everything wired together. One process, three threads.

    Fleet        N x (MJPEG in, /cmd out)      per robot, ~10 fps
    MapService   frames -> 3D -> 2D map        one round every ~25 s
    LiveLoop     map + pose -> track speeds    10 Hz

Ctrl-C stops every robot before exiting. So does any exception in any thread.

    python run_live.py --prior ../../out/world3/sparse --work ../../out/world3

--dry-run drives nothing. Use it first, every time: it proves video and the map
chain without a robot moving.
"""
import argparse, os, signal, sys, time

CLEAR = "\033[2J\033[H"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_source import Fleet
from map_service import MapService
from live_loop import LiveLoop
from bridge import TelemetryBridge


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", required=True, help="COLMAP sparse/ from the offline run")
    ap.add_argument("--work", required=True, help="dir holding db.db from the offline run")
    ap.add_argument("--map", default=None, help="map JSON to write and read")
    ap.add_argument("--every", type=float, default=25.0, help="seconds between map rounds")
    ap.add_argument("--batch", type=int, default=6, help="frames per map round")
    ap.add_argument("--dry-run", action="store_true", help="never send a drive command")
    ap.add_argument("--seconds", type=float, default=0, help="0 means run until Ctrl-C")
    ap.add_argument("--serve", type=int, default=8787, metavar="PORT",
                    help="port for the dashboard telemetry bridge")
    ap.add_argument("--no-serve", dest="serve", action="store_const", const=0,
                    help="do not serve telemetry at all")
    a = ap.parse_args()

    out_map = a.map or os.path.join(a.work, "live_map2d.json")
    fleet = Fleet().start()
    svc = MapService(a.work, a.prior, out_map, every=a.every, batch=a.batch)
    svc.start()
    loop = None
    if not a.dry_run:
        loop = LiveLoop(fleet, svc)
        loop.start()
    else:
        print("DRY RUN: no drive commands will be sent\n")

    # The bridge is started LAST and stopped FIRST: it only reads the other three,
    # and a dashboard poll must never be the thing that keeps a robot moving.
    t0 = time.time()
    bridge = None
    if a.serve:
        bridge = TelemetryBridge(fleet, svc, loop, port=a.serve, t0=t0)
        bridge.start()

    def bye(*_):
        print("\nstopping robots...")
        if bridge:
            bridge.stop()
        if loop:
            loop.stop_flag.set()
        svc.stop_flag.set()
        fleet.stop_all()
        time.sleep(0.3)
        fleet.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, bye)
    signal.signal(signal.SIGTERM, bye)

    last_submit = 0.0
    try:
        while True:
            # feed the map service; it drops what it cannot keep up with
            if time.time() - last_submit > 1.0:
                for name, (frame, ts) in fleet.frames().items():
                    svc.submit(name, frame)
                last_submit = time.time()

            st = fleet.status()
            ms = svc.stats()
            # Clearing the screen is for a terminal. Piped to a log it just makes
            # the file unreadable, so only do it when stdout is a tty.
            print((CLEAR if sys.stdout.isatty() else "\n") +
                  f" t+{time.time()-t0:6.1f}s")
            print(" ROBOTS")
            for n, s in st.items():
                print(f"   {n:9} {s['ip']:15} "
                      f"{'online ' if s['online'] else 'OFFLINE'} "
                      f"frames={s['frames']:6} cmd_fail={s['cmd_failures']:4} "
                      f"{s['stream_error'][:34]}")
            print(f"\n MAP SERVICE   rounds={ms['rounds']} "
                  f"submitted={ms['submitted']} registered={ms['registered']} "
                  f"(rate {ms['rate']})  last round {ms['last_round_s']}s")
            if ms["error"]:
                print(f"   last error: {ms['error'][:100]}")
            if ms["poses"]:
                print(f"   registered poses: {ms['poses']}")
            if ms["rate"] == 0 and ms["submitted"] > 6:
                print("   >> nothing has registered. Running on dead reckoning only.")
            if loop:
                ls = loop.status()
                print(f"\n CONTROL       ticks={ls['ticks']}  goal={ls['goal']}")
                for n, m in ls["mode"].items():
                    p = ls["poses"][n]
                    print(f"   {n:9} {m:12} pose=({p[0]:6.2f},{p[1]:6.2f}) "
                          f"conf={p[3]}")
                if ls["error"]:
                    print(f"   last error: {ls['error'][:100]}")
            else:
                print("\n CONTROL       dry run, not driving")
            if bridge:
                print(f"\n DASHBOARD     http://localhost:{a.serve}/api/live/state"
                      f"   ({bridge.requests} polls)"
                      + (f"  {bridge.last_error[:60]}" if bridge.last_error else ""))
            else:
                print("\n DASHBOARD     not serving (--no-serve)")
            print("\n Ctrl-C to stop every robot and exit.")

            if a.seconds and time.time() - t0 > a.seconds:
                bye()
            time.sleep(1.0)
    except Exception:
        import traceback
        traceback.print_exc()      # a silent exit here is impossible to debug
        bye()


if __name__ == "__main__":
    main()
