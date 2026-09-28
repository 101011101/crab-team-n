"""Check every link before you trust it. Run this before run_live.py, every time.

Six checks, in dependency order. A later check cannot pass if an earlier one fails,
so fix them top down and do not skip.
"""
import argparse, os, socket, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "central"))
from config import BOTS          # noqa: E402

OK, BAD = "  ok  ", " FAIL "


def line(state, what, detail=""):
    print(f"[{state}] {what:42} {detail}")
    return state == OK


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None, help="offline run dir holding db.db")
    ap.add_argument("--prior", default=None, help="offline sparse/ dir")
    a = ap.parse_args()
    good = True

    # 1. this machine's address tells you which subnet the hotspot handed out
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        me = s.getsockname()[0]
    except Exception:
        me = "0.0.0.0"
    finally:
        s.close()
    my_net = ".".join(me.split(".")[:3])
    good &= line(OK, "this laptop", me)

    # 2. the robots' static IPs are compiled in. They must be on this subnet.
    bot_nets = {".".join(b.ip.split(".")[:3]) for b in BOTS}
    if my_net in bot_nets:
        line(OK, "laptop and robots share a subnet", my_net + ".x")
    else:
        good &= line(BAD, "SUBNET MISMATCH",
                     f"laptop {my_net}.x, robots {'/'.join(bot_nets)}.x")
        print("       The static IP is compiled into the firmware. Either put the")
        print("       robots and laptop on a network that hands out that subnet, or")
        print("       change staticIP/gateway in scout_bot.ino and reflash.")

    # 3. ping, 4. command port, 5. video port
    import requests
    for b in BOTS:
        r = subprocess.run(["ping", "-c", "1", "-W", "1000", b.ip],
                           capture_output=True)
        good &= line(OK if r.returncode == 0 else BAD, f"ping {b.name}", b.ip)
        if r.returncode != 0:
            continue
        try:
            t = requests.get(f"{b.command_url}/status", timeout=2).text[:60]
            line(OK, f"{b.name} command port 80", t)
        except Exception as e:
            good &= line(BAD, f"{b.name} command port 80", str(e)[:50])
        try:
            resp = requests.get(b.stream_url, stream=True, timeout=3)
            chunk = next(resp.iter_content(4096))
            resp.close()
            good &= line(OK if chunk else BAD, f"{b.name} video port 81",
                         f"{len(chunk)} bytes")
        except Exception as e:
            good &= line(BAD, f"{b.name} video port 81", str(e)[:50])

    # 6. the offline world the live system extends
    good &= line(OK if subprocess.run(["which", "colmap"],
                                      capture_output=True).returncode == 0 else BAD,
                 "colmap on PATH")
    if a.work:
        good &= line(OK if os.path.exists(os.path.join(a.work, "db.db")) else BAD,
                     "prior COLMAP database", os.path.join(a.work, "db.db"))
    if a.prior:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "pipeline"))
        try:
            import world_io
            w = world_io.load(a.prior)
            line(OK, "prior model loads",
                 f"{len(w.cam)} cameras, {len(w.P)} points, R={w.R:.2f}")
        except Exception as e:
            good &= line(BAD, "prior model loads", str(e)[:60])

    print("\n" + ("ALL CHECKS PASSED — safe to run run_live.py"
                  if good else "FIX THE FAILURES ABOVE BEFORE RUNNING run_live.py"))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
