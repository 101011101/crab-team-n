"""Find the robots on whatever network you are actually on.

The firmware compiles a STATIC IP: 192.168.1.(50 + BOT_ID). That is fine on a router
you control and wrong on every hotspot, because a hotspot picks its own subnet:

    iPhone personal hotspot      172.20.10.x
    Android hotspot              192.168.43.x  (varies by vendor)
    macOS Internet Sharing       192.168.2.x
    most home routers            192.168.1.x   <- what the firmware assumes

So either the network hands out 192.168.1.x, or you change staticIP and gateway in
scout_bot.ino and reflash. This script tells you which robots are actually up and
prints the config.py block to paste, so you never guess an IP.

    python discover.py                 scan this laptop's subnet
    python discover.py --net 192.168.1 scan a specific one
"""
import argparse, concurrent.futures as cf, socket, sys

import requests


def my_subnet():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return ".".join(s.getsockname()[0].split(".")[:3])
    except Exception:
        return None
    finally:
        s.close()


def probe(ip, timeout=0.35):
    """A robot is something that answers /status with its own id."""
    try:
        r = requests.get(f"http://{ip}/status", timeout=timeout)
        if r.status_code == 200 and ("id" in r.text or "uptime" in r.text):
            return ip, r.text.strip()[:90]
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default=None, help="first three octets, e.g. 192.168.1")
    ap.add_argument("--timeout", type=float, default=0.35)
    a = ap.parse_args()

    net = a.net or my_subnet()
    if not net:
        sys.exit("no network. Join the hotspot first.")
    print(f"scanning {net}.1-254 for robots ...\n")

    found = []
    with cf.ThreadPoolExecutor(max_workers=128) as ex:
        for r in ex.map(lambda i: probe(f"{net}.{i}", a.timeout), range(1, 255)):
            if r:
                found.append(r)
                print(f"  FOUND {r[0]:15} {r[1]}")

    if not found:
        print("  nothing answered /status on this subnet.")
        print("\n  Check, in this order:")
        print("   1. Are the robots powered and their LEDs showing a Wi-Fi join?")
        print("   2. Does the firmware's WIFI_SSID match this hotspot exactly?")
        print("   3. Does the firmware's staticIP sit on THIS subnet?")
        print(f"      It is compiled as 192.168.1.5X; you are on {net}.x")
        print("   4. Is the hotspot isolating clients from each other? Many phone")
        print("      hotspots do. A cheap travel router does not.")
        return 1

    print("\npaste into central/config.py:\n")
    print("BOTS = [")
    for n, (ip, _) in enumerate(sorted(found), start=1):
        sector = "left" if n % 2 else "right"
        print(f'    BotConfig(bot_id={n}, name="scout-{n}", ip="{ip}", '
              f'sector="{sector}"),')
    print("]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
