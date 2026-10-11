"""Is the download ceiling the link, the server, or our own worker count?

The main fetch is running at ~3.8 MB/s. Earlier measurements on this machine found the
phone hotspot delivering 3.60 MB/s on one path and 4.71-4.98 MB/s aggregated over two,
which points at the phone's mobile data rather than at Wi-Fi or at ModelScope. This
tests that directly instead of trusting the earlier number:

  1. measure how fast the WLAN interface is receiving while the main job runs
  2. add an independent load from a DIFFERENT host (hf-mirror), with its own connections
  3. measure the interface again

If the total does not rise, the ceiling is shared and it is the phone link. If it
rises, the main job was not using the link and more workers would help.
"""
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "Qwen/Qwen-Image-2.1"
FILE = "text_encoder/model-00003-of-00004.safetensors"   # not being downloaded yet
HF = f"https://hf-mirror.com/{REPO}/resolve/main/{FILE}"
HDR = {"User-Agent": "Mozilla/5.0 (compatible; cpu-forge-fetch)"}
WIN = 4 * 1024 * 1024
MB = 1024 * 1024


def wlan_recv_bytes():
    ps = (
        "[System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() | "
        "Where-Object { $_.OperationalStatus -eq 'Up' -and $_.Name -eq 'WLAN' } | "
        "ForEach-Object { $_.GetIPv4Statistics().BytesReceived }"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True, timeout=60).stdout.strip()
    return int(out.splitlines()[0]) if out else 0


def measure_baseline(seconds=12):
    a = wlan_recv_bytes()
    t0 = time.time()
    time.sleep(seconds)
    b = wlan_recv_bytes()
    dt = time.time() - t0
    return (b - a) / dt / MB


def fetch_range(start, length, stop, counter, lock):
    req = urllib.request.Request(
        HF, headers={**HDR, "Range": f"bytes={start}-{start + length - 1}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            while not stop.is_set():
                c = r.read(65536)
                if not c:
                    break
                with lock:
                    counter[0] += len(c)
    except Exception as exc:  # noqa: BLE001
        with lock:
            counter[1] = f"{type(exc).__name__}: {str(exc)[:60]}"


def main():
    print(f"host under test: hf-mirror ({FILE})")
    print("\n[1] baseline: what the interface receives while the main job runs")
    base = measure_baseline(12)
    print(f"    {base:.2f} MB/s")

    print("\n[2] adding 8 independent connections from hf-mirror for 25 s")
    stop = threading.Event()
    counter = [0, ""]
    lock = threading.Lock()
    a = wlan_recv_bytes()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fetch_range, i * WIN, WIN, stop, counter, lock)
                for i in range(8)]
        time.sleep(25)
        stop.set()
        for f in futs:
            f.result(timeout=90)
    dt = time.time() - t0
    b = wlan_recv_bytes()
    total = (b - a) / dt / MB
    print(f"    added load moved {counter[0]/MB:.1f} MB in {dt:.1f} s "
          f"= {counter[0]/MB/dt:.2f} MB/s")
    if counter[1]:
        print(f"    some connections failed: {counter[1]}")
    print(f"    interface total during the overlap: {total:.2f} MB/s")

    print("\n[3] verdict")
    print(f"    baseline {base:.2f} -> overlap {total:.2f} MB/s "
          f"({total/max(base,1e-9):.2f}x)")
    if total <= base * 1.15:
        print("    => the link is saturated: two independent hosts, 14 connections "
              "total, and the interface still moves the same number of bytes.")
    else:
        print("    => the link had headroom: the main job is not saturating it, so "
              "raising its worker count (or splitting across hosts) would help.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
